from mmdet.registry import MODELS
from abc import ABCMeta
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule
from mmdet.structures import SampleList
from mmengine.structures import BaseDataElement
from typing import List, Tuple, Union
import torch
from torch import Tensor
import torch.nn.functional as F
import dgl
from .gnn_models import GraphTripleConvNet

@MODELS.register_module()
class GNNHead(BaseModule, metaclass=ABCMeta):
    """GNN module that takes a graph (BaseDataElement) object and processes the
    node and edge features.

    Args:
        num_layers (int)
        arch (str)
        add_self_loops (bool)
        norm (str)
        skip_connect (bool)
        viz_feat_size (int)
        semantic_feat_size (int)
    """
    def __init__(self, num_layers: int, arch: str, add_self_loops: bool, use_reverse_edges: bool,
            norm: str, skip_connect: bool, input_dim_node: int, input_dim_edge: int,
            hidden_dim: int = 512, dropout: float = 0.0,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.add_self_loops = add_self_loops
        self.use_reverse_edges = use_reverse_edges
        if 'tripleconv' in arch.lower():
            self.gnn_head = GraphTripleConvNet(input_dim_node, input_dim_edge,
                    hidden_dim=hidden_dim, num_layers=num_layers, mlp_normalization=norm,
                    skip_connect=skip_connect, dropout=dropout, use_net2=False,
                    use_edges=True, final_nonlinearity=False)
        else:
            raise NotImplementedError

    def __call__(self, graph: BaseDataElement) -> BaseDataElement:
        # construct dgl graph, deal with reverse edges, self loops
        dgl_g = self._create_dgl_graph(graph)

        # apply gnn
        node_feats, edge_feats = self.gnn_head(dgl_g.ndata['feats'],
                dgl_g.edata['feats'], torch.stack(dgl_g.edges(), 1), dgl_g)

        dgl_g.ndata['feats'] = node_feats
        dgl_g.edata['feats'] = edge_feats

        return dgl_g

    def _create_dgl_graph(self, graph: BaseDataElement) -> dgl.DGLGraph:
        # convert edge flats to batch edge flats
        edge_offsets = torch.cumsum(Tensor([0] + graph.nodes.nodes_per_img[:-1]), 0).to(graph.edges.edge_flats.device)
        batch_edge_flats = graph.edges.edge_flats[:, 1:] + edge_offsets[graph.edges.edge_flats[:, 0]].view(-1, 1).int()
        g = dgl.graph(batch_edge_flats.unbind(1), num_nodes=sum(graph.nodes.nodes_per_img))

        # add attributes to graph
        for k, v in graph.nodes.items():
            if k == 'nodes_per_img': continue
            g.ndata[k] = torch.cat([v_i[:n] for n, v_i in zip(graph.nodes.nodes_per_img, v)])

        for k, v in graph.edges.items():
            skip_keys = ['edges_per_img', 'batch_index', 'edge_flats', 'presence_logits', 'boxesA', 'boxesB']
            if k in skip_keys: continue
            if isinstance(v, tuple):
                v = torch.cat(v)

            g.edata[k] = v.view(-1, v.shape[-1])

        # add in batch info
        g.set_batch_num_nodes(Tensor(graph.nodes.nodes_per_img))
        g.set_batch_num_edges(graph.edges.edges_per_img)

        # add self loops
        g = g.remove_self_loop()
        if self.add_self_loops:
            g = g.add_self_loop()

        # remove reverse edges
        reverse_eids = torch.where((torch.stack(g.edges(), 1) == \
                torch.stack(dgl.reverse(g).edges(), 1)).all(1))[0]
        g = dgl.remove_edges(g, reverse_eids)

        # add reverse edges if specified
        if self.use_reverse_edges:
            g = dgl.add_reverse_edges(g, copy_edata=True)

        return g
