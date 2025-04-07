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
import random

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
            causal: bool = False, hidden_dim: int = 512, dropout: float = 0.0,
            feat_key: str = 'viz_feats', init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.add_self_loops = add_self_loops
        self.use_reverse_edges = use_reverse_edges
        if 'tripleconv' in arch.lower():
            self.gnn_head = GraphTripleConvNet(input_dim_node, input_dim_edge,
                    hidden_dim=hidden_dim, num_layers=num_layers, mlp_normalization=norm,
                    skip_connect=skip_connect, dropout=dropout, use_net2=False,
                    use_edges=True, final_nonlinearity=False, causal=causal)
        else:
            raise NotImplementedError

        # which feature from graph structure to apply gnn on
        self.feat_key = feat_key

    def __call__(self, graph: BaseDataElement) -> BaseDataElement:
        # construct dgl graph, deal with reverse edges, self loops
        dgl_g = self._create_dgl_graph(graph)

        # apply gnn
        node_feats, edge_feats = self.gnn_head(dgl_g.ndata[self.feat_key],
                dgl_g.edata[self.feat_key], torch.stack(dgl_g.edges(), 1), dgl_g)

        dgl_g.ndata['gnn_feats'] = node_feats
        dgl_g.edata['gnn_feats'] = edge_feats

        return dgl_g

    def _create_dgl_graph(self, graph: BaseDataElement) -> dgl.DGLGraph:
        # convert edge flats to batch edge flats
        if isinstance(graph.nodes.nodes_per_img[0], Tensor):
            device = graph.edges.edge_flats[0].device

            # need to compute edge offsets by first adding offset for each img in clip then
            # each clip in batch
            per_clip_edge_offsets = [torch.cat([torch.cumsum(Tensor([0] + npi[:-1].tolist()).to(
                device), 0), torch.zeros(1).to(device)]) for npi in graph.nodes.nodes_per_img]
            per_clip_edge_flats = [torch.cat([ef[:, 0:1], ef[:, 1:] + pcef[ef[:, 0]].view(-1, 1).int()],
                dim=1) for pcef, ef in zip(per_clip_edge_offsets, graph.edges.edge_flats)]

            # add batch id to each edge flat (so we have batch_id, img_id, edge_x, edge_y)
            batch_edge_flats = torch.cat([torch.cat([torch.ones_like(pcef[:, 0:1]) * ind,
                pcef], dim=1) for ind, pcef in enumerate(per_clip_edge_flats)])

            # compute offsets per batch and add to batch_edge_flats
            nodes_per_clip = [sum(x) for x in graph.nodes.nodes_per_img]
            batch_edge_offsets = torch.cumsum(Tensor([0] + nodes_per_clip[:-1]), 0).to(device)
            batch_edge_flats[:, -2:] += batch_edge_offsets[batch_edge_flats[:, 0]].view(-1, 1).int()

            # create dgl graph
            g = dgl.graph(batch_edge_flats[:, -2:].unbind(1), num_nodes=sum(nodes_per_clip))

            # add attributes to graph
            for k, v in graph.nodes.items():
                skip_keys = ['nodes_per_img', 'viz_feats', 'gnn_viz_feats', 'instance_feats']
                if k in skip_keys: continue

                # for each img in each clip, remove padded nodes and concatenate all values for batch of clips
                if torch.stack(graph.nodes.nodes_per_img).sum() > 0:
                    g.ndata[k] = torch.cat([torch.cat([v_i[:n] for v_i, n in zip(cv,
                        npi_i.int())]) for cv, npi_i in zip(v, graph.nodes.nodes_per_img)])
                else:
                    g.ndata[k] = torch.zeros(0, v.shape[-1]).to(v.device)

            for k, v in graph.edges.items():
                skip_keys = ['edges_per_img', 'edges_per_clip', 'batch_index', 'edge_flats',
                        'presence_logits', 'gnn_viz_feats', 'viz_feats', 'semantic_feats']
                if k in skip_keys: continue
                if isinstance(v, tuple) or isinstance(v, list):
                    v = torch.cat(v)

                g.edata[k] = v.view(-1, v.shape[-1])

            # add in batch info
            g.set_batch_num_nodes(Tensor(nodes_per_clip).int().to(device))
            g.set_batch_num_edges(Tensor(graph.edges.edges_per_clip).int().to(device))

        else:
            edge_offsets = torch.cumsum(Tensor([0] + graph.nodes.nodes_per_img[:-1]), 0).to(graph.edges.edge_flats.device)
            batch_edge_flats = graph.edges.edge_flats[:, 1:] + edge_offsets[graph.edges.edge_flats[:, 0]].view(-1, 1).int()
            g = dgl.graph(batch_edge_flats.unbind(1), num_nodes=sum(graph.nodes.nodes_per_img))

            # add attributes to graph
            for k, v in graph.nodes.items():
                skip_keys = ['nodes_per_img']
                if k in skip_keys: continue
                g.ndata[k] = torch.cat([v_i[:n] for n, v_i in zip(graph.nodes.nodes_per_img, v)])

            for k, v in graph.edges.items():
                skip_keys = ['edges_per_img', 'edges_per_clip', 'batch_index', 'edge_flats', 'presence_logits']
                if k in skip_keys: continue
                if isinstance(v, tuple):
                    v = torch.cat(v)

                g.edata[k] = v.view(-1, v.shape[-1])

            # add in batch info
            # g.set_batch_num_nodes(Tensor(graph.nodes.nodes_per_img))
            # g.set_batch_num_nodes(torch.tensor(graph.nodes.nodes_per_img, dtype=torch.int64))
            g.set_batch_num_nodes(torch.tensor(graph.nodes.nodes_per_img, dtype=torch.int64).to(g.device))
            # g.set_batch_num_edges(graph.edges.edges_per_img)
            g.set_batch_num_edges(torch.tensor(graph.edges.edges_per_img, dtype=torch.int64).to(g.device))


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
