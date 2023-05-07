from mmdet.registry import MODELS
from abc import ABCMeta
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType, InstanceList, OptMultiConfig
from mmengine.structures import BaseDataElement
from mmdet.structures import SampleList
from .modules.gnn import GNNHead
from .modules.layers import build_mlp
import torch
from torch import Tensor
from torch_scatter import scatter_mean
import torch.nn.functional as F
from typing import List

@MODELS.register_module()
class DSHead(BaseModule, metaclass=ABCMeta):
    """DS Head to predict downstream task from graph

    Args:
        num_classes (int)
        gnn_cfg (ConfigType): gnn cfg
        img_feat_key (str): use bb feats or fpn feats for img-level features
        graph_feat_input_dim (int): node and edge feat dim in graph structure
        graph_feat_projected_dim (int): node and edge feat dim to use in gnn
        loss (str): loss fn for ds (default: BCELoss)
        loss_consensus (str): how to deal with multiple annotations for ds task (default: mode)
        weight (List): per-class loss weight for ds (default: None)
        loss_weight: multiplier for ds loss
    """
    def __init__(self, num_classes: int, gnn_cfg: ConfigType,
            img_feat_key: str, img_feat_size: int, graph_feat_input_dim: int,
            graph_feat_projected_dim: int, loss: str, loss_weight: float,
            prediction_mode='ml', loss_consensus: str = 'mode', weight: List = None,
            num_predictor_layers: int = 2, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        # project node/edge feats in input graph
        self.graph_node_feat_projector = torch.nn.Linear(graph_feat_input_dim,
                graph_feat_projected_dim)
        self.graph_edge_feat_projector = torch.nn.Linear(graph_feat_input_dim,
                graph_feat_projected_dim)

        # construct gnn
        gnn_cfg.input_dim_node = graph_feat_projected_dim
        gnn_cfg.input_dim_edge = graph_feat_projected_dim
        self.gnn = MODELS.build(gnn_cfg)

        # img feat params
        self.img_feat_key = img_feat_key
        self.img_feat_projector = torch.nn.Linear(img_feat_size, graph_feat_projected_dim)

        # predictor params
        self.prediction_mode = prediction_mode
        if self.prediction_mode == 'ml':
            dim_list = [gnn_cfg.input_dim_node] * num_predictor_layers + [num_classes]
            self.ds_predictor = build_mlp(dim_list, final_nonlinearity=False)
        elif self.prediction_mode == 'mlmc':
            dim_list = [gnn_cfg.input_dim_node] * num_predictor_layers
            self.ds_predictor_head = build_mlp(dim_list)
            self.ds_predictor = torch.nn.ModuleList()
            for i in range(3): # separate predictor for each criterion
                self.ds_predictor.append(torch.nn.Linear(gnn_cfg.input_dim_node, 3)) # TODO(adit98) set this as a param

        # loss params
        self.loss_consensus = loss_consensus
        if loss == 'bce':
            if self.prediction_mode == 'mlmc':
                self.loss_fn = torch.nn.ModuleList([torch.nn.CrossEntropyLoss(weight=Tensor(weight[i])) \
                        for i in range(3)])

            else:
                self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=Tensor(weight))

        else:
            raise NotImplementedError

        self.loss_weight = loss_weight

    def predict(self, graph: BaseDataElement, feats: BaseDataElement) -> Tensor:
        # downproject graph feats
        graph.nodes.feats = self.graph_node_feat_projector(graph.nodes.feats)
        graph.edges.feats = self.graph_edge_feat_projector(graph.edges.feats)
        dgl_g = self.gnn(graph)

        # get node features and pool to get graph feats
        orig_node_feats = torch.cat([f[:npi] for f, npi in zip(graph.nodes.feats, graph.nodes.nodes_per_img)])
        node_feats = dgl_g.ndata['feats'] + orig_node_feats # skip connection
        npi_tensor = Tensor(graph.nodes.nodes_per_img).int()
        node_to_img = torch.arange(len(npi_tensor)).repeat_interleave(
                npi_tensor).long().to(node_feats.device)
        graph_feats = torch.zeros(npi_tensor.shape[0], node_feats.shape[-1]).to(node_feats.device)
        scatter_mean(node_feats, node_to_img, dim=0, out=graph_feats)

        # get img feats
        img_feats = feats.bb_feats[-1] if self.img_feat_key == 'bb' else feats.fpn_feats[-1]
        img_feats = self.img_feat_projector(F.adaptive_avg_pool2d(img_feats,
            1).squeeze(-1).squeeze(-1))

        # combine two types of feats
        final_feats = img_feats + graph_feats
        if isinstance(self.ds_predictor, torch.nn.ModuleList):
            ds_feats = self.ds_predictor_head(final_feats)
            ds_preds = torch.stack([p(ds_feats) for p in self.ds_predictor])
        else:
            ds_preds = self.ds_predictor(final_feats)

        return ds_preds

    def loss(self, graph: BaseDataElement, feats: BaseDataElement,
            batch_data_samples: SampleList) -> Tensor:
        ds_preds = self.predict(graph, feats)

        ds_gt = torch.stack([torch.from_numpy(b.ds) for b in batch_data_samples]).to(ds_preds.device)
        if self.loss_consensus == 'mode':
            ds_gt = ds_gt.float().round().long()

        if isinstance(self.loss_fn, torch.nn.ModuleList):
            # compute loss for each criterion and sum
            ds_loss = sum([self.loss_fn[i](ds_preds[i], ds_gt[:, i]) for i in range(len(self.loss_fn))]) / len(self.loss_fn)

        else:
            ds_loss = self.loss_fn(ds_preds, ds_gt)

        loss = {'ds_loss': ds_loss * self.loss_weight}

        return loss
