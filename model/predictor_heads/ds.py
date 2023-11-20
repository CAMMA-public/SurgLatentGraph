from mmdet.registry import MODELS
from abc import ABCMeta
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptConfigType, InstanceList, OptMultiConfig
from mmengine.structures import BaseDataElement
from mmdet.structures import SampleList
from .modules.gnn import GNNHead
from .modules.layers import build_mlp, PositionalEncoding
from .modules.utils import *
from .modules.mstcn import MultiStageModel as MSTCN
import torch
from torch import Tensor
from torch_scatter import scatter_mean, scatter_max
import torch.nn.functional as F
from typing import List, Union, Tuple
import torch.nn.functional as F
import random
import dgl

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
            img_feat_key: str, img_feat_size: int, input_viz_feat_size: int,
            input_sem_feat_size: int, final_viz_feat_size: int, final_sem_feat_size: int,
            loss: Union[List, ConfigType], use_img_feats: bool = True, img_feats_only: bool = False,
            use_gnn_feats: bool = True, loss_consensus: str = 'mode', predictor_hidden_dim: int = 1024,
            num_predictor_layers: int = 2, loss_weight: float = 1.0, add_noise: bool = True,
            semantic_loss_weight: float = 0.0, viz_loss_weight: float = 0.0,
            img_loss_weight: float = 0.0, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        # set viz and sem dims for projecting node/edge feats in input graph
        self.input_sem_feat_size = input_sem_feat_size
        self.input_viz_feat_size = input_viz_feat_size
        self.final_sem_feat_size = final_sem_feat_size
        self.final_viz_feat_size = final_viz_feat_size
        self.img_feat_size = img_feat_size
        self.use_gnn_feats = use_gnn_feats
        if self.use_gnn_feats:
            projector_dim_list = [input_viz_feat_size * 2, final_viz_feat_size]
        else:
            projector_dim_list = [input_viz_feat_size, final_viz_feat_size]

        self.node_viz_feat_projector = build_mlp(projector_dim_list, batch_norm='batch')
        self.edge_viz_feat_projector = build_mlp(projector_dim_list, batch_norm='batch')

        self.node_sem_feat_projector = build_mlp([input_sem_feat_size, final_sem_feat_size],
                batch_norm='batch')
        self.edge_sem_feat_projector = build_mlp([input_sem_feat_size, final_sem_feat_size],
                batch_norm='batch')

        self.graph_feat_projected_dim = final_viz_feat_size + final_sem_feat_size

        # feature perturb
        self.add_noise = add_noise
        self.viz_loss_weight = viz_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.img_loss_weight = img_loss_weight

        # construct gnn
        gnn_cfg.input_dim_node = self.graph_feat_projected_dim
        gnn_cfg.input_dim_edge = self.graph_feat_projected_dim
        gnn_cfg.feat_key = 'feats' # use overall feats for gnn
        self.gnn = MODELS.build(gnn_cfg)

        # img feat params
        self.use_img_feats = use_img_feats
        self.img_feats_only = img_feats_only
        self.img_feat_key = img_feat_key
        self.img_feat_projector = build_mlp([img_feat_size, self.graph_feat_projected_dim],
                batch_norm='batch')#, final_nonlinearity=False)

        # norm layers
        self.img_graph_feat_fusion = build_mlp([self.graph_feat_projected_dim * 2, self.graph_feat_projected_dim],
                batch_norm='batch')

        # predictor, loss params
        predictor_dim = self.graph_feat_projected_dim

        if isinstance(loss, list):
            # losses
            self.loss_fn = torch.nn.ModuleList([MODELS.build(l) for l in loss])

            # predictors
            dim_list = [predictor_dim] + [predictor_hidden_dim] * (num_predictor_layers - 1)
            self.ds_predictor_head = build_mlp(dim_list)
            self.ds_predictor = torch.nn.ModuleList()
            for i in range(3): # separate predictor for each criterion
                self.ds_predictor.append(torch.nn.Linear(predictor_hidden_dim, num_classes))

        else:
            # loss
            self.loss_fn = MODELS.build(loss)

            # predictor
            dim_list = [predictor_dim] + [predictor_hidden_dim] * (num_predictor_layers - 1) + [num_classes]
            self.ds_predictor = build_mlp(dim_list, final_nonlinearity=False, batch_norm='batch')

        self.loss_weight = loss_weight
        self.loss_consensus = loss_consensus

    def predict(self, graph: BaseDataElement, feats: BaseDataElement) -> Tensor:
        # downproject graph feats
        node_feats = []
        edge_feats = []
        if self.final_viz_feat_size > 0:
            input_node_viz_feats = feats.instance_feats
            input_edge_viz_feats = graph.edges.viz_feats
            if self.use_gnn_feats:
                input_node_viz_feats = torch.cat([input_node_viz_feats, graph.nodes.gnn_viz_feats], -1)
                input_edge_viz_feats = torch.cat([input_edge_viz_feats, graph.edges.gnn_viz_feats], -1)

            # project node feats
            if input_node_viz_feats.flatten(end_dim=1).shape[0] == 1:
                node_viz_feats = self.node_viz_feat_projector(input_node_viz_feats.flatten(end_dim=1).repeat(2, 1))[0].view(
                        input_node_viz_feats.shape[0], input_node_viz_feats.shape[1], self.final_viz_feat_size)
            else:
                node_viz_feats = self.node_viz_feat_projector(input_node_viz_feats.flatten(end_dim=1)).view(
                        input_node_viz_feats.shape[0], input_node_viz_feats.shape[1], self.final_viz_feat_size)

            # project edge feats
            if input_edge_viz_feats.shape[0] == 1:
                edge_viz_feats = self.edge_viz_feat_projector(input_edge_viz_feats.repeat(2, 1))[0].unsqueeze(0)
            else:
                edge_viz_feats = self.edge_viz_feat_projector(input_edge_viz_feats)

            node_feats.append(node_viz_feats)
            edge_feats.append(edge_viz_feats)

        else:
            node_feats.append(None)
            edge_feats.append(None)

        if self.final_sem_feat_size > 0:
            input_node_sem_feats = feats.semantic_feats
            if input_node_sem_feats.flatten(end_dim=1).shape[0] == 1:
                node_sem_feats = self.node_sem_feat_projector(input_node_sem_feats.flatten(end_dim=1).repeat(2, 1))[0].view(
                        input_node_sem_feats.shape[0], input_node_sem_feats.shape[1], self.final_sem_feat_size)
            else:
                node_sem_feats = self.node_sem_feat_projector(input_node_sem_feats.flatten(end_dim=1)).view(
                        input_node_sem_feats.shape[0], input_node_sem_feats.shape[1], self.final_sem_feat_size)

            input_edge_sem_feats = graph.edges.semantic_feats
            if input_edge_sem_feats.shape[0] == 1:
                edge_sem_feats = self.edge_sem_feat_projector(input_edge_sem_feats.repeat(2, 1))[0].unsqueeze(0)
            else:
                edge_sem_feats = self.edge_sem_feat_projector(input_edge_sem_feats)

            node_feats.append(node_sem_feats)
            edge_feats.append(edge_sem_feats)

        else:
            node_feats.append(None)
            edge_feats.append(None)

        # get img feats
        img_feats = feats.bb_feats[-1] if self.img_feat_key == 'bb' else feats.fpn_feats[-1]
        if img_feats.shape[0] == 1:
            img_feats = self.img_feat_projector(F.adaptive_avg_pool2d(img_feats,
                1).squeeze(-1).squeeze(-1).repeat(2, 1))[0].unsqueeze(0)
        else:
            img_feats = self.img_feat_projector(F.adaptive_avg_pool2d(img_feats,
                1).squeeze(-1).squeeze(-1))

        perturbed_ds_preds = {}
        if self.semantic_loss_weight > 0 and self.final_sem_feat_size > 0:
            graph_sem_feats_only = self.feature_perturbation(node_feats, edge_feats, img_feats, 'sem')
            perturbed_ds_preds['graph_sem'] = self.forward(graph, *graph_sem_feats_only)
        if self.viz_loss_weight > 0 and self.final_viz_feat_size > 0:
            graph_viz_feats_only = self.feature_perturbation(node_feats, edge_feats, img_feats, 'viz')
            perturbed_ds_preds['graph_viz'] = self.forward(graph, *graph_viz_feats_only)
        if self.img_loss_weight > 0 and self.use_img_feats:
            img_feats_only = self.feature_perturbation(node_feats, edge_feats, img_feats, 'img')
            perturbed_ds_preds['img'] = self.forward(graph, *img_feats_only)

        # get rid of None
        node_feats = [f for f in node_feats if f is not None]
        edge_feats = [f for f in edge_feats if f is not None]

        if len(node_feats) == 0 or len(edge_feats) == 0:
            raise ValueError("Sum of final_viz_feat_size and final_sem_feat_size must be > 0")

        # run forward pass with all the components to get ds preds
        ds_preds = self.forward(graph, node_feats, edge_feats, img_feats)

        return ds_preds, perturbed_ds_preds

    def forward(self, graph, node_feats, edge_feats, img_feats):
        graph.nodes.feats = torch.cat(node_feats, -1)
        graph.edges.feats = torch.cat(edge_feats, -1)
        dgl_g = self.gnn(graph)

        # get node features and pool to get graph feats
        orig_node_feats = torch.cat([f[:npi] for f, npi in zip(graph.nodes.feats, graph.nodes.nodes_per_img)])
        node_feats = dgl_g.ndata['feats'] + orig_node_feats # skip connection
        npi_tensor = Tensor(graph.nodes.nodes_per_img).int()
        node_to_img = torch.arange(len(npi_tensor)).repeat_interleave(
                npi_tensor).long().to(node_feats.device)
        graph_feats = torch.zeros(npi_tensor.shape[0], node_feats.shape[-1]).to(node_feats.device)
        scatter_mean(node_feats, node_to_img, dim=0, out=graph_feats)

        # combine two types of feats
        if self.use_img_feats:
            pre_fusion_feat = torch.cat([img_feats, graph_feats], -1)
            if pre_fusion_feat.shape[0] == 1:
                final_feats = self.img_graph_feat_fusion(pre_fusion_feat.repeat(2, 1))[0].unsqueeze(0)
            else:
                final_feats = self.img_graph_feat_fusion(pre_fusion_feat)

        else:
            final_feats = graph_feats

        if isinstance(self.ds_predictor, torch.nn.ModuleList):
            ds_feats = self.ds_predictor_head(final_feats)
            ds_preds = torch.stack([p(ds_feats) for p in self.ds_predictor], 1)
        else:
            if final_feats.shape[0] == 1:
                ds_preds = self.ds_predictor(final_feats.repeat(2, 1))[0].unsqueeze(0)
            else:
                ds_preds = self.ds_predictor(final_feats)

        return ds_preds

    def feature_perturbation(self, node_feats, edge_feats, img_feats, keep_modality) -> Tuple[List]:
        perturbed_node_feats = [None for x in node_feats]
        perturbed_edge_feats = [None for x in edge_feats]
        if keep_modality == 'viz':
            perturb_img = True
            mode_to_drop = 1
        elif keep_modality == 'sem':
            perturb_img = True
            mode_to_drop = 0
        elif keep_modality == 'img':
            perturb_img = False

        if perturb_img:
            # mask img feats
            perturbed_img_feats = torch.normal(torch.zeros_like(img_feats),
                    torch.ones_like(img_feats)).to(img_feats.device)

            # mask one modality in both node and edge feats
            if node_feats[mode_to_drop] is not None:
                f = node_feats[mode_to_drop]
                perturbed_node_feats[mode_to_drop] = torch.normal(torch.zeros_like(f),
                        torch.ones_like(f)).to(f.device)

            if edge_feats[mode_to_drop] is not None:
                f = edge_feats[mode_to_drop]
                perturbed_edge_feats[mode_to_drop] = torch.normal(torch.zeros_like(f),
                        torch.ones_like(f)).to(f.device)
        else:
            # keep orig img feats
            perturbed_img_feats = img_feats

            # mask all node feats
            f = node_feats[0]
            perturbed_node_feats[0] = torch.normal(torch.zeros_like(f),
                    torch.ones_like(f)).to(f.device)
            f = node_feats[1]
            perturbed_node_feats[1] = torch.normal(torch.zeros_like(f),
                    torch.ones_like(f)).to(f.device)

            # mask all edge feats
            f = edge_feats[0]
            perturbed_edge_feats[0] = torch.normal(torch.zeros_like(f),
                    torch.ones_like(f)).to(f.device)
            f = edge_feats[1]
            perturbed_edge_feats[1] = torch.normal(torch.zeros_like(f),
                    torch.ones_like(f)).to(f.device)

        if self.add_noise:
            # add noise to graph_feats
            for i in range(len(node_feats)):
                if node_feats[i] is not None:
                    f = node_feats[i]
                    a = torch.normal(torch.ones_like(f), torch.ones_like(f)).to(f.device)
                    b = torch.normal(torch.zeros_like(f), torch.ones_like(f)).to(f.device)
                    perturbed_node_feats[i] = f * a + b # randomly scale and add noise

                if edge_feats[i] is not None:
                    f = edge_feats[i]
                    a = torch.normal(torch.ones_like(f), torch.ones_like(f)).to(f.device)
                    b = torch.normal(torch.zeros_like(f), torch.ones_like(f)).to(f.device)
                    perturbed_edge_feats[i] = f * a + b # randomly scale and add noise

            a = torch.normal(torch.ones_like(img_feats), torch.ones_like(img_feats)).to(img_feats.device)
            b = torch.normal(torch.zeros_like(img_feats), torch.ones_like(img_feats)).to(img_feats.device)
            perturbed_img_feats = img_feats * a + b # randomly scale and add noise

        perturbed_node_feats = [node_feats[i] if perturbed_node_feats[i] is None \
                else perturbed_node_feats[i] for i in range(len(node_feats))]
        perturbed_edge_feats = [edge_feats[i] if perturbed_edge_feats[i] is None \
                else perturbed_edge_feats[i] for i in range(len(edge_feats))]

        perturbed_node_feats = [f for f in perturbed_node_feats if f is not None]
        perturbed_edge_feats = [f for f in perturbed_edge_feats if f is not None]

        return perturbed_node_feats, perturbed_edge_feats, perturbed_img_feats

    def loss(self, graph: BaseDataElement, feats: BaseDataElement,
            batch_data_samples: SampleList) -> Tensor:
        ds_preds, perturbed_ds_preds = self.predict(graph, feats)
        ds_gt = torch.stack([torch.from_numpy(b.ds) for b in batch_data_samples]).to(ds_preds.device)

        if self.loss_consensus == 'mode':
            ds_gt = ds_gt.float().round().long()
        elif self.loss_consensus == 'prob':
            # interpret GT as probability of 1 and randomly generate gt
            random_probs = torch.rand_like(ds_gt) # random probability per label per example
            ds_gt = torch.le(random_probs, ds_gt).long().to(ds_gt.device)
        else:
            ds_gt = ds_gt.long()

        perturbed_ds_loss = {}
        if isinstance(self.loss_fn, torch.nn.ModuleList):
            # compute loss for each criterion and sum
            ds_loss = sum([self.loss_fn[i](ds_preds[:, i], ds_gt[:, i]) for i in range(len(self.loss_fn))]) / len(self.loss_fn)
            for k, v in perturbed_ds_preds.items():
                wt = self.viz_loss_weight if k == 'graph_viz' else self.semantic_loss_weight if k == 'graph_sem' else self.img_loss_weight
                loss_val = sum([self.loss_fn[i](v[:, i], ds_gt[:, i]) for i in range(len(self.loss_fn))]) / len(self.loss_fn)
                perturbed_ds_loss.update({'{}_ds_loss'.format(k): loss_val * wt * self.loss_weight})

        else:
            ds_loss = self.loss_fn(ds_preds, ds_gt)
            for k, v in perturbed_ds_preds.items():
                wt = self.viz_loss_weight if k == 'graph_viz' else self.semantic_loss_weight if k == 'graph_sem' else self.img_loss_weight
                loss_val = self.loss_fn(v, ds_gt)
                perturbed_ds_loss.update({'{}_ds_loss'.format(k): loss_val * wt * self.loss_weight})

        loss = {'ds_loss': ds_loss * self.loss_weight}
        loss.update(perturbed_ds_loss)

        return loss

@MODELS.register_module()
class STDSHead(DSHead):
    def __init__(self, num_temp_frames: int, graph_pooling_window: int = 1,
            use_temporal_model: bool = False, temporal_arch: str = 'transformer',
            pred_per_frame: bool = False, per_video: bool = False,
            use_node_positional_embedding: bool = True,
            use_positional_embedding: bool = False, edit_graph: bool = False,
            reassign_edges: bool = True, combine_nodes: bool = True,
            causal: bool = False, gnn_cfg: ConfigType = None,
            keep_all_instances: list = [0, 0, 0, 0, 0, 1], **kwargs) -> None:

        # set causal in gnn_cfg
        gnn_cfg.causal = causal
        self.gnn_cfg = gnn_cfg
        super().__init__(gnn_cfg=gnn_cfg, **kwargs)
        self.num_temp_frames = num_temp_frames
        self.use_temporal_model = use_temporal_model
        self.graph_pooling_window = graph_pooling_window
        self.pred_per_frame = pred_per_frame
        self.per_video = per_video
        self.edit_graph = edit_graph
        self.reassign_edges = reassign_edges
        self.combine_nodes = combine_nodes
        self.keep_all_instances = keep_all_instances
        self.causal = causal

        # positional embedding
        self.use_node_positional_embedding = use_node_positional_embedding
        self.use_positional_embedding = use_positional_embedding
        if self.use_node_positional_embedding:
            self.node_pe = PositionalEncoding(self.graph_feat_projected_dim,
                    batch_first=True, return_enc_only=True, dropout=0)

        if self.use_positional_embedding:
            self.pe = PositionalEncoding(self.img_feat_size, batch_first=True,
                    return_enc_only=True, dropout=0)

        # construct temporal model
        self.temporal_arch = temporal_arch
        if self.use_temporal_model:
            self.img_feat_temporal_model = self._create_temporal_model()
            self.img_feat_projector = torch.nn.Linear(2048, self.img_feat_projector.out_features)

    def predict(self, graph: BaseDataElement, feats: BaseDataElement) -> Tensor:
        # get dims
        B, T, N, _ = graph.nodes.feats.shape

        # downproject graph feats
        node_feats = []
        edge_feats = []
        if self.final_viz_feat_size > 0:
            node_viz_feats = self.node_viz_feat_projector(graph.nodes.feats[..., :self.input_viz_feat_size])
            edge_viz_feats = self.edge_viz_feat_projector(torch.cat(graph.edges.feats)[..., :self.input_viz_feat_size])
            node_feats.append(node_viz_feats)
            edge_feats.append(edge_viz_feats)

        if self.final_sem_feat_size > 0:
            node_sem_feats = self.node_sem_feat_projector(graph.nodes.feats[..., -self.input_sem_feat_size:])
            edge_sem_feats = self.edge_sem_feat_projector(torch.cat(graph.edges.feats)[..., -self.input_sem_feat_size:])
            node_feats.append(node_sem_feats)
            edge_feats.append(edge_sem_feats)

        if len(node_feats) == 0 or len(edge_feats) == 0:
            raise ValueError("Sum of final_viz_feat_size and final_sem_feat_size must be > 0")

        # add positional embedding to node feats
        node_feats = torch.cat(node_feats, -1)
        if self.use_node_positional_embedding:
            # use node_to_fic_id to arrange pos_embeds
            pos_embed = self.node_pe(torch.zeros(1, T, node_feats.shape[-1]).to(node_feats.device))

            # add to node_feats
            node_feats = F.dropout(node_feats + pos_embed.unsqueeze(2), 0.1,
                    training=self.training)

        graph.nodes.feats = node_feats
        graph.edges.feats = torch.cat(edge_feats, -1)
        dgl_g = self.gnn(graph)

        # edit graph
        if self.edit_graph:
            dgl_g, edited_nodes_per_img = self._edit_graph(dgl_g, graph.nodes.nodes_per_img)
            graph.nodes.nodes_per_img = edited_nodes_per_img

        # get node features and pool to get graph feats
        node_feats = dgl_g.ndata['feats'] + dgl_g.ndata['orig_feats'] # skip connection

        # pool node feats by img
        node_to_img = torch.cat([ind * T + torch.arange(T).repeat_interleave(n.int()).long().to(
            node_feats.device) for ind, n in enumerate(graph.nodes.nodes_per_img)])
        graph_feats = torch.zeros(B * T, node_feats.shape[-1]).to(node_feats.device)
        scatter_mean(node_feats, node_to_img, dim=0, out=graph_feats)

        # combine two types of feats
        if self.use_img_feats:
            # get img feats
            img_feats = feats.bb_feats[-1] if self.img_feat_key == 'bb' else feats.fpn_feats[-1]
            img_feats = F.adaptive_avg_pool2d(img_feats, 1).squeeze(-1).squeeze(-1)

            if self.use_temporal_model:
                if self.temporal_arch == 'tcn':
                    tcn_output = self.img_feat_temporal_model(img_feats.permute(0, 2, 1))
                    img_feats = tcn_output.sum(0).permute(0, 2, 1)
                else:
                    img_feats = self.img_feat_temporal_model(img_feats)

            elif self.use_positional_embedding:
                pos_embed = self.pe(torch.zeros(1, T, img_feats.shape[-1]).to(img_feats.device))
                img_feats = F.dropout(img_feats + pos_embed, 0.1, training=self.training).mean(1, keepdims=True)

            img_feats = self.img_feat_projector(img_feats)
            if self.img_feats_only:
                final_feats = img_feats
            else:
                #final_feats = torch.cat([img_feats, graph_feats.view(B, T, -1)], -1)
                final_feats = img_feats + graph_feats.view(B, T, -1)

        else:
            final_feats = graph_feats.view(B, T, -1)

        # 2 modes: 1 prediction per clip for clip classification, or output per-keyframe for
        # whole-video inputs

        # pred-per-frame handles the second case, but can also apply to clip classification
        # during training, in case we still want per-frame output for clip classification
        if self.pred_per_frame:
            ds_preds = self._ds_predict(final_feats)

            if not self.training and not self.per_video:
                # keep only prediction for last frame in clip
                ds_preds = ds_preds[:, -1]

        else:
            if self.graph_pooling_window == -1:
                self.graph_pooling_window = T # keep all frame feats

            # pool based on pooling window
            final_feats = final_feats[:, -self.graph_pooling_window:].mean(1)
            ds_preds = self._ds_predict(final_feats)

        return ds_preds

    def loss(self, graph: BaseDataElement, feats: BaseDataElement,
            batch_data_samples: SampleList) -> Tensor:
        ds_preds = self.predict(graph, feats)
        ds_gt = torch.stack([torch.stack([torch.from_numpy(b.ds) for b in vds]) \
                for vds in batch_data_samples]).to(ds_preds.device)

        # preprocess gt
        if self.loss_consensus == 'mode':
            ds_gt = ds_gt.float().round().long()
        else:
            ds_gt = ds_gt.long()

        # TODO handle case when loss_fn is module list (multi-task ds head)
        # reshape preds and gt according to prediction settings
        if not self.pred_per_frame:
            # keep only last gt per clip
            ds_gt = ds_gt[:, -1]
        else:
            ds_preds = ds_preds.flatten(end_dim=1)
            ds_gt = ds_gt.flatten(end_dim=1)

        if isinstance(self.loss_fn, torch.nn.ModuleList):
            # compute loss for each criterion and sum
            ds_loss = sum([self.loss_fn[i](ds_preds[:, i], ds_gt[:, i]) for i in range(len(self.loss_fn))]) / len(self.loss_fn)

        else:
            ds_loss = self.loss_fn(ds_preds, ds_gt)

        loss = {'ds_loss': ds_loss * self.loss_weight}

        return loss

    def _create_temporal_model(self):
        if self.temporal_arch.lower() == 'transformer':
            pe = PositionalEncoding(d_model=2048, batch_first=True)
            decoder_layer = torch.nn.TransformerDecoderLayer(d_model=2048, nhead=8,
                    batch_first=True, dropout=0)
            temp_model = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)
            if self.graph_pooling_window != -1:
                model = CustomSequential(pe, DuplicateItem(), temp_model)
            else:
                model = CustomSequential(pe, DuplicateItem(), temp_model,
                        torch.nn.MaxPool2d((self.num_temp_frames, 1)),
                        SqueezeItem(1))

        elif self.temporal_arch.lower() == 'tcn':
            model = MSTCN(2, 8, 32, 2048, 2048, self.causal)

        else:
            if self.temporal_arch.lower() == 'gru':
                temp_model = torch.nn.GRU(2048, 2048, 2, batch_first=True, dropout=0)
            elif self.temporal_arch.lower() == 'lstm':
                temp_model = torch.nn.LSTM(2048, 2048, 2, batch_first=True, dropout=0)
            else:
                raise NotImplementedError("Temporal architecture " + self.temporal_arch + " not implemented.")

            if self.graph_pooling_window != -1:
                model = torch.nn.Sequential(temp_model, SelectItem(0))
            else:
                model = torch.nn.Sequential(temp_model, SelectItem(0),
                        torch.nn.MaxPool2d((self.num_temp_frames, 1)),
                        SqueezeItem(1))

        return model

    def _ds_predict(self, final_feats):
        if isinstance(self.ds_predictor, torch.nn.ModuleList):
            ds_feats = self.ds_predictor_head(final_feats)
            ds_preds = torch.stack([p(ds_feats) for p in self.ds_predictor], 1)
        else:
            if final_feats.ndim > 2:
                ds_preds = self.ds_predictor(final_feats.view(-1, *final_feats.shape[2:]))
                ds_preds = ds_preds.view(*final_feats.shape[:2], -1)
            else:
                ds_preds = self.ds_predictor(final_feats)

        return ds_preds

    def _edit_graph(self, batched_graph, nodes_per_img):
        if (self.training and random.random() > 0.5):
            return batched_graph, nodes_per_img

        # split graph into clip graphs
        node_labels = batched_graph.ndata['labels']
        node_features = batched_graph.ndata['feats']
        nodes_per_clip = batched_graph.batch_num_nodes()
        edges_per_clip = batched_graph.batch_num_edges()
        edge_flats = torch.stack(batched_graph.edges(), -1)
        clip_graphs = dgl.unbatch(batched_graph)

        # compute node degrees and split by frame
        node_degrees_frame = [(g.in_degrees() + g.out_degrees()).split(npi.int().tolist()) for g, npi in zip(clip_graphs, nodes_per_img)]

        # split labels, node_degrees by frame
        node_labels_frame = [g.split(npi.int().tolist()) for g, npi in zip(node_labels.long().split(nodes_per_clip.tolist()), nodes_per_img)]

        # compute the instance of each class that has max degree in each frame
        max_degree_inds = [[scatter_max(d, l)[1] for d, l in zip(ndf, nlf)] for ndf, nlf in zip(node_degrees_frame, node_labels_frame)]

        # compute the node that each node was reduced to (either the same node or maps to another node of the same class)
        node_reassignment_map = [[m[c] for m, c in zip(mc, nc)] for mc, nc in zip(max_degree_inds, node_labels_frame)]

        # if we want to keep all instances of an object class, set that
        keep_instance_inds = (node_labels.unsqueeze(-1) == (torch.tensor(self.keep_all_instances).to(node_labels) - 1)).any(-1)
        keep_instance_inds_frame = [x.split(npi.int().tolist()) for x, npi in zip(keep_instance_inds.split(
            nodes_per_clip.tolist()), nodes_per_img)]

        # get the indices where the actual degree matches the max degree
        inds_to_keep_frame = [[torch.cat([m[m < len(l)], torch.where(k)[0]]).unique() for d, l, k, m in zip(
            ndf, nlf, kiif, md)] for ndf, nlf, kiif, md in zip(node_degrees_frame,
                    node_labels_frame, keep_instance_inds_frame, max_degree_inds)]

        # inds_to_keep_frame -> inds_to_keep
        frame_offsets = [torch.tensor([0] + npi.int()[:-1].tolist()).cumsum(0) for npi in nodes_per_img]
        clip_offsets = torch.tensor([0] + [sum(npi.int()) for npi in nodes_per_img[:-1]]).cumsum(0)
        inds_to_keep = [[i + o + co for i, o in zip(clip_inds, fo)] for clip_inds, fo, co in zip(
            inds_to_keep_frame, frame_offsets, clip_offsets)]

        # flatten node reassignment map and add offsets
        node_reassignment_map = torch.cat([torch.cat([n + o + co for n, o in zip(nrm, fo)]) for nrm, fo, co in zip(
            node_reassignment_map, frame_offsets, clip_offsets)])

        if self.combine_nodes and (not self.training or random.random() > 0.5): # always edit in eval, random in train
            # use node reassignment map to combine node features, set in batched graph
            updated_node_features = scatter_mean(node_features, node_reassignment_map,
                    dim=0, dim_size=node_features.shape[0])
            batched_graph.ndata['feats'] = updated_node_features

        reassign_edges = False
        if self.reassign_edges and (not self.training or random.random() > 0.5): # always edit in eval, random in train
            reassign_edges = True
            # use node reassignment map to edit edge flats
            updated_edge_flats = node_reassignment_map[edge_flats[edge_flats[:, 0].sort().indices]]

            # get inds where edge was updated
            updated_inds = torch.where((edge_flats[edge_flats[:, 0].sort().indices] != updated_edge_flats).any(-1))[0]

            # of these inds, select unique inds, filter updated_edge_flats
            _, idx, counts = updated_edge_flats[updated_inds].unique(dim=0,
                    return_counts=True, return_inverse=True)
            unique_inds = updated_inds[torch.where(counts[idx] <= 1)] # stores the id of the original edge from which we want to copy the edge data
            updated_edge_flats = updated_edge_flats[unique_inds]

            # get remaining edge data, filter with unique_inds
            updated_edge_data = {}
            for k in batched_graph.edata.keys():
                updated_edge_data.update({k: batched_graph.edata[k][unique_inds]})

            # add edges to graph
            batched_graph = dgl.add_edges(batched_graph, updated_edge_flats[:, 0],
                    updated_edge_flats[:, 1], data=updated_edge_data)

            # remove self loops that may have been added
            batched_graph = batched_graph.remove_self_loop()
            if self.gnn_cfg.add_self_loops:
                batched_graph = batched_graph.add_self_loops()

        # now only keep inds_to_keep in batched_graph, recompute nodes_per_img
        edited_nodes_per_img = [Tensor([len(x) for x in i]) for i in inds_to_keep]
        inds_to_keep_tensor = torch.cat([torch.cat(i) for i in inds_to_keep])
        edited_batched_graph = batched_graph.subgraph(inds_to_keep_tensor)

        # finally update batch nodes per img and batch edges per img
        batch_num_nodes = torch.tensor([sum(x) for x in edited_nodes_per_img])
        if not reassign_edges:
            # compute batch num edges
            edges_to_keep = (edge_flats.unsqueeze(-1) == inds_to_keep_tensor).any(-1).all(-1)
            batch_num_edges = torch.tensor([x.sum() for x in edges_to_keep.split(edges_per_clip.tolist())])
        else:
            # compute batch_num_edges
            all_edge_flats = torch.stack(batched_graph.edges(), -1)
            edge_to_clip = (all_edge_flats.unsqueeze(-1) >= batch_num_nodes.to(all_edge_flats.device).cumsum(0)).sum(-1).max(-1).values
            _, batch_num_edges = edge_to_clip.unique(sorted=True, return_counts=True)

        edited_batched_graph.set_batch_num_nodes(batch_num_nodes)
        edited_batched_graph.set_batch_num_edges(batch_num_edges)

        return edited_batched_graph, edited_nodes_per_img
