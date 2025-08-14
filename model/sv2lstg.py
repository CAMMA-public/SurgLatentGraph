import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from typing import List, Tuple, Union
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from mmdet.structures import SampleList, OptSampleList, DetDataSample
from mmdet.structures.bbox import bbox2roi, roi2bbox, scale_boxes
from mmdet.models.detectors.base import BaseDetector
from mmengine.structures import BaseDataElement, InstanceData
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
from .lg import LGDetector
from .predictor_heads.modules.layers import build_mlp
from .predictor_heads.modules.utils import apply_sparse_mask, get_sparse_mask_inds

@MODELS.register_module()
class SV2LSTG(BaseDetector):
    def __init__(self, lg_detector: BaseDetector, ds_head: ConfigType, clip_size: int,
            viz_feat_size: int, semantic_feat_size: int, reencode_semantics: bool = False,
            use_gnn_feats: bool = True, num_spatial_edge_classes: int = 3,
            use_spat_graph: bool = True, use_viz_graph: bool = True, learn_sim_graph: bool = False,
            sem_feat_hidden_dim: int = 2048, semantic_feat_projector_layers: int = 3,
            sem_feat_use_bboxes: bool = True, sem_feat_use_class_logits: bool = True,
            sem_feat_use_masks: bool = False, sem_feat_use_temporal_window: bool = True,
            num_sim_topk: int = 2, temporal_edge_ranges: str = 'exp', edge_max_temporal_range: int = -1,
            use_max_iou_only: bool = True, use_temporal_edges_only: bool = False,
            per_video: bool = False, **kwargs):
        super().__init__(**kwargs)

        # init lg detector
        self.lg_detector = MODELS.build(lg_detector)
        self.clip_size = clip_size

        # viz feature fusion
        self.use_gnn_feats = use_gnn_feats
        if self.use_gnn_feats:
            projector_dim_list = [lg_detector.viz_feat_size * 2, viz_feat_size]
        else:
            projector_dim_list = [lg_detector.viz_feat_size, viz_feat_size]

        self.node_viz_feat_projector = build_mlp(projector_dim_list, batch_norm='batch')
        self.edge_viz_feat_projector = build_mlp(projector_dim_list, batch_norm='batch')

        # visual similarity edge kernel functions
        self.sim_embed1 = torch.nn.Linear(viz_feat_size + semantic_feat_size,
                viz_feat_size + semantic_feat_size, bias=False)
        self.sim_embed2 = torch.nn.Linear(viz_feat_size + semantic_feat_size,
                viz_feat_size + semantic_feat_size, bias=False)

        # set extra params
        self.num_spatial_edge_classes = num_spatial_edge_classes
        self.use_spat_graph = use_spat_graph
        self.use_viz_graph = use_viz_graph
        self.learn_sim_graph = learn_sim_graph
        self.viz_feat_size = viz_feat_size
        self.semantic_feat_size = semantic_feat_size
        self.reencode_semantics = reencode_semantics

        self.num_temp_edge_classes = 0
        if self.use_viz_graph:
            self.num_temp_edge_classes += 1
        if self.use_spat_graph:
            self.num_temp_edge_classes += 1

        # edge semantic feat projector
        sem_input_dim = 0
        self.sem_feat_use_bboxes = sem_feat_use_bboxes
        self.sem_feat_use_class_logits = sem_feat_use_class_logits
        self.sem_feat_use_masks = sem_feat_use_masks
        self.sem_feat_use_temporal_window = sem_feat_use_temporal_window
        if self.sem_feat_use_bboxes:
            sem_input_dim += 4
        if self.sem_feat_use_class_logits:
            sem_input_dim += self.num_temp_edge_classes
        if self.sem_feat_use_temporal_window:
            sem_input_dim += self.clip_size

        dim_list = [sem_input_dim] + [sem_feat_hidden_dim] * \
                (semantic_feat_projector_layers - 1) + [semantic_feat_size]
        self.temporal_edge_semantic_feat_projector = build_mlp(dim_list, batch_norm='batch')

        self.num_sim_topk = num_sim_topk
        self.temporal_edge_ranges = temporal_edge_ranges
        self.edge_max_temporal_range = edge_max_temporal_range
        self.use_max_iou_only = use_max_iou_only
        self.use_temporal_edges_only = use_temporal_edges_only
        self.perturb = self.lg_detector.perturb_factor > 0

        # set prediction params
        self.per_video = per_video

        # init ds head
        ds_head.per_video = per_video
        ds_head.num_temp_frames = clip_size
        self.ds_head = MODELS.build(ds_head)

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        losses = {}
        if self.per_video:
            filtered_batch_data_samples = [[b for ind, b in enumerate(bds) \
                    if ind in bds.key_frames_inds] for bds in batch_data_samples]
        else:
            filtered_batch_data_samples = batch_data_samples

        # extract frame-wise graphs by running lg detector on all images and reshape values
        feats, graphs, clip_results, _ = self.extract_feat(batch_inputs, filtered_batch_data_samples, losses)

        # build spatiotemporal graph for each item in batch
        st_graphs = self.build_st_graph(graphs, clip_results)

        # run ds head
        ds_losses = self.ds_head.loss(st_graphs, feats, filtered_batch_data_samples)
        losses.update(ds_losses)

        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        if self.per_video:
            filtered_batch_data_samples = [[b for ind, b in enumerate(bds) \
                    if ind in bds.key_frames_inds] for bds in batch_data_samples]
        else:
            filtered_batch_data_samples = batch_data_samples

        # extract frame-wise graphs by running lg detector on all images and reshape values
        feats, graphs, clip_results, results = self.extract_feat(batch_inputs,
                filtered_batch_data_samples)

        # build spatiotemporal graph for each item in batch
        st_graphs = self.build_st_graph(graphs, clip_results)

        # run ds head
        ds_preds, _ = self.ds_head.predict(st_graphs, feats)

        # update results
        T = len(filtered_batch_data_samples[0])

        # only keep keyframes
        if self.per_video:
            # upsample predictions if needed
            if len(filtered_batch_data_samples[0]) < len(batch_data_samples[0]):
                # compute upsample factor
                upsample_factor = int(np.ceil(len(batch_data_samples[0]) / len(filtered_batch_data_samples[0])))

                # pad predictions
                ds_preds = ds_preds.repeat_interleave(upsample_factor, dim=1)[:,
                        :len(batch_data_samples[0])]

                # pad results and add ds_preds
                padded_results = []
                dummy_instance_data = InstanceData(bboxes=torch.zeros(0, 4),
                        scores=torch.zeros(0), labels=torch.zeros(0)).to(ds_preds.device)

                for ind, (b, ds) in enumerate(zip(batch_data_samples[0], ds_preds.flatten(end_dim=1))):
                    if ind % upsample_factor == 0:
                        r = results[ind // upsample_factor]

                    else:
                        r = DetDataSample(metainfo=b.metainfo,
                                pred_instances=dummy_instance_data)

                    r.pred_ds = ds
                    padded_results.append(r)

            else:
                for r in results:
                    for r, p in zip(results, ds_preds.view(-1, ds_preds.shape[-1])):
                        r.pred_ds = p

        else:
            results = results[T-1::T]
            for r, p in zip(results, ds_preds):
                r.pred_ds = p

        return results

    def reshape_as_clip(self, feats: BaseDataElement, graphs: BaseDataElement, results: SampleList, B: int, T: int) -> Tuple[BaseDataElement]: # reshape quantities in feats by clip
        feats.bb_feats = [x.view(B, T, *x.shape[1:]) for x in feats.bb_feats]
        feats.neck_feats = [x.view(B, T, *x.shape[1:]) for x in feats.neck_feats]
        feats.instance_feats = feats.instance_feats.view(B, T, *feats.instance_feats.shape[1:])
        if 'semantic_feats' in feats:
            feats.semantic_feats = feats.semantic_feats.view(B, T, *feats.semantic_feats.shape[1:])

        # reshape graph nodes by clip
        graphs.nodes.nodes_per_img = Tensor(graphs.nodes.nodes_per_img).split(T)
        graphs.nodes.feats = graphs.nodes.feats.view(B, T, *graphs.nodes.feats.shape[1:])
        if 'semantic_feats' in graphs.nodes:
            graphs.nodes.semantic_feats = graphs.nodes.semantic_feats.view(B, T,
                    *graphs.nodes.semantic_feats.shape[1:])
        graphs.nodes.labels = graphs.nodes.labels.view(B, T, *graphs.nodes.labels.shape[1:])
        graphs.nodes.bboxes = graphs.nodes.bboxes.view(B, T, *graphs.nodes.bboxes.shape[1:])
        graphs.nodes.scores = graphs.nodes.scores.view(B, T, *graphs.nodes.scores.shape[1:])
        if 'masks' in graphs.nodes:
            graphs.nodes.masks = graphs.nodes.masks.view(B, T, *graphs.nodes.masks.shape[1:])

        # set graph edge_flats first dim to be frame_id within clip, not frame_id within batch
        graphs.edges.edge_flats[:, 0] = graphs.edges.edge_flats[:, 0] % T

        # add E_T dummy classes to class logits
        try:
            dummy_logits = torch.ones(graphs.edges.class_logits.shape[0],
                    self.num_temp_edge_classes).to(graphs.edges.class_logits.device) * graphs.edges.class_logits.min()
        except:
            dummy_logits = torch.zeros(graphs.edges.class_logits.shape[0],
                    self.num_temp_edge_classes).to(graphs.edges.class_logits.device)

        graphs.edges.class_logits = torch.cat([graphs.edges.class_logits, dummy_logits], dim=1)

        # reshape graph edges by clip
        graphs.edges.edges_per_img = list(graphs.edges.edges_per_img.split(T))
        graphs.edges.edges_per_clip = [sum(x) for x in graphs.edges.edges_per_img]
        graphs.edges.edge_flats = list(graphs.edges.edge_flats.split(graphs.edges.edges_per_clip))
        graphs.edges.class_logits = list(graphs.edges.class_logits.split(graphs.edges.edges_per_clip))
        graphs.edges.feats = list(graphs.edges.feats.split(graphs.edges.edges_per_clip))
        graphs.edges.viz_feats = list(graphs.edges.viz_feats.split(graphs.edges.edges_per_clip))
        if 'semantic_feats' in graphs.edges:
            graphs.edges.semantic_feats = list(graphs.edges.semantic_feats.split(graphs.edges.edges_per_clip))

        graphs.edges.boxes = list(graphs.edges.boxes.split(graphs.edges.edges_per_clip))
        graphs.edges.boxesA = list(graphs.edges.boxesA.split(graphs.edges.edges_per_clip))
        graphs.edges.boxesB = list(graphs.edges.boxesB.split(graphs.edges.edges_per_clip))
        graphs.edges.labelsA = list(graphs.edges.labelsA.split(graphs.edges.edges_per_clip))
        graphs.edges.labelsB = list(graphs.edges.labelsB.split(graphs.edges.edges_per_clip))

        if 'presence_logits' in graphs.edges.keys():
            del graphs.edges.presence_logits

        # reshape results
        if results is None:
            clip_results = None
        else:
            clip_results = [results[T*i:T*(i+1)] for i in range(B)]

        return feats, graphs, clip_results

    def build_st_graph(self, graphs: BaseDataElement, clip_results: SampleList):
        node_boxes = pad_sequence([pad_sequence([x.pred_instances.bboxes for x in cr]) \
                for cr in clip_results], batch_first=True).transpose(1, 2)
        node_labels = pad_sequence([pad_sequence([x.pred_instances.labels for x in cr]) \
                for cr in clip_results], batch_first=True).transpose(1, 2)

        viz_graph, spat_graph = None, None
        if self.use_viz_graph:
            viz_graph = self._build_visual_edges(graphs)

        if self.use_spat_graph:
            spat_graph = self._build_spatial_edges(node_boxes)

        if viz_graph is not None or spat_graph is not None:
            # add viz and spat edges to st_graph, being mindful of indexing, and extract edge features
            st_graph = self._featurize_st_graph(spat_graph, viz_graph, node_boxes,
                    node_labels, graphs, clip_results[0][0].ori_shape)
        else:
            st_graph = graphs

        return st_graph

    def _featurize_st_graph(self, spat_graph: Tensor, viz_graph: Tensor, node_boxes: Tensor,
            node_labels: Tensor, graphs: BaseDataElement, box_shape: Tensor):
        # extract shape quantities, device
        B, T, N, _ = node_boxes.shape
        M = T * N

        if M == 0: # no fg objects in batch
            # leave graphs as is (node feats contain some dummy features which will be used for classification)
            graphs.edges.edges_per_clip = [sum(x) for x in graphs.edges.edges_per_img]
            return graphs

        device = spat_graph.device if spat_graph is not None else viz_graph.device

        # define graphs to use
        graphs_to_use = []
        if spat_graph is not None:
            graphs_to_use.append(spat_graph)
        if viz_graph is not None:
            graphs_to_use.append(viz_graph)

        # pad spat graph if needed
        if len(graphs_to_use) == 2 and graphs_to_use[0].shape[-1] != graphs_to_use[1].shape[-1]:
            padded_spat_graph = torch.sparse_coo_tensor(graphs_to_use[0].indices(),
                    graphs_to_use[0].values(), size=graphs_to_use[1].shape)
            graphs_to_use[0] = padded_spat_graph

            # pad node_boxes
            pad_dim = (padded_spat_graph.shape[-1] - M) // T
            node_boxes = F.pad(node_boxes, (0, 0, 0, pad_dim))
            node_labels = F.pad(node_labels, (0, 0, 0, pad_dim))

            # recompute dims
            B, T, N, _ = node_boxes.shape
            M = T * N

        # create meshgrid to store node indices corresponding to each edge
        edge_inds = torch.arange(M).to(device)

        # calculate offsets to add to the meshgrid computed indices based on the number of nodes in each graph in each clip
        batch_nodes_per_img = torch.stack(graphs.nodes.nodes_per_img)
        offsets = torch.cumsum(torch.cat([torch.zeros(batch_nodes_per_img.shape[0], 1).to(device),
            batch_nodes_per_img[:, :-1].to(device)], -1), -1).unsqueeze(-1).repeat(1, 1, N)

        # get corrected edge indices
        edge_inds = (edge_inds - N * torch.arange(T).repeat_interleave(N).to(device)).unsqueeze(0) + \
                offsets.view(B, -1)

        # stack graphs to use to get temporal_edge_class
        temporal_edge_class = torch.stack(graphs_to_use, -1)

        # upper triangular mask
        mask = torch.triu(torch.ones(*temporal_edge_class.shape[:-1]), diagonal=1).to_sparse().to(device)

        # apply mask
        temporal_edge_class = apply_sparse_mask(temporal_edge_class, mask)

        # set invalid inds to 0 (based on nodes per img)
        npi = graphs.nodes.nodes_per_img
        npi_mask = torch.stack([pad_sequence(list(torch.ones(T).repeat_interleave(n.int()).split(
            n.int().tolist())) + [torch.zeros(N)], batch_first=True)[:-1].view(M,).to_sparse() for n in npi]).to(device)
        valid_edge_inds = npi_mask.float()

        # mask out invalid_edge_inds on both axes
        temporal_edge_class = temporal_edge_class * valid_edge_inds.to_dense().view(B, -1, 1, 1)
        temporal_edge_class = temporal_edge_class * valid_edge_inds.to_dense().view(B, 1, -1, 1)
        temporal_edge_class = torch.masked._combine_input_and_mask(sum, temporal_edge_class,
                temporal_edge_class)

        # update graphs.edges with temporal edge quantities
        for ind, ec in enumerate(temporal_edge_class):
            # get information from ec
            all_nonzero_vals = ec._values()
            all_nonzero_inds = ec._indices()
            nonzero_uids, nonzero_idx, _ = torch.unique(all_nonzero_inds[:2],
                    dim=-1, sorted=True, return_inverse=True, return_counts=True)

            # UPDATE EDGE FLATS
            extra_edge_flats = edge_inds[ind][nonzero_uids]

            # add img id for temporal edges (set as T, 0 to T-1 being the frame ids)
            extra_edge_flats = torch.cat([torch.ones(1, extra_edge_flats.shape[-1]).to(device) * T,
                extra_edge_flats]).T.long()

            if self.use_temporal_edges_only:
                graphs.edges.edge_flats[ind] = extra_edge_flats
            else:
                graphs.edges.edge_flats[ind] = torch.cat([graphs.edges.edge_flats[ind],
                    extra_edge_flats])

            # UPDATE CLASS LOGITS

            # define extra_edge_class_logits
            extra_edge_class_logits = torch.ones(nonzero_uids.shape[-1], ec.shape[-1]).to(device)

            # loop through edge type and populate class logits
            for i in range(ec.shape[-1]):
                inds_i = (all_nonzero_inds[-1] == i)

                # find target index for each edge, assign value
                extra_edge_class_logits[nonzero_idx[inds_i], i] = all_nonzero_vals[inds_i]

            if self.use_temporal_edges_only:
                extra_edge_class_logits = torch.cat([torch.zeros(extra_edge_class_logits.shape[0],
                    self.num_spatial_edge_classes).to(device), extra_edge_class_logits], 1)
                graphs.edges.class_logits[ind] = extra_edge_class_logits
            else:
                extra_edge_class_logits = torch.cat([torch.zeros(extra_edge_class_logits.shape[0],
                    self.num_spatial_edge_classes).to(device), extra_edge_class_logits], 1)
                graphs.edges.class_logits[ind] = torch.cat([graphs.edges.class_logits[ind],
                        extra_edge_class_logits])

            # update boxes
            extra_boxesA = node_boxes[ind].flatten(end_dim=1)[nonzero_uids[0]]
            extra_boxesB = node_boxes[ind].flatten(end_dim=1)[nonzero_uids[1]]
            extra_edge_boxes = self._box_union(extra_boxesA, extra_boxesB)
            extra_labelsA = node_labels[ind].flatten(end_dim=1)[nonzero_uids[0]]
            extra_labelsB = node_labels[ind].flatten(end_dim=1)[nonzero_uids[1]]
            if self.use_temporal_edges_only:
                graphs.edges.boxes[ind] = extra_edge_boxes
                graphs.edges.boxesA[ind] = extra_boxesA
                graphs.edges.boxesB[ind] = extra_boxesB
                graphs.edges.labelsA[ind] = extra_labelsA
                graphs.edges.labelsB[ind] = extra_labelsB
            else:
                graphs.edges.boxes[ind] = torch.cat([graphs.edges.boxes[ind], extra_edge_boxes])
                graphs.edges.boxesA[ind] = torch.cat([graphs.edges.boxesA[ind], extra_boxesA])
                graphs.edges.boxesB[ind] = torch.cat([graphs.edges.boxesB[ind], extra_boxesB])
                graphs.edges.labelsA[ind] = torch.cat([graphs.edges.labelsA[ind], extra_labelsA])
                graphs.edges.labelsB[ind] = torch.cat([graphs.edges.labelsB[ind], extra_labelsB])

            # update viz feats
            extra_edge_viz_feats = graphs.nodes.feats[ind].view(M, -1)[nonzero_uids.T].mean(1)

            if self.semantic_feat_size > 0:
                # compute map storing the temporal window size for each edge
                edge_to_window_val = torch.abs(torch.arange(T) - torch.arange(T).repeat(T, 1).T).to(nonzero_uids.device)
                nonzero_uids_frame = nonzero_uids // N
                edge_temporal_windows = F.one_hot(edge_to_window_val[nonzero_uids_frame[0],
                        nonzero_uids_frame[1]], num_classes=T)

                # compute sem feats
                extra_edge_sem_feats = self._compute_st_sem_feats(extra_edge_boxes,
                        extra_edge_class_logits[:, -self.num_temp_edge_classes:],
                        edge_temporal_windows, box_shape)

            if self.use_temporal_edges_only:
                graphs.edges.feats[ind] = extra_edge_viz_feats
                if self.semantic_feat_size > 0:
                    graphs.edges.semantic_feats[ind] = extra_edge_sem_feats

            else:
                graphs.edges.feats[ind] = torch.cat([graphs.edges.feats[ind], extra_edge_viz_feats])
                if self.semantic_feat_size > 0:
                    graphs.edges.semantic_feats[ind] = torch.cat([graphs.edges.semantic_feats[ind],
                        extra_edge_sem_feats])

            # update edges per img, temporal edges are grouped into one category
            graphs.edges.edges_per_img[ind] = torch.cat([graphs.edges.edges_per_img[ind],
                Tensor([extra_edge_viz_feats.shape[0]]).to(graphs.edges.edges_per_img[ind])])

        # update edges per clip after adding temporal edges
        graphs.edges.edges_per_clip = [sum(x) for x in graphs.edges.edges_per_img]

        return graphs

    def _compute_st_sem_feats(self, boxes: Tensor, class_logits: Tensor,
            edge_temporal_windows: Tensor, box_shape: Tuple) -> Tensor:

        # run temporal edge semantic feat projector
        sem_feat_input = []
        if self.sem_feat_use_bboxes:
            b_norm = boxes / Tensor(box_shape).flip(0).repeat(2).to(boxes.device) # make 0-1
            sem_feat_input.append(b_norm)
        if self.sem_feat_use_class_logits:
            sem_feat_input.append(class_logits)
        if self.sem_feat_use_temporal_window:
            sem_feat_input.append(edge_temporal_windows)

        sem_feat_input = torch.cat(sem_feat_input, -1)

        if sem_feat_input.shape[0] == 1:
            sem_feats = self.temporal_edge_semantic_feat_projector(sem_feat_input.repeat(2, 1))[0].unsqueeze(0)
        else:
            sem_feats = self.temporal_edge_semantic_feat_projector(sem_feat_input)

        return sem_feats

    def _box_union(self, boxesA, boxesB):
        """
        Compute the union of boxes in two lists.

        Args:
            boxesA (torch.Tensor): Tensor of shape (N, 4) representing the boxes in list A.
            boxesB (torch.Tensor): Tensor of shape (N, 4) representing the boxes in list B.

        Returns:
            torch.Tensor: Tensor of shape (N, 4) representing the union of boxes.
        """

        # Compute the minimum x and y coordinates
        x_min = torch.min(boxesA[:, 0], boxesB[:, 0])
        y_min = torch.min(boxesA[:, 1], boxesB[:, 1])

        # Compute the maximum x and y coordinates
        x_max = torch.max(boxesA[:, 2], boxesB[:, 2])
        y_max = torch.max(boxesA[:, 3], boxesB[:, 3])

        # Create the union boxes tensor
        union_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

        return union_boxes

    def _build_visual_edges(self, graphs: BaseDataElement):
        # store components of shape
        B, T, N, _ = graphs.nodes.feats.size()

        # run kernel fns
        if self.learn_sim_graph:
            sim1 = self.sim_embed1(graphs.nodes.feats).flatten(start_dim=1, end_dim=2)
            sim2 = self.sim_embed2(graphs.nodes.feats).flatten(start_dim=1, end_dim=2).transpose(1, 2)
        else:
            sim1 = graphs.nodes.feats.flatten(start_dim=1, end_dim=2)
            sim2 = graphs.nodes.feats.flatten(start_dim=1, end_dim=2).transpose(1, 2)

        # COSINE SIMILARITY
        # compute pairwise dot products
        sm_graph = torch.bmm(sim1, sim2) # d x d mat.
        sm_graph_norm_factor = torch.bmm(torch.linalg.norm(sim1, dim=-1, keepdim=True),
                torch.linalg.norm(sim2, dim=1, keepdim=True)) + 1e-5
        sm_graph = torch.clamp(sm_graph / sm_graph_norm_factor, 0, 1)
        if sm_graph.shape[-1] == 0:
            return sm_graph.to_sparse()

        # 0 out intra-frame edges
        intra_frame_inds = (torch.stack(torch.meshgrid(torch.arange(N),
            torch.arange(N))).to(graphs.nodes.feats.device).unsqueeze(-1) + \
                    torch.arange(T).to(graphs.nodes.feats.device) * N).flatten(start_dim=1).long()
        sm_graph[torch.arange(sm_graph.shape[0]).view(-1, 1, 1), intra_frame_inds[0], intra_frame_inds[1]] -= 50

        # 0 out padded edges in both directions
        npi_all = graphs.nodes.nodes_per_img
        for ind, npi_clip in enumerate(npi_all):
            for n in npi_clip:
                offset = ind * N
                sm_graph[ind, offset + n.int():offset + N, offset:offset + N] -= 50
                sm_graph[ind, offset:offset + N, offset + n.int():offset + N] -= 50

        # only keep topk most similar edges per node
        topk_sm_graph = torch.zeros_like(sm_graph)
        topk_vals, inds = sm_graph.topk(self.num_sim_topk, dim=-1)

        # now concatenate arange with inds to get proper indices
        topk_vals_norm = topk_vals / (topk_vals.sum(-1).unsqueeze(-1) + 1e-5)
        topk_sm_graph.scatter_(-1, inds, topk_vals_norm)

        return topk_sm_graph.to_sparse()

    def _build_spatial_edges(self, boxes: Tensor):
        B, T, N, _ = boxes.size()
        M = T*N

        front_graph = torch.zeros(B, M, M).to(boxes.device)
        back_graph = torch.zeros(B, M, M).to(boxes.device)

        if M == 0:
            return None

        areas = (boxes[:,:,:,3] - boxes[:,:,:,1] + 1) * \
                (boxes[:,:,:,2] - boxes[:,:,:,0] + 1)

        # set temporal edge ranges
        edge_max_temporal_range = self.edge_max_temporal_range if self.edge_max_temporal_range > 0 else T
        if self.temporal_edge_ranges == 'exp':
            ranges = [2 ** x for x in range(int(np.sqrt(edge_max_temporal_range)) + 1)]
        else:
            ranges = range(1, edge_max_temporal_range + 1)

        for r in ranges:
            for t in range(T-1, 0, -1): # build starting at last (latest) node
                if t - r < 0:
                    continue
                for i in range(N):
                    ious = self._compute_iou(boxes[:,t,i], boxes[:,t-r], areas[:,t,i:i+1],
                            areas[:,t-r]).nan_to_num(0)
                    if self.use_max_iou_only:
                        norm_iou = F.one_hot(ious.argmax(-1), num_classes=N)
                    else:
                        norm_iou = ious / (ious.sum(-1).unsqueeze(-1) + 1e-5)

                    front_graph[:, t*N + i, (t-r)*N:(t-r+1)*N] = norm_iou
                    back_graph[:, (t-r)*N:(t-r+1)*N, t*N + i] = ious

                # normalize back graph by sum of all incoming edges for each pair in temporal range r
                if self.use_max_iou_only:
                    bg_vals = back_graph[:, (t-r)*N:(t-r+1)*N, t*N:(t+1)*N]
                    back_graph[:, (t-r)*N:(t-r+1)*N, t*N:(t+1)*N] = F.one_hot(bg_vals.argmax(-1),
                            num_classes=N)

                else:
                    back_graph[:, (t-r)*N:(t-r+1)*N, t*N:(t+1)*N] /= back_graph[:,
                            (t-r)*N:(t-r+1)*N, t*N:(t+1)*N].sum(-1).unsqueeze(-1)

        # combine forward and backward graph
        fb_graph = torch.maximum(front_graph.transpose(1, 2), back_graph).to_sparse()

        return fb_graph

    def _compute_iou(self, roi, rois, area, areas):
        y_min = torch.max(roi[:,0:1], rois[:,:,0])
        x_min = torch.max(roi[:,1:2], rois[:,:,1])
        y_max = torch.min(roi[:,2:3], rois[:,:,2])
        x_max = torch.min(roi[:,3:4], rois[:,:,3])
        axis0 = x_max - x_min + 1
        axis1 = y_max - y_min + 1
        axis0[axis0 < 0] = 0
        axis1[axis1 < 0] = 0
        intersection = axis0 * axis1
        iou = intersection / (areas + area - intersection)

        return iou

    def extract_feat(self, batch_inputs: Tensor, batch_data_samples: SampleList,
            losses: dict = None) -> Tuple[BaseDataElement]:
        B = len(batch_data_samples)
        T = len(batch_data_samples[0])
        if 'lg' in batch_data_samples[0][0].metainfo:
            lg_list = [x.pop('lg') for b in batch_data_samples for x in b]

            # box perturb
            if self.training and self.perturb:
                perturbed_boxes = self.lg_detector.box_perturbation([l.nodes.bboxes \
                        for l in lg_list], batch_data_samples[0][0].img_shape,
                        clip_size=self.clip_size)
                for l, p in zip(lg_list, perturbed_boxes):
                    l.nodes.bboxes = p

            graphs = BaseDataElement()
            graphs.nodes = BaseDataElement()
            graphs.edges = BaseDataElement()

            if self.reencode_semantics:
                self.compute_lg_semantic_feat(lg_list, graphs)

            # collate node info
            graphs.nodes.viz_feats = pad_sequence([l.nodes.viz_feats \
                    for l in lg_list], batch_first=True)
            N = graphs.nodes.viz_feats.shape[1]

            if self.use_gnn_feats:
                graphs.nodes.gnn_viz_feats = pad_sequence([l.nodes.gnn_viz_feats \
                        for l in lg_list], batch_first=True)
                graphs.nodes.feats = self.node_viz_feat_projector(torch.cat(
                    [graphs.nodes.viz_feats, graphs.nodes.gnn_viz_feats], -1).flatten(end_dim=1)).view(
                            B*T, graphs.nodes.viz_feats.shape[1], self.viz_feat_size)
            else:
                graphs.nodes.feats = self.node_viz_feat_projector(graphs.nodes.viz_feats.flatten(end_dim=1)).view(
                            B*T, graphs.nodes.viz_feats.shape[1], self.viz_feat_size)

            if 'semantic_feats' in lg_list[0].nodes:
                graphs.nodes.semantic_feats = pad_sequence([l.nodes.semantic_feats \
                        for l in lg_list], batch_first=True)

            graphs.nodes.nodes_per_img = [l.nodes.nodes_per_img for l in lg_list]
            graphs.nodes.bboxes = pad_sequence([l.nodes.bboxes for l in lg_list],
                    batch_first=True)
            graphs.nodes.labels = pad_sequence([l.nodes.labels for l in lg_list],
                    batch_first=True)
            graphs.nodes.scores = pad_sequence([l.nodes.scores for l in lg_list],
                    batch_first=True)
            if 'masks' in lg_list[0].nodes.keys():
                graphs.nodes.masks = pad_sequence([l.nodes.masks for l in lg_list],
                        batch_first=True)

            # collate edge info
            graphs.edges.viz_feats = torch.cat([l.edges.viz_feats for l in lg_list])
            graphs.edges.gnn_viz_feats = torch.cat([l.edges.gnn_viz_feats for l in lg_list])
            if self.use_gnn_feats:
                graphs.edges.feats = self.edge_viz_feat_projector(torch.cat(
                    [graphs.edges.viz_feats, graphs.edges.gnn_viz_feats], -1))
            else:
                graphs.edges.feats = self.edge_viz_feat_projector(graphs.edges.viz_feats)
            if 'semantic_feats' in lg_list[0].edges:
                graphs.edges.semantic_feats = torch.cat([l.edges.semantic_feats for l in lg_list])

            graphs.edges.boxes = torch.cat([l.edges.boxes for l in lg_list])
            graphs.edges.boxesA = torch.cat([l.edges.boxesA for l in lg_list])
            graphs.edges.boxesB = torch.cat([l.edges.boxesB for l in lg_list])
            graphs.edges.labelsA = torch.cat([l.edges.labelsA for l in lg_list])
            graphs.edges.labelsB = torch.cat([l.edges.labelsB for l in lg_list])
            graphs.edges.class_logits = torch.cat([l.edges.class_logits for l in lg_list])
            graphs.edges.edge_flats = torch.cat([torch.cat([torch.ones(
                l.edges.edge_flats.shape[0], 1).to(l.edges.edge_flats) * ind,
                l.edges.edge_flats], dim=1) for ind, l in enumerate(lg_list)])
            graphs.edges.edges_per_img = Tensor([l.edges.boxes.shape[0] for l in lg_list]).int()

            # set feats
            feats = BaseDataElement()
            feats.bb_feats = (torch.stack([l.img_feats.view(-1, 1, 1) for l in lg_list]),)
            feats.neck_feats = (torch.stack([l.img_feats.view(-1, 1, 1) for l in lg_list]),)
            feats.instance_feats = graphs.nodes.viz_feats
            if 'semantic_feats' in graphs.nodes:
                feats.semantic_feats = graphs.nodes.semantic_feats

            # add node info to results
            metainfo = [x.metainfo for b in batch_data_samples for x in b]
            pred_instances = [InstanceData(bboxes=l.nodes.bboxes, scores=l.nodes.scores,
                labels=l.nodes.labels) for l in lg_list]
            results = [DetDataSample(pred_instances=p, metainfo=m) for p, m in zip(pred_instances, metainfo)]

        else:
            feats, graphs, results, _, _ = self.lg_detector.extract_lg(
                    batch_inputs.flatten(end_dim=1),
                    [x for y in batch_data_samples for x in y],
                    losses=losses)
            N = graphs.nodes.viz_feats.shape[1]
            graphs.nodes.labels = pad_sequence([r.pred_instances.labels for r in results],
                    batch_first=True)
            graphs.nodes.scores = pad_sequence([r.pred_instances.scores for r in results],
                    batch_first=True)
            graphs.nodes.bboxes = pad_sequence([r.pred_instances.bboxes for r in results],
                    batch_first=True)
            if 'instance_feats' in feats:
                graphs.nodes.instance_feats = feats.instance_feats
            if 'semantic_feats' in feats:
                graphs.nodes.semantic_feats = feats.semantic_feats

            # concatenate boxes
            graphs.edges.boxes = torch.cat(graphs.edges.boxes)
            graphs.edges.boxesA = torch.cat(graphs.edges.boxesA)
            graphs.edges.boxesB = torch.cat(graphs.edges.boxesB)
            graphs.edges.labelsA = torch.cat(graphs.edges.labelsA)
            graphs.edges.labelsB = torch.cat(graphs.edges.labelsB)

            # fuse viz feats, gnn viz feats
            if self.use_gnn_feats:
                input_node_viz_feat = torch.cat([graphs.nodes.viz_feats,
                    graphs.nodes.gnn_viz_feats], -1)
                graphs.nodes.feats = self.node_viz_feat_projector(
                        input_node_viz_feat.flatten(end_dim=1)).view(B*T, N, self.viz_feat_size)
                input_edge_viz_feat = torch.cat([graphs.edges.viz_feats,
                    graphs.edges.gnn_viz_feats], -1)
                graphs.edges.feats = self.edge_viz_feat_projector(input_edge_viz_feat)
            else:
                graphs.nodes.feats = self.node_viz_feat_projector(
                        graphs.nodes.viz_feats.flatten(end_dim=1)).view(B*T, N, self.viz_feat_size)
                graphs.edges.feats = self.edge_viz_feat_projector(graphs.edges.viz_feats)

        # reorganize feats and graphs by clip
        feats, graphs, clip_results = self.reshape_as_clip(feats, graphs, results, B, T)

        return feats, graphs, clip_results, results

    def compute_lg_semantic_feat(self, lg_list: List, graphs: BaseDataElement) -> Tensor:
        device = lg_list[0].nodes.viz_feats.device
        boxes = [l.nodes.bboxes for l in lg_list]
        classes = [l.nodes.labels for l in lg_list]
        scores = [l.nodes.scores for l in lg_list]
        masks = None
        if 'masks' in lg_list[0].nodes:
            masks = [l.nodes.masks for l in lg_list]

        # compute semantic feat
        c = pad_sequence(classes, batch_first=True)
        b = pad_sequence(boxes, batch_first=True)
        s = pad_sequence(scores, batch_first=True)
        b_norm = b / Tensor(lg_list[0].ori_shape).flip(0).repeat(2).to(device)
        c_one_hot = F.one_hot(c, num_classes=self.lg_detector.num_classes)

        sem_feat_input = []
        if self.sem_feat_use_bboxes:
            sem_feat_input.append(b_norm)

        if self.sem_feat_use_class_logits:
            sem_feat_input.append(c_one_hot)

        # process masks
        if self.sem_feat_use_masks:
            # iterate through masks and convert to polygon mask
            polygon_masks = self.lg_detector.masks_to_polygons(masks)

            # process masks
            polygon_masks = pad_sequence(polygon_masks, batch_first=True) # B x N x P x 2
            polygon_masks_norm = polygon_masks / Tensor(lg_list[0].ori_shape).flip(0).to(device)

            sem_feat_input.append(polygon_masks_norm.flatten(start_dim=-2))

        sem_feat_input.append(s.unsqueeze(-1))

        sem_feat_input = torch.cat(sem_feat_input, -1).flatten(end_dim=1)
        if sem_feat_input.shape[0] == 1:
            s = self.lg_detector.semantic_feat_projector(torch.cat([sem_feat_input, sem_feat_input]))[0]
        else:
            s = self.lg_detector.semantic_feat_projector(sem_feat_input)

        # add node sem feats to graph
        graphs.nodes.semantic_feats = s.view(b_norm.shape[0], b_norm.shape[1], s.shape[-1])

        # compute edge semantic feats
        eb_norm = torch.cat([l.edges.boxes for l in lg_list]) / Tensor(lg_list[0].batch_input_shape).flip(0).repeat(2).to(device) # make 0-1
        ec = torch.cat([l.edges.class_logits for l in lg_list])
        edge_sem_input = torch.cat([eb_norm, ec], -1) # detach class logits to prevent backprop
        if edge_sem_input.shape[1] == 1:
            edge_sem_feats = self.lg_detector.edge_semantic_feat_projector(edge_sem_input.repeat(2, 1))[0].unsqueeze(0)
        else:
            edge_sem_feats = self.lg_detector.edge_semantic_feat_projector(edge_sem_input)

        graphs.edges.semantic_feats = edge_sem_feats

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError
