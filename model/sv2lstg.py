import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Tuple, Union
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import bbox2roi, roi2bbox, scale_boxes
from mmdet.models.detectors.base import BaseDetector
from mmengine.structures import BaseDataElement
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
from .lg_cvs import LGDetector
from .predictor_heads.modules.layers import build_mlp

@MODELS.register_module()
class SV2LSTG(BaseDetector):
    def __init__(self, lg_detector: BaseDetector, ds_head: ConfigType,
            viz_feat_size=256, sem_feat_size=256, sim_embedder_feat_size=256,
            num_frame_edge_classes=4, use_spat_graph: bool = False,
            use_viz_graph: bool = False, learn_sim_graph: bool = True,
            semantic_feat_projector_layers: int = 3, pred_per_frame: bool = False,
            use_positional_embedding: bool = True, num_sim_topk: int = 2,
            temporal_edge_ranges: str = 'exp', edge_max_temporal_range: int = -1,
            use_max_iou_only: bool = True, use_temporal_edges_only: bool = False,
            per_video: bool = False, **kwargs):
        super().__init__(**kwargs)

        # init lg detector
        self.lg_detector = MODELS.build(lg_detector)

        # init ds head
        ds_head.pred_per_frame = pred_per_frame
        ds_head.per_video = per_video
        self.ds_head = MODELS.build(ds_head)

        # visual graph feature projectors
        self.sim_embed1 = torch.nn.Linear(viz_feat_size + sem_feat_size,
                sim_embedder_feat_size, bias=False)
        self.sim_embed2 = torch.nn.Linear(viz_feat_size + sem_feat_size,
                sim_embedder_feat_size, bias=False)

        # set extra params
        self.num_frame_edge_classes = num_frame_edge_classes
        self.use_spat_graph = use_spat_graph
        self.use_viz_graph = use_viz_graph
        self.learn_sim_graph = learn_sim_graph
        self.viz_feat_size = viz_feat_size
        self.sem_feat_size = sem_feat_size

        self.num_temp_edge_classes = 0
        if self.use_viz_graph:
            self.num_temp_edge_classes += 1
        if self.use_spat_graph:
            self.num_temp_edge_classes += 1

        # edge semantic feat projector
        self.num_edge_classes = self.num_frame_edge_classes + self.num_temp_edge_classes
        dim_list = [self.num_edge_classes + 4] + [512] * \
                (semantic_feat_projector_layers - 1) + [sem_feat_size]
        self.edge_semantic_feat_projector = build_mlp(dim_list, batch_norm='none')

        self.num_sim_topk = num_sim_topk
        self.temporal_edge_ranges = temporal_edge_ranges
        self.edge_max_temporal_range = edge_max_temporal_range
        self.use_max_iou_only = use_max_iou_only
        self.use_temporal_edges_only = use_temporal_edges_only
        self.use_positional_embedding = use_positional_embedding

        # set prediction params
        self.per_video = per_video
        self.pred_per_frame = pred_per_frame or per_video

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        losses = {}

        # extract frame-wise graphs by running lg detector on all images and reshape values
        feats, graphs, clip_results, _ = self.extract_feat(batch_inputs, batch_data_samples)

        # build spatiotemporal graph for each item in batch
        st_graphs = self.build_st_graph(graphs, clip_results)

        # run ds head
        ds_losses = self.ds_head.loss(st_graphs, feats, batch_data_samples)
        losses.update(ds_losses)

        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        # extract frame-wise graphs by running lg detector on all images and reshape values
        feats, graphs, clip_results, results = self.extract_feat(batch_inputs,
                batch_data_samples)

        # build spatiotemporal graph for each item in batch
        st_graphs = self.build_st_graph(graphs, clip_results)

        # run ds head
        ds_preds = self.ds_head.predict(st_graphs, feats, results)

        # update results
        B, T, _, _, _ = batch_inputs.shape

        # only keep keyframes
        if self.per_video:
            for r in results:
                for r, p in zip(results, ds_preds.repeat_interleave(T, dim=0)):
                    r.pred_ds = p

        else:
            results = results[T-1::T]
            for r, p in zip(results, ds_preds):
                r.pred_ds = p

        return results

    def reshape_as_clip(self, feats: BaseDataElement, graphs: BaseDataElement, results: SampleList, B: int, N: int) -> Tuple[BaseDataElement]:
        # reshape quantities in feats by clip
        feats.bb_feats = [x.view(B, N, *x.shape[1:]) for x in feats.bb_feats]
        feats.neck_feats = [x.view(B, N, *x.shape[1:]) for x in feats.neck_feats]
        feats.instance_feats = feats.instance_feats.view(B, N, *feats.instance_feats.shape[1:])
        feats.semantic_feats = feats.semantic_feats.view(B, N, *feats.semantic_feats.shape[1:])

        # reshape graph nodes by clip
        graphs.nodes.nodes_per_img = Tensor(graphs.nodes.nodes_per_img).split(N)
        graphs.nodes.feats = graphs.nodes.feats.view(B, N, *graphs.nodes.feats.shape[1:])

        # set graph edge_flats first dim to be frame_id within clip, not frame_id within batch
        graphs.edges.edge_flats[:, 0] = graphs.edges.edge_flats[:, 0] % N

        # add E_T dummy classes to class logits
        try:
            dummy_logits = torch.ones(graphs.edges.class_logits.shape[0],
                    self.num_temp_edge_classes).to(graphs.edges.class_logits.device) * graphs.edges.class_logits.min()
        except:
            dummy_logits = torch.zeros(graphs.edges.class_logits.shape[0],
                    self.num_temp_edge_classes).to(graphs.edges.class_logits.device)

        graphs.edges.class_logits = torch.cat([graphs.edges.class_logits, dummy_logits], dim=1)

        # reshape graph edges by clip
        graphs.edges.edges_per_img = list(graphs.edges.edges_per_img.split(N))
        graphs.edges.edges_per_clip = [sum(x) for x in graphs.edges.edges_per_img]
        graphs.edges.edge_flats = list(graphs.edges.edge_flats.split(graphs.edges.edges_per_clip))
        graphs.edges.class_logits = list(graphs.edges.class_logits.split(graphs.edges.edges_per_clip))
        graphs.edges.feats = list(graphs.edges.feats.split(graphs.edges.edges_per_clip))
        graphs.edges.presence_logits = graphs.edges.presence_logits.view(B, N,
                *graphs.edges.presence_logits.shape[1:])
        graphs.edges.boxes = list(torch.cat(graphs.edges.boxes).split(graphs.edges.edges_per_clip))
        graphs.edges.boxesA = list(torch.cat(graphs.edges.boxesA).split(graphs.edges.edges_per_clip))
        graphs.edges.boxesB = list(torch.cat(graphs.edges.boxesB).split(graphs.edges.edges_per_clip))

        # reshape results
        clip_results = [results[N*i:N*(i+1)] for i in range(B)]

        return feats, graphs, clip_results

    def build_st_graph(self, graphs: BaseDataElement, clip_results: SampleList):
        viz_graph, spat_graph = None, None
        if self.use_viz_graph:
            viz_graph = self._build_visual_edges(graphs)

        if self.use_spat_graph:
            node_boxes = pad_sequence([pad_sequence([x.pred_instances.bboxes for x in cr]) \
                    for cr in clip_results], batch_first=True).transpose(1, 2)
            spat_graph = self._build_spatial_edges(node_boxes)

        if viz_graph is not None or spat_graph is not None:
            # add viz and spat edges to st_graph, being mindful of indexing, and extract edge features
            st_graph = self._featurize_st_graph(spat_graph, viz_graph, node_boxes,
                    graphs, clip_results[0][0].img_shape)
        else:
            st_graph = graphs

        return st_graph

    def _featurize_st_graph(self, spat_graph: Tensor, viz_graph: Tensor, node_boxes: Tensor,
            graphs: BaseDataElement, batch_input_shape: Tensor):
        # extract shape quantities, device
        B, T, N, _ = graphs.nodes.feats.shape
        M = T * N
        device = spat_graph.device if spat_graph is not None else viz_graph.device

        # define graphs to use
        graphs_to_use = []
        if spat_graph is not None:
            graphs_to_use.append(spat_graph)
        if viz_graph is not None:
            graphs_to_use.append(viz_graph)

        # create meshgrid to store node indices corresponding to each edge
        edge_x = torch.meshgrid(torch.arange(M), torch.arange(M))[0].to(device)

        # calculate offsets to add to the meshgrid computed indices based on the number of nodes in each graph in each clip
        batch_nodes_per_img = torch.stack(graphs.nodes.nodes_per_img).to(device)
        offsets = torch.cumsum(torch.cat([torch.zeros(batch_nodes_per_img.shape[0], 1).to(device),
            batch_nodes_per_img[:, :-1]], -1), -1).unsqueeze(-1).repeat(1, 1, N)

        # get corrected edge indices
        edge_x = (edge_x - N * torch.arange(T).unsqueeze(-1).repeat(1, N).view(-1, 1).to(device)) + \
                offsets.view(offsets.shape[0], -1, 1)
        edge_y = edge_x.transpose(1, 2)

        # compute edge presence
        temporal_edge_adj_mat = torch.triu(sum(graphs_to_use), diagonal=1) # 0 if no edge, > 0 if there is an edge

        # TODO(adit98) use nodes per img for this
        # set invalid inds to 0 (find this out by checking for boxes that are all 0 bc of the pad operation)
        invalid_edge_inds = (node_boxes == torch.zeros(4).to(device)).all(-1).flatten(start_dim=1)
        temporal_edge_adj_mat[invalid_edge_inds] = 0
        temporal_edge_adj_mat[invalid_edge_inds.unsqueeze(1).repeat(1, M, 1)] = 0

        # compute presence, box, class of each edge
        temporal_edge_class = torch.stack(graphs_to_use, -1)

        # update graphs.edges with temporal edge quantities
        for ind, (eam, ec, edge_x_i, edge_y_i) in enumerate(zip(temporal_edge_adj_mat,
                temporal_edge_class, edge_x, edge_y)):

            # update edge flats
            extra_edge_flats = torch.stack([edge_x_i[eam > 0], edge_y_i[eam > 0]], -1).long()

            # add img id for temporal edges (set as T, 0 to T-1 being the frame ids)
            extra_edge_flats = torch.cat([torch.ones_like(extra_edge_flats)[:, 0:1] * T,
                extra_edge_flats], dim=1)

            if self.use_temporal_edges_only:
                graphs.edges.edge_flats[ind] = extra_edge_flats
            else:
                graphs.edges.edge_flats[ind] = torch.cat([graphs.edges.edge_flats[ind],
                    extra_edge_flats])

            # update class logits
            extra_edge_class_logits = ec[eam > 0]
            if self.use_temporal_edges_only:
                graphs.edges.class_logits[ind] = extra_edge_classes
            else:
                extra_edge_class_logits = torch.cat([torch.zeros(extra_edge_class_logits.shape[0],
                    self.num_frame_edge_classes).to(device), extra_edge_class_logits], 1)
                graphs.edges.class_logits[ind] = torch.cat([graphs.edges.class_logits[ind],
                        extra_edge_class_logits])

            # update boxes
            eb = self._box_union(node_boxes.flatten(start_dim=1, end_dim=2),
                    node_boxes.flatten(start_dim=1, end_dim=2))[ind]
            pad_size = eam.shape[0] - eb.shape[0]
            eb = F.pad(eb.permute(2, 0, 1), (0, pad_size, 0, pad_size)).permute(1, 2, 0)
            extra_edge_boxes = eb[eam > 0]
            extra_boxesA = node_boxes[ind].flatten(end_dim=1)[torch.nonzero(eam > 0)[:, 0]]
            extra_boxesB = node_boxes[ind].flatten(end_dim=1)[torch.nonzero(eam > 0)[:, 1]]
            if self.use_temporal_edges_only:
                graphs.edges.boxes[ind] = extra_edge_boxes
                graphs.edges.boxesA[ind] = extra_boxesA
                graphs.edges.boxesB[ind] = extra_boxesB
            else:
                graphs.edges.boxes[ind] = torch.cat([graphs.edges.boxes[ind], extra_edge_boxes])
                graphs.edges.boxesA[ind] = torch.cat([graphs.edges.boxesA[ind], extra_boxesA])
                graphs.edges.boxesB[ind] = torch.cat([graphs.edges.boxesB[ind], extra_boxesB])

            try:
                # update viz feats
                extra_edge_viz_feats = graphs.nodes.feats[ind].view(M, -1)[extra_edge_flats].mean(1)[:, :self.viz_feat_size]
                # compute sem feats
                extra_edge_sem_feats = self._compute_sem_feats(extra_edge_boxes,
                        extra_edge_class_logits, batch_input_shape)
                extra_edge_feats = torch.cat([extra_edge_viz_feats, extra_edge_sem_feats], dim=-1)

            except:
                extra_edge_feats = torch.zeros(0, graphs.edges.feats[ind].shape[-1]).to(graphs.edges.feats[ind])

            if self.use_temporal_edges_only:
                graphs.edges.feats[ind] = extra_edge_feats
            else:
                graphs.edges.feats[ind] = torch.cat([graphs.edges.feats[ind], extra_edge_feats])

            # update edges per img, temporal edges are grouped into one category
            graphs.edges.edges_per_img[ind] = torch.cat([graphs.edges.edges_per_img[ind],
                Tensor([extra_edge_feats.shape[0]]).to(graphs.edges.edges_per_img[ind])])

        # update edges per clip after adding temporal edges
        graphs.edges.edges_per_clip = [sum(x) for x in graphs.edges.edges_per_img]

        # return updated graphs
        return graphs

    def _compute_sem_feats(self, boxes: Tensor, class_logits: Tensor, batch_input_shape: Tuple) -> Tensor:
        # run edge semantic feat projector, concat with viz feats, add to edges
        b_norm = boxes / Tensor(batch_input_shape).flip(0).repeat(2).to(boxes.device) # make 0-1
        sem_feats = self.edge_semantic_feat_projector(torch.cat([b_norm, class_logits], -1))

        # TODO(adit98) add embedding representing temporal edge window

        return sem_feats

    def _box_union(self, boxes1, boxes2):
        # boxes1, boxes2: Tensors of shape (B, N1, 4) and (B, N2, 4) representing bounding boxes in (x1, y1, x2, y2) format
        B, N1, _ = boxes1.shape
        B, N2, _ = boxes2.shape

        # Expand dimensions to perform broadcasting
        boxes1 = boxes1.unsqueeze(2)  # (B, N1, 1, 4)
        boxes2 = boxes2.unsqueeze(1)  # (B, 1, N2, 4)

        # Compute the coordinates of the intersection bounding boxes
        union_x1 = torch.min(boxes1[:, :, :, 0], boxes2[:, :, :, 0])  # (B, N1, N2)
        union_y1 = torch.min(boxes1[:, :, :, 1], boxes2[:, :, :, 1])  # (B, N1, N2)
        union_x2 = torch.max(boxes1[:, :, :, 2], boxes2[:, :, :, 2])  # (B, N1, N2)
        union_y2 = torch.max(boxes1[:, :, :, 3], boxes2[:, :, :, 3])  # (B, N1, N2)

        return torch.stack([union_x1, union_y1, union_x2, union_y2], -1)

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
            return sm_graph

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

        return topk_sm_graph

    def _build_spatial_edges(self, boxes: Tensor):
        B, T, N, _ = boxes.size()
        M = T*N

        front_graph = torch.zeros((B, M, M)).to(boxes.device)
        back_graph = torch.zeros((B, M, M)).to(boxes.device)

        if M == 0:
            return front_graph

        areas = (boxes[:,:,:,3] - boxes[:,:,:,1] + 1) * \
                (boxes[:,:,:,2] - boxes[:,:,:,0] + 1)

        # TODO(adit98) vectorize this
        edge_max_temporal_range = self.edge_max_temporal_range if self.edge_max_temporal_range > 0 \
                else T
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
                            areas[:,t-r])
                    if self.use_max_iou_only:
                        norm_iou = F.one_hot(ious.argmax(-1), num_classes=N)
                    else:
                        norm_iou = ious / ious.sum(-1).unsqueeze(-1)

                    # update front graph with normalized ious
                    front_graph[:, t*N+i, (t-r)*N:(t-r+1)*N] = norm_iou

                    # update back graph with raw ious
                    back_graph[:, (t-r)*N:(t-r+1)*N, t*N+i] = ious

                # normalize back graph by sum of all incoming edges for each pair in temporal range r
                if self.use_max_iou_only:
                    back_graph[:, (t-r)*N:(t-r+1)*N, t*N:(t+1)*N] = F.one_hot(back_graph[:,
                        (t-r)*N:(t-r+1)*N, t*N:(t+1)*N].argmax(-1), num_classes=N)
                else:
                    back_graph[:, (t-r)*N:(t-r+1)*N, t*N:(t+1)*N] /= back_graph[:,
                            (t-r)*N:(t-r+1)*N, t*N:(t+1)*N].sum(-1).unsqueeze(-1)

        # NaN to zero
        front_graph[front_graph != front_graph] = 0
        back_graph[back_graph != back_graph] = 0

        fb_graph = torch.maximum(front_graph.transpose(1, 2), back_graph)

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

    def extract_feat(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Tuple[BaseDataElement]:
        # run lg detector on all images
        feats, graphs, results = self.lg_detector.extract_lg(batch_inputs.flatten(end_dim=1),
                [x for y in batch_data_samples for x in y])

        # reorganize feats and graphs by clip
        B, N = batch_inputs.shape[:2]
        feats, graphs, clip_results = self.reshape_as_clip(feats, graphs, results, B, N)

        return feats, graphs, clip_results, results

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError
