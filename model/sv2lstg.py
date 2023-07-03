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

@MODELS.register_module()
class SV2LSTG(BaseDetector):
    def __init__(self, lg_detector: BaseDetector, ds_head: ConfigType,
            viz_feat_size=256, sem_feat_size=256, sim_embedder_feat_size=256,
            num_temp_edge_classes=2, learn_sim_graph=False, **kwargs):
        super().__init__(**kwargs)

        # init lg detector
        self.lg_detector = MODELS.build(lg_detector)

        # init ds head
        self.ds_head = MODELS.build(ds_head)

        # visual graph feature projectors
        self.sim_embed1 = torch.nn.Linear(viz_feat_size + sem_feat_size,
                sim_embedder_feat_size, bias=False)
        self.sim_embed2 = torch.nn.Linear(viz_feat_size + sem_feat_size,
                sim_embedder_feat_size, bias=False)

        # set extra params
        self.num_temp_edge_classes = num_temp_edge_classes
        self.learn_sim_graph = learn_sim_graph

        # TODO(adit98) load these from cfg
        self.num_sim_topk = 2
        self.temporal_edge_ranges = 'exp'
        self.edge_max_temporal_range = 15
        self.use_max_iou_only = True

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        # extract frame-wise graphs by running lg detector on all images and reshape values
        feats, graphs, clip_results = self.extract_feat(batch_inputs, batch_data_samples)

        # build spatiotemporal graph for each item in batch
        st_graphs = self.build_st_graph(feats, graphs, clip_results)
        breakpoint()

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        # extract frame-wise graphs by running lg detector on all images and reshape values
        feats, graphs, clip_results = self.extract_feat(batch_inputs, batch_data_samples)
        breakpoint()

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
        dummy_logits = torch.ones(graphs.edges.class_logits.shape[0],
                self.num_temp_edge_classes).to(graphs.edges.class_logits.device) * graphs.edges.class_logits.min()
        graphs.edges.class_logits = torch.cat([graphs.edges.class_logits, dummy_logits], dim=1)

        # reshape graph edges by clip
        graphs.edges.edges_per_img = graphs.edges.edges_per_img.split(N)
        graphs.edges.edges_per_clip = [sum(x) for x in graphs.edges.edges_per_img]
        graphs.edges.edge_flats = graphs.edges.edge_flats.split(graphs.edges.edges_per_clip)
        graphs.edges.class_logits = graphs.edges.class_logits.split(graphs.edges.edges_per_clip)
        graphs.edges.feats = graphs.edges.feats.split(graphs.edges.edges_per_clip)
        graphs.edges.presence_logits = graphs.edges.presence_logits.view(B, N,
                *graphs.edges.presence_logits.shape[1:])
        graphs.edges.boxes = torch.cat(graphs.edges.boxes).split(graphs.edges.edges_per_clip)
        graphs.edges.boxesA = torch.cat(graphs.edges.boxesA).split(graphs.edges.edges_per_clip)
        graphs.edges.boxesB = torch.cat(graphs.edges.boxesB).split(graphs.edges.edges_per_clip)

        # reshape results
        clip_results = [results[N*i:N*(i+1)] for i in range(B)]

        return feats, graphs, clip_results

    def build_st_graph(self, feats: BaseDataElement, graphs: BaseDataElement, clip_results: SampleList):
        viz_graph = self._build_visual_edges(feats)
        node_boxes = pad_sequence([pad_sequence([x.pred_instances.bboxes for x in cr]) for cr in clip_results],
                batch_first=True).transpose(1, 2)
        spat_graph = self._build_spatial_edges(node_boxes)
        breakpoint()

        raise NotImplementedError

    def _postprocess_st_graph(self):
        # create meshgrid to store node indices corresponding to each edge
        edge_x = torch.meshgrid(torch.arange(M), torch.arange(M))[0].to(node_boxes.device)

        # calculate offsets to add to the meshgrid computed indices based on the number of nodes in each graph in each clip
        batch_nodes_per_img = torch.stack([torch.tensor(npi) for g, npi in zip(graphs,
            nodes_per_img)]).to(node_boxes.device)
        offsets = torch.cumsum(torch.cat([torch.zeros(batch_nodes_per_img.shape[0], 1).to(
            batch_nodes_per_img.device), batch_nodes_per_img[:, :-1]], -1), -1).unsqueeze(
                    -1).repeat(1, 1, N)

        # get corrected edge indices
        edge_x = (edge_x - N * torch.arange(T).unsqueeze(-1).repeat(1, N).view(-1, 1).to(edge_x.device)) + \
                offsets.view(offsets.shape[0], -1, 1)
        edge_y = edge_x.transpose(1, 2)

        # compute edge presence
        temporal_edge_adj_mat = torch.triu(sum(graphs_to_use), diagonal=1) # 0 if no edge, > 0 if there is an edge

        # set invalid inds to 0
        invalid_edge_inds = (node_boxes == torch.zeros(_).to(node_boxes.device)).all(-1).flatten(start_dim=1)
        temporal_edge_adj_mat[invalid_edge_inds] = 0
        temporal_edge_adj_mat[invalid_edge_inds.unsqueeze(1).repeat(1, M, 1)] = 0

        # compute presence, box, class of each edge
        temporal_edge_class = torch.stack(graphs_to_use, -1)

        # iterate through each graph, add temporal edge indices and features
        for g, eam, ec, edge_x_i, edge_y_i in zip(graphs, temporal_edge_adj_mat, temporal_edge_class, edge_x, edge_y):
            # remove existing processed semantic feats
            if 'node_semantic_feats' in g:
                del g['node_semantic_feats']

            if 'edge_semantic_feats' in g:
                del g['edge_semantic_feats']

            # add edge flats
            extra_edge_flats = torch.stack([edge_x_i[eam > 0], edge_y_i[eam > 0]], -1).long()
            if self.use_temporal_edges_only:
                g['edge_flats'] = extra_edge_flats
            else:
                g['edge_flats'] = torch.cat([g['edge_flats'], extra_edge_flats])

            # get classes for new edges, one hot encode, prepend 0s for spatial edge class probs
            extra_edge_classes = ec[eam > 0]

            if self.use_temporal_edges_only:
                g['edge_semantics'] = extra_edge_classes
            else:
                extra_edge_classes = torch.cat([torch.zeros(extra_edge_classes.shape[0],
                    self.num_edge_classes).to(ec.device), extra_edge_classes], -1)
                # edit existing edges to append 0s for temporal edge class probs, and encoded extra edge classes
                g['edge_semantics'] = torch.cat([torch.cat([g['edge_semantics'].softmax(-1),
                    torch.zeros_like(g['edge_semantics'])[:, :self.num_temp_edge_classes]],
                        -1), extra_edge_classes])

            # get viz feats
            # TODO(adit98) add embedding representing temporal edge window
            extra_edge_viz_feats = g['node_feats'][extra_edge_flats].mean(1)

            if self.use_seg_grounding:
                if self.use_temporal_edges_only:
                    raise NotImplementedError

                # Seg Grounding: expand each node feat to mask size, concatenate all 4 components and
                # pass through projector

                # get mask pairs and smooth
                extra_edge_masks = g['masks'][extra_edge_flats]
                extra_edge_masks = F.adaptive_avg_pool2d(self.edge_gaussian(
                    extra_edge_masks), output_size=14)

                # binarize and multiply with class
                extra_edge_classes = g['labels'][extra_edge_flats][:, 1:].argmax(-1) + 1
                extra_edge_masks = (extra_edge_masks.sigmoid() >= 0.5).int() * extra_edge_classes.unsqueeze(-1).unsqueeze(-1)

                # get feats of incident nodes and pad to mask size

                # padding params
                _, _, h, _ = extra_edge_masks.shape
                offset = 1 - (h % 2) # offset is 1 if h is even, since we need uneven padding

                # reshape and pad viz feats
                pad_dims = (h//2 - 1, h//2  - 1 + offset, h//2 - 1, h//2 - 1 + offset)
                extra_edge_viz_feats = F.pad(extra_edge_viz_feats.unsqueeze(-1).unsqueeze(-1),
                        pad_dims, mode='replicate')

                # pass edge viz feats, masks through refiner to get seg grounded edge viz feats
                extra_edge_viz_feats = self.temporal_edge_feat_refiner(
                        torch.cat([extra_edge_masks, extra_edge_viz_feats], dim=1)).squeeze(-1).squeeze(-1)

            if self.use_temporal_edges_only:
                g['edge_feats'] = extra_edge_viz_feats
            else:
                g['edge_feats'] = torch.cat([g['edge_feats'], extra_edge_viz_feats])

            # generate new edge_boxes
            eb = box_union(g['boxes'], g['boxes'])
            pad_size =  eam.shape[0] - eb.shape[0]
            eb = F.pad(eb.permute(2, 0, 1), (0, pad_size, 0, pad_size)).permute(1, 2, 0)
            extra_edge_boxes = eb[eam > 0]
            if self.use_temporal_edges_only:
                g['edge_boxes'] = extra_edge_boxes
            else:
                g['edge_boxes'] = torch.cat([g['edge_boxes'], extra_edge_boxes])

            # generate new edge_masks
            if 'edge_masks' in g:
                extra_edge_masks = g['masks'][extra_edge_flats]
                if self.use_temporal_edges_only:
                    g['edge_masks'] = extra_edge_masks
                else:
                    g['edge_masks'] = torch.cat([g['edge_masks'], extra_edge_masks])

        # return updated graphs
        return graphs

    def _build_visual_edges(self, feats: BaseDataElement):
        # store components of shape
        B, T, N, _ = feats.instance_feats.size()

        # run kernel fns
        if self.learn_sim_graph:
            sim1 = self.sim_embed1(feats.instance_feats).flatten(start_dim=1, end_dim=2)
            sim2 = self.sim_embed2(feats.instance_feats).flatten(start_dim=1, end_dim=2).transpose(1, 2)
        else:
            sim1 = feats.instance_feats.flatten(start_dim=1, end_dim=2)
            sim2 = feats.instance_feats.flatten(start_dim=1, end_dim=2).transpose(1, 2)

        sm_graph = torch.bmm(sim1, sim2) # d x d mat.
        #sm_graph_norm_factor = torch.bmm(torch.norm(sim1, dim=-1).unsqueeze(-1),
        #        torch.norm(sim1, dim=-1).unsqueeze(1)) + 1e-5
        #sm_graph = sm_graph / sm_graph_norm_factor
        if sm_graph.shape[-1] == 0:
            return sm_graph

        # 0 out intra-frame edges
        intra_frame_inds = (torch.stack(torch.meshgrid(torch.arange(N),
            torch.arange(N))).to(feats.instance_feats.device).unsqueeze(-1) + \
                    torch.arange(T).to(feats.instance_feats.device) * N).flatten(start_dim=1).long()
        sm_graph[torch.arange(sm_graph.shape[0]).view(-1, 1, 1), intra_frame_inds[0], intra_frame_inds[1]] -= 50

        # 0 out padded edges
        sm_graph[sm_graph == 0] -= 50

        # only keep topk most similar edges per node
        topk_sm_graph = torch.zeros_like(sm_graph)
        topk_vals, inds = sm_graph.topk(self.num_sim_topk, dim=-1)
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
        if self.temporal_edge_ranges == 'exp':
            ranges = [2 ** x for x in range(int(np.sqrt(self.edge_max_temporal_range)) + 1)]
        else:
            ranges = range(1, self.edge_max_temporal_range + 1)

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

        return feats, graphs, clip_results

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError
