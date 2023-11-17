from mmdet.registry import MODELS
from abc import ABCMeta
from mmdet.utils import ConfigType, OptConfigType, InstanceList, OptMultiConfig
from mmengine.model import BaseModule
from mmengine.structures import BaseDataElement
from mmdet.models.roi_heads.roi_extractors import BaseRoIExtractor
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_overlaps
from mmdet.structures.bbox.transforms import bbox2roi, scale_boxes
from typing import List, Tuple, Union
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torchvision.transforms import functional as TF, InterpolationMode
import math
import dgl
from .modules.layers import build_mlp
from .modules.gnn import GNNHead

@MODELS.register_module()
class GraphHead(BaseModule, metaclass=ABCMeta):
    """Graph Head to construct graph from detections

    Args:
        edges_per_node (int)
        viz_feat_size (int)
        roi_extractor
        gnn_cfg (ConfigType): gnn cfg
    """
    def __init__(self, edges_per_node: int, viz_feat_size: int, semantic_feat_size: int,
            roi_extractor: BaseRoIExtractor, num_edge_classes: int,
            presence_loss_cfg: ConfigType, presence_loss_weight: float,
            classifier_loss_cfg: ConfigType, classifier_loss_weight: float,
            gt_use_pred_detections: bool = False, sem_feat_hidden_dim: int = 2048,
            semantic_feat_projector_layers: int = 3, num_roi_feat_maps: int = 4,
            gnn_cfg: ConfigType = None, compute_gt_eval: bool = False,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        # attributes for building graph from detections
        self.edges_per_node = edges_per_node
        self.viz_feat_size = viz_feat_size
        self.semantic_feat_size = semantic_feat_size
        self.roi_extractor = roi_extractor
        self.num_roi_feat_maps = num_roi_feat_maps
        dim_list = [viz_feat_size, 64, 64]
        self.edge_mlp_sbj = build_mlp(dim_list, batch_norm='batch',
                final_nonlinearity=False)
        self.edge_mlp_obj = build_mlp(dim_list, batch_norm='batch',
                final_nonlinearity=False)
        self.gt_use_pred_detections = gt_use_pred_detections
        self.compute_gt_eval = compute_gt_eval

        # presence loss
        self.presence_loss = MODELS.build(presence_loss_cfg)
        self.presence_loss_weight = presence_loss_weight

        # edge classifier loss
        self.classifier_loss = MODELS.build(classifier_loss_cfg)
        self.classifier_loss_weight = classifier_loss_weight

        # gnn attributes
        if gnn_cfg is not None:
            gnn_cfg.input_dim_node = viz_feat_size
            gnn_cfg.input_dim_edge = viz_feat_size
            self.gnn = MODELS.build(gnn_cfg)
        else:
            self.gnn = None

        # attributes for predicting relation class
        self.num_edge_classes = num_edge_classes
        dim_list = [viz_feat_size, viz_feat_size, self.num_edge_classes + 1] # predict no edge or which class
        self.edge_predictor = build_mlp(dim_list, batch_norm='batch', final_nonlinearity=False)

        # make query projector if roi_extractor is None
        if self.roi_extractor is None:
            dim_list = [viz_feat_size] * 3
            self.edge_query_projector = build_mlp(dim_list, batch_norm='batch')

    def _predict_edge_presence(self, node_features, nodes_per_img):
        # EDGE PREDICTION
        mlp_input = node_features.flatten(end_dim=1)
        if mlp_input.shape[0] == 1:
            sbj_feats = self.edge_mlp_sbj(torch.cat([mlp_input, mlp_input]))[0].unsqueeze(0)
        else:
            sbj_feats = self.edge_mlp_sbj(mlp_input)
        if mlp_input.shape[0] == 1:
            obj_feats = self.edge_mlp_obj(torch.cat([mlp_input, mlp_input]))[0].unsqueeze(0)
        else:
            obj_feats = self.edge_mlp_obj(mlp_input)

        sbj_feats = sbj_feats.view(len(node_features), -1,
                sbj_feats.size(-1)) # B x N x F, where F is feature dimension
        obj_feats = obj_feats.view(len(node_features), -1,
                obj_feats.size(-1)) # B x N x F, where F is feature dimension

        # get likelihood of edge between each pair of proposals using kernel
        edge_presence_logits = torch.bmm(sbj_feats, obj_feats.transpose(1, 2)) # B x N x N

        # mask using nodes_per_img
        mask = torch.zeros_like(edge_presence_logits)
        for i, num_nodes in enumerate(nodes_per_img):
            mask[i, num_nodes:, :] = float('-inf')  # Set mask for subject nodes beyond the number of nodes in the image to -inf
            mask[i, :, num_nodes:] = float('-inf')  # Set mask for object nodes beyond the number of nodes in the image to -inf

        # also mask diagonal
        mask = mask + (torch.eye(mask.shape[1]).unsqueeze(0).to(mask.device) * float('-inf')).nan_to_num(0)

        edge_presence_masked = edge_presence_logits + mask

        return edge_presence_logits, edge_presence_masked

    def _build_edges(self, results: SampleList, nodes_per_img: List, feats: BaseDataElement = None) -> SampleList:
        # get boxes, rescale
        scale_factor = results[0].scale_factor
        boxes = pad_sequence([r.pred_instances.bboxes for r in results], batch_first=True)
        boxes_per_img = [len(r.pred_instances.bboxes) for r in results]
        rescaled_boxes = scale_boxes(boxes.float(), scale_factor)

        # compute all box_unions
        edge_boxes = self.box_union(rescaled_boxes, rescaled_boxes)

        # select valid edge boxes
        valid_edge_boxes = [e[:b, :b].flatten(end_dim=1) for e, b in zip(edge_boxes, boxes_per_img)]
        edge_rois = bbox2roi(valid_edge_boxes)

        # compute edge feats
        if self.roi_extractor is not None:
            roi_input_feats = feats.neck_feats[:self.num_roi_feat_maps] \
                    if feats.neck_feats is not None else feats.bb_feats[:self.num_roi_feat_maps]
            edge_viz_feats = self.roi_extractor(roi_input_feats, edge_rois).squeeze(-1).squeeze(-1)
        else:
            # offset is just cum sum of edges_per_img
            edges_per_img = torch.tensor([b*b for b in boxes_per_img])
            edge_offsets = torch.cat([torch.zeros(1), torch.cumsum(edges_per_img, 0)[:-1]]).repeat_interleave(edges_per_img)

            # define edges to keep
            edges_to_keep = (torch.cat([torch.arange(e) for e in edges_per_img]) + edge_offsets).to(boxes.device)

            # densely add object queries to get edge feats, select edges with edges_to_keep
            edge_viz_feats = self.edge_query_projector((feats.instance_feats.unsqueeze(1) + \
                    feats.instance_feats.unsqueeze(2)).flatten(end_dim=-2))[edges_to_keep.long()]

        # predict edge presence
        edge_presence_logits, edge_presence_masked = self._predict_edge_presence(feats.instance_feats, nodes_per_img)

        # collate all edge information
        edges = BaseDataElement()
        edges.boxes = torch.cat(valid_edge_boxes)
        edges.boxesA = torch.cat([b[:num_b].repeat_interleave(num_b, dim=0) for num_b, b in zip(
                boxes_per_img, rescaled_boxes)])
        edges.boxesB = torch.cat([b[:num_b].repeat(num_b, 1) for num_b, b in zip(
                boxes_per_img, rescaled_boxes)])
        edges.edges_per_img = [num_b * num_b for num_b in boxes_per_img]
        edges.viz_feats = edge_viz_feats

        # store presence logits
        edges.presence_logits = edge_presence_masked

        return edges, edge_presence_logits

    def _predict_edge_classes(self, graph: BaseDataElement, batch_input_shape: tuple) -> BaseDataElement:
        # predict edge class
        edge_predictor_input = graph.edges.viz_feats + graph.edges.gnn_viz_feats
        if edge_predictor_input.shape[0] == 1:
            graph.edges.class_logits = self.edge_predictor(edge_predictor_input.repeat(2, 1))[0].unsqueeze(0)
        else:
            graph.edges.class_logits = self.edge_predictor(edge_predictor_input)

        return graph

    def _select_edges(self, edges: BaseDataElement, nodes_per_img: List) -> BaseDataElement:
        # SELECT TOP E EDGES PER NODE USING PRESENCE LOGITS, COMPUTE EDGE FLATS
        presence_logits = edges.presence_logits # B x N x N

        # pick top E edges per node, or N - 1 if num nodes is too small
        edge_flats, edge_indices = self._edge_flats_from_adj_mat(presence_logits, nodes_per_img)
        edges_per_img = Tensor([len(ef) for ef in edge_flats]).to(presence_logits.device).int()

        edges.edges_per_img = edges_per_img
        edges.batch_index = torch.arange(len(edges_per_img)).to(edges_per_img.device).repeat_interleave(
                edges_per_img).view(-1, 1) # stores the batch_id of each edge
        edges.edge_flats = torch.cat([edges.batch_index, torch.cat(edge_flats)], dim=1)
        edges.boxes = edges.boxes[edge_indices]
        edges.boxesA = edges.boxesA[edge_indices]
        edges.boxesB = edges.boxesB[edge_indices]
        edges.viz_feats = edges.viz_feats[edge_indices]

        return edges

    def _edge_flats_from_adj_mat(self, presence_logits, nodes_per_img):
        edge_flats = []
        edge_indices = []
        edge_index_offset = 0
        num_edges = torch.minimum(torch.ones(len(presence_logits)) * self.edges_per_node,
                Tensor(nodes_per_img) - 1).int()

        for pl, ne, nn in zip(presence_logits, num_edges, nodes_per_img):
            if nn == 0:
                edge_flats.append(torch.zeros(0, 2).to(pl.device).int())
                edge_indices.append(torch.zeros(0).to(pl.device))
                continue

            row_indices = torch.arange(nn).to(pl.device).view(-1, 1).repeat(1, ne.item())
            topk_indices = torch.topk(pl, k=ne.item(), dim=1).indices
            edge_flat = torch.stack([row_indices, topk_indices[:nn]], dim=-1).long().view(-1, 2)
            edge_flat = edge_flat[self.drop_duplicates(edge_flat.sort(dim=1).values).long()]
            edge_indices.append(torch.arange(nn * nn).to(pl.device).view(nn, nn)[edge_flat[:, 0], edge_flat[:, 1]] + \
                    edge_index_offset)
            edge_index_offset = edge_index_offset + nn * nn
            edge_flats.append(edge_flat)

        return edge_flats, torch.cat(edge_indices).long()

    def predict(self, results: SampleList, feats: BaseDataElement) -> Tuple[BaseDataElement]:
        nodes_per_img = [len(r.pred_instances.bboxes) for r in results]

        if self.compute_gt_eval:
            # build edges for GT
            gt_edges = self._build_gt_edges(results)
        else:
            gt_edges = None

        # build edges
        edges, _ = self._build_edges(results, nodes_per_img, feats)

        # select edges
        edges = self._select_edges(edges, nodes_per_img)

        # construct graph out of edges and result
        graph = self._construct_graph(feats, edges, nodes_per_img)

        # apply gnn
        if self.gnn is not None:
            dgl_g = self.gnn(graph)

            # update graph
            graph = self._update_graph(graph, dgl_g)

        # predict edge classes
        graph = self._predict_edge_classes(graph, results[0].batch_input_shape)

        return graph, gt_edges

    def _construct_graph(self, feats: BaseDataElement, edges: BaseDataElement,
                nodes_per_img: List) -> BaseDataElement:
        graph = BaseDataElement()
        graph.edges = edges

        # move result data into nodes
        nodes = BaseDataElement()
        nodes.viz_feats = feats.instance_feats
        nodes.nodes_per_img = nodes_per_img
        graph.nodes = nodes

        return graph

    def _update_graph(self, graph: BaseDataElement, dgl_g: dgl.DGLGraph) -> BaseDataElement:
        # update node viz feats (leave semantic feats the same, add to original feats)
        updated_node_feats = pad_sequence(dgl_g.ndata['viz_feats'].split(graph.nodes.nodes_per_img),
                batch_first=True)
        graph.nodes.gnn_viz_feats = updated_node_feats

        # update graph structure
        graph.edges.edges_per_img = dgl_g.batch_num_edges()
        graph.edges.batch_index = torch.arange(len(graph.edges.edges_per_img)).to(
                graph.edges.edges_per_img.device).repeat_interleave(graph.edges.edges_per_img).view(-1, 1)

        batch_edge_offset = torch.cat([torch.zeros(1),
                dgl_g.batch_num_nodes()[:-1]], 0).cumsum(0).to(graph.edges.batch_index.device)
        edge_flats = torch.stack(dgl_g.edges(), 1) - \
                batch_edge_offset[graph.edges.batch_index].view(-1, 1)
        graph.edges.edge_flats = torch.cat([graph.edges.batch_index, edge_flats], 1).long()

        # update edge data (skip connection to orig edge feats)
        graph.edges.boxes = dgl_g.edata['boxes'].split(graph.edges.edges_per_img.tolist())
        graph.edges.boxesA = dgl_g.edata['boxesA'].split(graph.edges.edges_per_img.tolist())
        graph.edges.boxesB = dgl_g.edata['boxesB'].split(graph.edges.edges_per_img.tolist())
        graph.edges.gnn_viz_feats = dgl_g.edata['gnn_feats']

        return graph

    def _build_gt_edges(self, results: SampleList) -> BaseDataElement:
        boxes_per_img = []
        bounding_boxes = []
        for r in results:
            if r.is_det_keyframe and not self.gt_use_pred_detections:
                boxes = r.gt_instances.bboxes
                bounding_boxes.append(boxes)
                boxes_per_img.append(len(boxes))

            else:
                # get boxes, rescale
                boxes = r.pred_instances.bboxes
                scores = r.pred_instances.scores

                # use score thresh 0.3 to filter boxes
                boxes = boxes[scores > 0.3]

                # convert to tensor and scale boxes
                bounding_boxes.append(scale_boxes(boxes.float(), r.scale_factor))
                boxes_per_img.append(len(bounding_boxes))

        # convert to tensor
        bounding_boxes = pad_sequence(bounding_boxes, batch_first=True)

        # compute centroids and distances for general use
        centroids = (bounding_boxes[:, :, :2] + bounding_boxes[:, :, 2:]) / 2
        distance_x = centroids[:, :, 0].unsqueeze(-1) - centroids[:, :, 0].unsqueeze(-2)
        distance_y = centroids[:, :, 1].unsqueeze(-1) - centroids[:, :, 1].unsqueeze(-2)

        relationships = []

        # FIRST COMPUTE INSIDE-OUTSIDE MASK

        # compute areas of all boxes and create meshgrid
        B, N, _ = bounding_boxes.shape
        areas = self.box_area(bounding_boxes) # B x N x 1
        areas_x = areas.unsqueeze(-1).expand(B, N, N)
        areas_y = areas.unsqueeze(-2).expand(B, N, N)

        # compute intersection
        intersection = self.box_intersection(bounding_boxes, bounding_boxes) # B x N x N

        # inside-outside is when intersection is close to the area of the smaller box
        inside_outside_matrix = intersection / torch.minimum(areas_x, areas_y)
        inside_outside_mask = (inside_outside_matrix >= 0.8)

        # COMPUTE LEFT-RIGHT, ABOVE-BELOW, INSIDE-OUTSIDE MASKS

        # compute angle matrix using distance x and distance y
        angle_matrix = torch.atan2(distance_y, distance_x)
        left_right_mask = ((angle_matrix > (-math.pi / 4)) & (angle_matrix <= (math.pi / 4))) | \
                ((angle_matrix > (3 * math.pi / 4)) | (angle_matrix <= (-3 * math.pi / 4)))
        above_below_mask = ((angle_matrix > (math.pi / 4)) & (angle_matrix <= (3 * math.pi / 4))) | \
                ((angle_matrix > (-3 * math.pi / 4)) & (angle_matrix <= (-math.pi / 4)))

        # left right and above below are only when inside outside is False
        left_right_mask = left_right_mask.int() * (~inside_outside_mask).int() # 1 for left-right
        above_below_mask = above_below_mask.int() * (~inside_outside_mask).int() * 2 # 2 for above-below
        inside_outside_mask = inside_outside_mask.int() * 3 # 3 for inside-outside

        relationships = (left_right_mask + above_below_mask + inside_outside_mask).long()

        # SELECT E EDGES PER NODE BASED ON gIoU
        iou_matrix = bbox_overlaps(bounding_boxes, bounding_boxes)
        diag_mask = torch.eye(iou_matrix.shape[-1]).repeat(iou_matrix.shape[0], 1, 1).bool()
        iou_matrix[diag_mask] = float('-inf') # set diagonal to -inf
        iou_matrix[torch.minimum(areas_x, areas_y) == 0] = float('-inf') # set all entries where bbox area is 0 to -inf
        valid_nodes_per_img = [(a > 0).sum().item() for a in areas]
        num_edges_per_img = [min(self.edges_per_node, max(0, v-1)) for v in valid_nodes_per_img] # limit edges based on number of valid boxes per img
        selected_edges = [torch.topk(iou_mat[:v, :v], e, dim=1).indices for iou_mat, v, e in zip(iou_matrix, valid_nodes_per_img, num_edges_per_img)]

        # COMPUTE EDGE FLATS (PAIRS OF OBJECT IDS CORRESPONDING TO EACH EDGE)
        edge_flats = []
        for s in selected_edges:
            # edge flats is just arange, each column of selected edges
            ef = torch.stack([torch.arange(s.shape[0]).view(-1, 1).repeat(1, s.shape[1]).to(s.device),
                s], -1).view(-1, 2)

            # DROP DUPLICATES
            ef = ef[self.drop_duplicates(ef.sort(dim=1).values).long()]

            edge_flats.append(ef)

        # COMPUTE EDGE BOXES AND SELECT USING EDGE FLATS
        edge_boxes = self.box_union(bounding_boxes, bounding_boxes)
        selected_edge_boxes = [eb[ef[:, 0], ef[:, 1]] for eb, ef in zip(edge_boxes, edge_flats)]
        selected_boxesA = [b[ef[:, 0]] for b, ef in zip(bounding_boxes, edge_flats)]
        selected_boxesB = [b[ef[:, 1]] for b, ef in zip(bounding_boxes, edge_flats)]

        # SELECT RELATIONSHIPS USING EDGE FLATS
        selected_relations = [er[ef[:, 0], ef[:, 1]] for er, ef in zip(relationships, edge_flats)]

        # add edge flats, boxes, and relationships to gt_graph structure
        gt_edges = BaseDataElement()
        gt_edges.edge_flats = edge_flats
        gt_edges.edge_boxes = selected_edge_boxes
        gt_edges.boxesA = selected_boxesA
        gt_edges.boxesB = selected_boxesB
        gt_edges.edge_relations = selected_relations

        return gt_edges

    def drop_duplicates(self, A):
        if A.shape[0] == 0:
            return torch.zeros(0).to(A.device)

        unique, idx, counts = torch.unique(A, dim=0, sorted=True, return_inverse=True,
                return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(A.device), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]

        return first_indices

    def box_area(self, boxes):
        # boxes: Tensor of shape (batch_size, num_boxes, 4) representing bounding boxes in (x1, y1, x2, y2) format
        width = boxes[..., 2] - boxes[..., 0]  # Compute width
        height = boxes[..., 3] - boxes[..., 1]  # Compute height
        area = width * height  # Compute area

        return area

    def box_union(self, boxes1, boxes2):
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

    def box_intersection(self, boxes1, boxes2):
        # boxes1, boxes2: Tensors of shape (B, N1, 4) and (B, N2, 4) representing bounding boxes in (x1, y1, x2, y2) format
        B, N1, _ = boxes1.shape
        B, N2, _ = boxes2.shape

        # Expand dimensions to perform broadcasting
        boxes1 = boxes1.unsqueeze(2)  # (B, N1, 1, 4)
        boxes2 = boxes2.unsqueeze(1)  # (B, 1, N2, 4)

        # Compute the coordinates of the intersection bounding boxes
        intersection_x1 = torch.max(boxes1[:, :, :, 0], boxes2[:, :, :, 0])  # (B, N1, N2)
        intersection_y1 = torch.max(boxes1[:, :, :, 1], boxes2[:, :, :, 1])  # (B, N1, N2)
        intersection_x2 = torch.min(boxes1[:, :, :, 2], boxes2[:, :, :, 2])  # (B, N1, N2)
        intersection_y2 = torch.min(boxes1[:, :, :, 3], boxes2[:, :, :, 3])  # (B, N1, N2)

        # Compute the areas of the intersection bounding boxes
        intersection_width = torch.clamp(intersection_x2 - intersection_x1, min=0)  # (B, N1, N2)
        intersection_height = torch.clamp(intersection_y2 - intersection_y1, min=0)  # (B, N1, N2)
        intersection_area = intersection_width * intersection_height  # (B, N1, N2)

        return intersection_area

    def loss_and_predict(self, results: SampleList, feats: BaseDataElement) -> Tuple[SampleList, dict]:
        # init loss dict
        losses = {}

        # build edges for GT
        gt_edges = self._build_gt_edges(results)

        # build edges and compute presence probabilities
        nodes_per_img = [len(r.pred_instances.bboxes) for r in results]
        edges, presence_logits = self._build_edges(results, nodes_per_img, feats)

        # compute edge presence loss
        edge_presence_loss = self.edge_presence_loss(presence_logits, edges, gt_edges)

        # select edges, construct graph, apply gnn, and predict edge classes
        edges = self._select_edges(edges, nodes_per_img)

        # construct graph out of edges and result
        graph = self._construct_graph(feats, edges, nodes_per_img)

        # apply gnn
        if self.gnn is not None:
            dgl_g = self.gnn(graph)

            # update graph
            graph = self._update_graph(graph, dgl_g)

        # predict edge classes
        graph = self._predict_edge_classes(graph, results[0].batch_input_shape)

        # compute edge classifier loss
        edge_classifier_loss = self.edge_classifier_loss(graph.edges, gt_edges)

        # update losses
        losses.update(edge_presence_loss)
        losses.update(edge_classifier_loss)

        return losses, graph

    def edge_presence_loss(self, presence_logits, edges, gt_edges):
        # first match edge boxes to gt edge boxes
        bA = edges.boxesA.split(edges.edges_per_img)
        bB = edges.boxesB.split(edges.edges_per_img)
        pred_matched_inds, pred_unmatched_inds, _ = self.match_boxes(bA, bB,
                gt_edges.boxesA, gt_edges.boxesB, num=32, iou_threshold=0.5, iou_lower_bound=0.2)

        # assign labels (1 if matched, 0 if unmatched)
        training_inds = [torch.cat([m.view(-1), u.view(-1)]) for m, u in zip(pred_matched_inds, pred_unmatched_inds)]
        flat_edge_relations = []
        for pl, t in zip(presence_logits.flatten(start_dim=1), training_inds):
            flat_edge_relations.append(pl[t])

        flat_edge_relations = torch.cat(flat_edge_relations)
        edge_presence_gt = torch.cat([torch.cat([torch.ones_like(m).view(-1), torch.zeros_like(u).view(-1)]) \
                for m, u in zip(pred_matched_inds, pred_unmatched_inds)])

        presence_loss = self.presence_loss(flat_edge_relations, edge_presence_gt).nan_to_num(0) * self.presence_loss_weight

        return {'loss_edge_presence': presence_loss}

    def edge_classifier_loss(self, edges, gt_edges):
        # first match edge boxes to gt edge boxes
        pred_matched_inds, _, gt_matched_inds = self.match_boxes(edges.boxesA,
                edges.boxesB, gt_edges.boxesA, gt_edges.boxesB, num=16,
                pos_fraction=0.875, iou_lower_bound=0.2)

        # assign labels (1 if matched, 0 if unmatched)
        flat_edge_classes = torch.cat([cl[t.view(-1)] for cl, t in zip(edges.class_logits.split(
            edges.edges_per_img.tolist()), pred_matched_inds)])
        edge_classifier_gt = torch.cat([r[t.view(-1)] for r, t in zip(gt_edges.edge_relations, gt_matched_inds)])

        classifier_loss = self.classifier_loss(flat_edge_classes, edge_classifier_gt).nan_to_num(0) * self.classifier_loss_weight

        return {'loss_edge_classifier': classifier_loss}

    def match_boxes(self, pred_boxes_A, pred_boxes_B, gt_boxes_A, gt_boxes_B, iou_threshold=0.5, iou_lower_bound=0.5, num=50, pos_fraction=0.5):
        # pred_boxes_A: List of tensors of length B, where each tensor has shape (N, 4) representing predicted bounding boxes A in (x1, y1, x2, y2) format
        # pred_boxes_B: List of tensors of length B, where each tensor has shape (N, 4) representing predicted bounding boxes B in (x1, y1, x2, y2) format
        # gt_boxes_A: List of tensors of length B, where each tensor has shape (M, 4) representing ground truth bounding boxes A in (x1, y1, x2, y2) format
        # gt_boxes_B: List of tensors of length B, where each tensor has shape (M, 4) representing ground truth bounding boxes B in (x1, y1, x2, y2) format
        # iou_threshold: IoU threshold for matching
        # iou_lower_bound: Lower bound on IoU for returning unmatched boxes
        B = len(pred_boxes_A)

        pred_matched_indices = []
        pred_unmatched_indices = []
        gt_matched_indices = []

        for b in range(B):
            p_A = pred_boxes_A[b]
            p_B = pred_boxes_B[b]
            g_A = gt_boxes_A[b]
            g_B = gt_boxes_B[b]

            N, _ = p_A.shape
            M, _ = g_A.shape

            # compute overlaps, handle no GT boxes
            if M == 0:
                overlaps = torch.zeros(N, 1).to(p_A.device)
            elif N == 0:
                pred_matched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                pred_unmatched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                gt_matched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                continue
            else:
                overlaps_AA = bbox_overlaps(p_A, g_A)
                overlaps_BB = bbox_overlaps(p_B, g_B)
                overlaps_AB = bbox_overlaps(p_A, g_B)
                overlaps_BA = bbox_overlaps(p_B, g_A)
                overlaps = torch.max(torch.min(overlaps_AA, overlaps_BB), torch.min(overlaps_AB, overlaps_BA))

            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            matched_indices = torch.nonzero(max_overlaps >= iou_threshold, as_tuple=False).squeeze()
            unmatched_indices = torch.nonzero(max_overlaps < iou_lower_bound, as_tuple=False).squeeze()

            # sample
            sampled_matched_inds, sampled_unmatched_inds = self.sample_indices(
                    matched_indices, unmatched_indices, num, pos_fraction)

            pred_matched_indices.append(sampled_matched_inds)
            pred_unmatched_indices.append(sampled_unmatched_inds)
            gt_matched_indices.append(argmax_overlaps[sampled_matched_inds])

        return pred_matched_indices, pred_unmatched_indices, gt_matched_indices

    def sample_indices(self, matched_indices, unmatched_indices, N, R):
        # Get the device of the input tensors
        device = matched_indices.device

        # reshape
        matched_indices = matched_indices.view(-1)
        unmatched_indices = unmatched_indices.view(-1)

        # Get the number of positive and negative matches
        num_positive_matches = matched_indices.shape[0]
        num_negative_matches = unmatched_indices.shape[0]

        # Calculate the number of matched indices based on the desired ratio
        num_matched_indices = int(N * R)
        num_matched_indices = min(num_matched_indices, num_positive_matches)

        if num_matched_indices > 0:
            # Sample the required number of matched indices
            sampled_matched_indices = matched_indices[random.sample(range(num_positive_matches), num_matched_indices)]
        else:
            sampled_matched_indices = torch.tensor([], dtype=torch.int64, device=device)

        # Calculate the remaining number of indices to sample
        remaining_indices = N - num_matched_indices

        # Adjust the remaining number of indices if there aren't enough unmatched indices
        remaining_indices = min(remaining_indices, num_negative_matches)

        # Sample the remaining number of unmatched indices
        if remaining_indices > 0:
            sampled_unmatched_indices = unmatched_indices[random.sample(range(num_negative_matches), remaining_indices)]
        else:
            sampled_unmatched_indices = torch.tensor([], dtype=torch.int64, device=device)

        # Return the sampled matched and unmatched indices separately
        return sampled_matched_indices, sampled_unmatched_indices
