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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torchvision.transforms import functional as TF, InterpolationMode
from torchvision.ops.boxes import box_area
import dgl
from .modules.utils import box_union
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
            roi_extractor: BaseRoIExtractor, num_edge_classes: int, presence_loss_cfg: ConfigType,
            num_roi_feat_maps: int = 4, gnn_cfg: ConfigType = None,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        # attributes for building graph from detections
        self.edges_per_node = edges_per_node
        self.viz_feat_size = viz_feat_size
        self.semantic_feat_size = semantic_feat_size
        self.roi_extractor = roi_extractor
        self.num_roi_feat_maps = num_roi_feat_maps
        dim_list = [viz_feat_size + semantic_feat_size, 64, 64]
        self.edge_mlp_sbj = build_mlp(dim_list, batch_norm='batch',
                final_nonlinearity=False)
        self.edge_mlp_obj = build_mlp(dim_list, batch_norm='batch',
                final_nonlinearity=False)

        # gnn attributes
        if gnn_cfg is not None:
            gnn_cfg.input_dim_node = viz_feat_size + semantic_feat_size
            gnn_cfg.input_dim_edge = viz_feat_size
            self.gnn = MODELS.build(gnn_cfg)
        else:
            self.gnn = None

        # attributes for predicting relation class
        self.num_edge_classes = num_edge_classes
        dim_list = [viz_feat_size, viz_feat_size, self.num_edge_classes]
        self.edge_predictor = build_mlp(dim_list)
        self.edge_semantic_feat_projector = torch.nn.Linear(num_edge_classes + 4,
                semantic_feat_size)

        # make query projector if roi_extractor is None
        if self.roi_extractor is None:
            dim_list = [256, 256, 256] # HACK put query size in cfg
            self.edge_query_projector = build_mlp(dim_list, batch_norm='batch')

    def _predict_edge_presence(self, node_features, nodes_per_img):
        # EDGE PREDICTION
        sbj_feats = self.edge_mlp_sbj(node_features.flatten(end_dim=1))
        sbj_feats = sbj_feats.view(len(node_features), -1,
                sbj_feats.size(-1)) # B x N x F, where F is feature dimension
        obj_feats = self.edge_mlp_obj(node_features.flatten(end_dim=1))
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
        mask += (torch.eye(mask.shape[1]).unsqueeze(0).to(mask.device) * float('-inf')).nan_to_num(0)

        edge_presence_logits += mask

        return edge_presence_logits

    def _build_edges(self, results: SampleList, nodes_per_img: List, feats: BaseDataElement = None) -> SampleList:
        # get boxes, rescale
        scale_factor = results[0].scale_factor
        boxes = pad_sequence([r.pred_instances.bboxes for r in results], batch_first=True)
        rescaled_boxes = scale_boxes(boxes, scale_factor).view(-1, 4)

        # compute all box_unions
        edge_boxes = box_union(rescaled_boxes, rescaled_boxes)
        batch_ids = torch.arange(boxes.shape[0]).repeat_interleave(boxes.shape[1])
        batch_mg = torch.meshgrid(batch_ids, batch_ids)
        edges_to_keep = batch_mg[0] == batch_mg[1]
        edge_boxes = edge_boxes[torch.where(edges_to_keep)]
        edge_box_batch_id = batch_mg[0][torch.where(edges_to_keep)].to(edge_boxes.device)
        edge_rois = torch.cat([edge_box_batch_id.view(-1, 1), edge_boxes], 1)

        # compute edge feats
        if self.roi_extractor is not None:
            roi_input_feats = feats.neck_feats[:self.num_roi_feat_maps] \
                    if feats.neck_feats is not None else feats.bb_feats[:self.num_roi_feat_maps]
            edge_viz_feats = self.roi_extractor(roi_input_feats, edge_rois).squeeze(-1).squeeze(-1)
        else:
            edge_viz_feats = self.edge_query_projector((feats.instance_feats.view(1, -1, 256) + \
                    feats.instance_feats.view(-1, 1, 256))[torch.where(edges_to_keep)])

        # predict edge presence
        node_feats = torch.cat([feats.instance_feats, feats.semantic_feats], dim=-1)
        edge_presence_logits = self._predict_edge_presence(node_feats, nodes_per_img)

        # collate all edge information
        edges = BaseDataElement()
        edges.boxes = edge_boxes.view(-1, edge_boxes.shape[-1])
        edges.presence_logits = edge_presence_logits
        edges.feats = edge_viz_feats

        return edges

    def _predict_edge_classes(self, graph: BaseDataElement, batch_input_shape: tuple) -> BaseDataElement:
        # predict edge class
        graph.edges.class_logits = self.edge_predictor(graph.edges.feats)

        # run edge semantic feat projector, concat with viz feats, add to edges
        eb_norm = graph.edges.boxes / Tensor(batch_input_shape).flip(0).repeat(2).to(graph.edges.boxes.device) # make 0-1
        edge_sem_feats = self.edge_semantic_feat_projector(torch.cat([eb_norm, graph.edges.class_logits], -1))
        graph.edges.feats = torch.cat([graph.edges.feats, edge_sem_feats], -1)

        return graph

    def _select_edges(self, edges: BaseDataElement, nodes_per_img: List) -> BaseDataElement:
        # SELECT TOP E EDGES PER NODE USING PRESENCE LOGITS, COMPUTE EDGE FLATS
        presence_logits = edges.presence_logits # B x N x N

        # pick top E edges per node, or N - 1 if num nodes is too small
        edge_flats = self._edge_flats_from_adj_mat(presence_logits, nodes_per_img)
        edges_per_img = Tensor([len(ef) for ef in edge_flats]).to(presence_logits.device).int()

        # get indices of selected edges and index edge boxes, feats, and class logits
        index_mat = torch.arange(presence_logits.flatten().shape[0]).view_as(presence_logits)
        batch_index = torch.arange(len(edges_per_img)).to(edges_per_img.device).repeat_interleave(edges_per_img).view(-1, 1)
        batch_edge_flats = torch.cat([batch_index, torch.cat(edge_flats)], dim=1)
        edge_indices = index_mat[batch_edge_flats.unbind(1)]

        edges.edges_per_img = edges_per_img
        edges.batch_index = batch_index # stores the batch_id of each edge
        edges.edge_flats = batch_edge_flats
        edges.boxes = edges.boxes[edge_indices]
        edges.feats = edges.feats[edge_indices]

        return edges

    def _edge_flats_from_adj_mat(self, presence_logits, nodes_per_img):
        edge_flats = []
        num_edges = torch.minimum(torch.ones(len(presence_logits)) * self.edges_per_node,
                Tensor(nodes_per_img) - 1).int()

        for pl, ne, nn in zip(presence_logits, num_edges, nodes_per_img):
            if nn == 0:
                edge_flats.append(torch.zeros(0, 2).to(pl.device).int())
                continue

            row_indices = torch.arange(nn).to(pl.device).view(-1, 1).repeat(1, ne.item())
            topk_indices = torch.topk(pl, k=ne.item(), dim=1).indices
            edge_flat = torch.stack([row_indices, topk_indices[:nn]], dim=-1).long().view(-1, 2)
            edge_flats.append(edge_flat)

        return edge_flats

    def predict(self, results: SampleList, feats: BaseDataElement) -> Tuple[BaseDataElement]:
        nodes_per_img = [len(r.pred_instances.bboxes) for r in results]

        # build edges
        edges = self._build_edges(results, nodes_per_img, feats)

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

        # update node viz feats (leave semantic feats the same!)
        feats.instance_feats = graph.nodes.feats[..., :self.viz_feat_size]

        return feats, graph

    def _construct_graph(self, feats: BaseDataElement, edges: BaseDataElement,
            nodes_per_img: List) -> BaseDataElement:
        graph = BaseDataElement()
        graph.edges = edges

        # move result data into nodes
        nodes = BaseDataElement()
        nodes.feats = torch.cat([feats.instance_feats, feats.semantic_feats], -1)
        nodes.nodes_per_img = nodes_per_img
        graph.nodes = nodes

        return graph

    def _update_graph(self, graph: BaseDataElement, dgl_g: dgl.DGLGraph) -> BaseDataElement:
        # update node viz feats (leave semantic feats the same, add to original feats)
        updated_node_feats = pad_sequence(dgl_g.ndata['feats'].split(graph.nodes.nodes_per_img),
                batch_first=True)
        graph.nodes.feats[..., :self.viz_feat_size] += updated_node_feats[..., :self.viz_feat_size]

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
        graph.edges.boxes = dgl_g.edata['boxes']
        graph.edges.feats += dgl_g.edata['feats']

        return graph

    def _build_gt_graph(self, result: SampleList) -> BaseDataElement:
        num_boxes = len(result)

        bounding_boxes = result.bbox.tensor
        centroids = (bounding_boxes[:, :2] + bounding_boxes[:, 2:]) / 2

        distance_x = centroids[:, 0, None] - centroids[:, 0]
        distance_y = centroids[:, 1, None] - centroids[:, 1]

        relationships = []

        # FIRST COMPUTE INSIDE-OUTSIDE MASK

        # compute areas of all boxes and create meshgrid
        areas = box_area(bounding_boxes)
        areas_x, areas_y = torch.meshgrid(areas, areas)

        # compute intersection
        intersection = box_intersection(bounding_boxes, bounding_boxes)

        # inside-outside is when intersection is close to the area of the smaller box
        inside_outside_matrix = intersection / torch.minumum(torch.stack(areas_x, areas_y), dim=0)
        inside_outside_mask = (inside_outside_matrix >= 0.8)

        # COMPUTE LEFT-RIGHT, ABOVE-BELOW, INSIDE-OUTSIDE MASKS
        left_right_mask = ((angle_matrix > -45) & (angle_matrix <= 45)) | \
                ((angle_matrix > 135) | (angle_matrix <= -135))
        above_below_mask = ((angle_matrix > 45) & (angle_matrix <= 135))

        # left right and above below are only when inside outside is False
        left_right_mask = left_right_mask.int() * (1 - inside_outside_mask) # 1 for left-right
        above_below_mask = above_below_mask.int() * (1 - inside_outside_mask) * 2 # 2 for above-below
        inside_outside_mask = inside_outside_mask.int() * 3 # 3 for inside-outside

        relationships.extend(torch.nonzero(left_right_mask, as_tuple=False).tolist())
        relationships.extend(torch.nonzero(above_below_mask, as_tuple=False).tolist())
        relationships.extend(inside_outside_mask)

        # SELECT E EDGES PER NODE BASED ON gIoU
        iou_matrix.fill_diagonal_(all_box_ious.min().item() - 1) # first artificially set diagonal to min - 1
        num_edges = min(self.num_edges, iou_matrix.shape[0] - 1)
        selected_edges = torch.topk(iou_matrix, num_edges, dim=1).indices

        # COMPUTE EDGE FLATS (PAIRS OF OBJECT IDS CORRESPONDING TO EACH EDGE)
        edge_flats = []
        for e in selected_edges.unbind(1):
            # edge flats is just arange, each column of selected edges
            edge_flats.append(torch.stack([torch.arange(len(e)).to(e.device), e], dim=1).long())

        edge_flats = torch.cat(edge_flats)

        # COMPUTE EDGE BOXES AND SELECT USING EDGE FLATS
        edge_boxes = self.box_union(boxes, boxes)
        selected_edge_boxes = edge_boxes[edge_flats[:, 0], edge_flats[:, 1]]

        # SELECT RELATIONSHIPS USING EDGE FLATS
        selected_relations = relationships[edge_flats[:, 0], edge_flats[:, 1]]

        # add edge flats, boxes, and relationships to gt_graph structure
        gt_edges = BaseDataElement()
        gt_edges.edge_flats = edge_flats
        gt_edges.edge_boxes = selected_edge_boxes
        gt_edges.edge_relations = selected_relations

        return gt_edges

    def box_union(self, boxes1, boxes2):
        # boxes1, boxes2: Tensors of shape (N1, 4) and (N2, 4) representing bounding boxes in (x1, y1, x2, y2) format
        N1 = boxes1.shape[0]
        N2 = boxes2.shape[0]

        # Expand dimensions to perform broadcasting
        boxes1 = boxes1.unsqueeze(1)  # (N1, 1, 4)
        boxes2 = boxes2.unsqueeze(0)  # (1, N2, 4)

        # Compute the coordinates of the union bounding boxes
        union_x1 = torch.min(boxes1[:, :, 0], boxes2[:, :, 0])  # (N1, N2)
        union_y1 = torch.min(boxes1[:, :, 1], boxes2[:, :, 1])  # (N1, N2)
        union_x2 = torch.max(boxes1[:, :, 2], boxes2[:, :, 2])  # (N1, N2)
        union_y2 = torch.max(boxes1[:, :, 3], boxes2[:, :, 3])  # (N1, N2)

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
        intersection_width = torch.clamp(intersection_x2 - intersection_x1 + 1, min=0)  # (B, N1, N2)
        intersection_height = torch.clamp(intersection_y2 - intersection_y1 + 1, min=0)  # (B, N1, N2)
        intersection_area = intersection_width * intersection_height  # (B, N1, N2)

        return intersection_area

    def loss_and_predict(self, results: SampleList, feats: BaseDataElement) -> Tuple[SampleList, dict]:
        # init loss dict
        losses = {}

        # build edges for GT
        gt_edges = self._build_gt_edges(results)

        # build edges and compute presence probabilities
        edges = self._build_edges(results, feats)

        # compute edge presence loss
        edge_presence_loss = self.edge_presence_loss(edges, gt_edges)

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
        #edge_classifier_loss = self.edge_classifier_loss(graph.edges, gt_edges)

        # update node viz feats (leave semantic feats the same!)
        feats.instance_feats = graph.nodes.feats[..., :self.viz_feat_size]

        # update losses
        losses.update(edge_presence_loss)
        #losses.update(edge_classifier_loss)

        return losses, feats, graph

    def edge_presence_loss(self, edges, gt_edges):
        # first match edge boxes to gt edge boxes
        pred_matched_inds, pred_unmatched_inds, _ = self.match_boxes(edges.edge_boxes, gt_edges.edge_boxes)

        # assign labels (1 if matched, 0 if unmatched)
        training_inds = torch.cat([pred_matched_inds, pred_unmatched_inds])
        breakpoint()
        flat_edge_relations = edges.presence_logits.view(-1, edges.presence_logits.shape[-1])[training_inds]
        edge_presence_gt = torch.cat([torch.ones_like(pred_matched_inds), torch.zeros_like(pred_unmatched_inds)])

        presence_loss = self.presence_loss(flat_edge_relations, edge_presence_gt)

        return {'loss_edge_presence': presence_loss}

    def edge_classifier_loss(self, edges, gt_edges):
        raise NotImplementedError

    def match_boxes(predicted_boxes, gt_boxes, iou_threshold, iou_lower_bound=0.0):
        # predicted_boxes: Tensor of shape (B, N, 4) representing predicted bounding boxes in (x1, y1, x2, y2) format
        # gt_boxes: Tensor of shape (B, M, 4) representing ground truth bounding boxes in (x1, y1, x2, y2) format
        # iou_threshold: IoU threshold for matching
        # iou_lower_bound: Lower bound on IoU for returning unmatched boxes

        B, N, _ = predicted_boxes.shape
        _, M, _ = gt_boxes.shape

        overlaps = bbox_overlaps(predicted_boxes.view(-1, 4), gt_boxes.view(-1, 4)).view(B, N, M)  # Compute IoU between predicted and ground truth boxes
        max_overlaps, argmax_overlaps = overlaps.max(dim=2)  # Find the maximum IoU and its corresponding index along the M dimension

        pred_matched_indices = torch.nonzero(max_overlaps >= iou_threshold, as_tuple=False).squeeze()  # Get the indices of matched boxes
        pred_unmatched_indices = torch.nonzero(max_overlaps < iou_lower_bound, as_tuple=False).squeeze()  # Get the indices of unmatched boxes

        gt_matched_indices = argmax_overlaps.view(-1)[matched_indices]

        gt_matched_indices = argmax_overlaps[matched_indices]

        return pred_matched_indices, pred_unmatched_indices, gt_matched_indices
