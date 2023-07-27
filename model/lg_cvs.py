import copy
import warnings
from typing import List, Tuple, Union
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.nn.utils.rnn import pad_sequence

from mmdet.registry import MODELS
from mmengine.structures import BaseDataElement, InstanceData
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import bbox2roi, roi2bbox, scale_boxes
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector
from mmengine.model.base_module import Sequential
from .predictor_heads.reconstruction import ReconstructionHead
from .predictor_heads.modules.loss import ReconstructionLoss
from .predictor_heads.modules.layers import build_mlp
from .predictor_heads.graph import GraphHead
from .predictor_heads.ds import DSHead
from .roi_extractors.sg_single_level_roi_extractor import SgSingleRoIExtractor

@MODELS.register_module()
class LGDetector(BaseDetector):
    """Detector that also outputs a scene graph, reconstructed image, and any downstream predictions.

    Args:
        detector (ConfigType): underlying object detector config
        reconstruction_head (ConfigType): reconstruction head config
        reconstruction_loss (ConfigType): reconstruction loss config
        reconstruction_img_stats (ConfigType): reconstructed image mean and std
    """

    def __init__(self, detector: ConfigType, num_classes: int, semantic_feat_size: int,
            semantic_feat_projector_layers: int = 3, perturb_factor: float = 0.0,
            use_pred_boxes_recon_loss: bool = False, reconstruction_head: ConfigType = None,
            reconstruction_loss: ConfigType = None, reconstruction_img_stats: ConfigType = None,
            graph_head: ConfigType = None, ds_head: ConfigType = None, roi_extractor: ConfigType = None,
            use_gt_dets: bool = False, trainable_detector_cfg: OptConfigType = None,
            trainable_backbone_cfg: OptConfigType = None,
            trainable_neck_cfg: OptConfigType = None, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.detector = MODELS.build(detector)
        self.roi_extractor = MODELS.build(roi_extractor) if roi_extractor is not None else None
        self.use_gt_dets = use_gt_dets
        self.perturb_factor = perturb_factor if not use_gt_dets else 0

        # if trainable detector cfg is defined, that is used for trainable backbone
        if trainable_detector_cfg is not None:
            self.trainable_backbone = MODELS.build(trainable_detector_cfg)
        elif trainable_backbone_cfg is not None:
            bb = MODELS.build(trainable_backbone_cfg)
            if trainable_neck_cfg is not None:
                neck = MODELS.build(trainable_neck_cfg)
                self.trainable_backbone = Sequential(OrderedDict([
                        ('backbone', bb),
                        ('neck', neck),
                    ])
                )

            else:
                self.trainable_backbone = Sequential(OrderedDict([
                        ('backbone', bb),
                        ('neck', torch.nn.Identity()),
                    ])
                )
        else:
            self.trainable_backbone = None

        # add obj feat size to recon cfg
        if reconstruction_head is not None:
            reconstruction_head.obj_viz_feat_size = graph_head.viz_feat_size
            self.reconstruction_head = MODELS.build(reconstruction_head)
        else:
            self.reconstruction_head = None

        self.reconstruction_loss = MODELS.build(reconstruction_loss) if reconstruction_loss is not None else None
        self.reconstruction_img_stats = reconstruction_img_stats if reconstruction_img_stats is not None else None

        # add roi extractor to graph head
        graph_head.roi_extractor = self.roi_extractor
        self.graph_head = MODELS.build(graph_head) if graph_head is not None else None
        self.ds_head = MODELS.build(ds_head) if ds_head is not None else None

        # build semantic feat projector (input feat size is classes+box coords+score)
        dim_list = [num_classes + 5] + [512] * (semantic_feat_projector_layers - 1) + [semantic_feat_size]
        self.semantic_feat_projector = build_mlp(dim_list, batch_norm='batch')

        self.use_pred_boxes_recon_loss = use_pred_boxes_recon_loss

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        if self.detector.training:
            losses = self.detector.loss(batch_inputs, batch_data_samples)
        else:
            losses = {}

        # extract LG
        feats, graph, detached_results, _, _ = self.extract_lg(batch_inputs,
                batch_data_samples)

        # use feats and detections to reconstruct img
        if self.reconstruction_head is not None:
            reconstructed_imgs, img_targets, rescaled_results = self.reconstruction_head.predict(
                    detached_results, feats, batch_inputs)

            if self.use_pred_boxes_recon_loss:
                recon_boxes = [r.pred_instances.bboxes for r in rescaled_results]
            else:
                recon_boxes = [r.gt_instances.bboxes for r in rescaled_results]

            reconstruction_losses = self.reconstruction_loss(reconstructed_imgs,
                    img_targets, recon_boxes)

            # update losses
            losses.update(reconstruction_losses)

        if self.ds_head is not None:
            try:
                ds_losses = self.ds_head.loss(graph, feats, batch_data_samples)
                losses.update(ds_losses)
            except AttributeError as e:
                print(e)
                raise NotImplementedError("Must have graph head in order to do downstream prediction")

        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        # extract LG
        feats, graph, detached_results, results, gt_edges = self.extract_lg(batch_inputs,
                batch_data_samples)

        if gt_edges is not None:
            # add graph to result
            for ind, r in enumerate(results):
                # GT
                r.gt_edges = InstanceData()
                r.gt_edges.edge_flats = gt_edges.edge_flats[ind]
                r.gt_edges.edge_boxes = gt_edges.edge_boxes[ind]
                r.gt_edges.relations = gt_edges.edge_relations[ind]

                # PRED
                r.pred_edges = InstanceData()

                # select correct batch
                batch_inds = graph.edges.edge_flats[:, 0] == ind
                r.pred_edges.edge_flats = graph.edges.edge_flats[batch_inds][:, 1:] # remove batch id
                r.pred_edges.edge_boxes = graph.edges.boxes[ind] # already a list
                r.pred_edges.relations = graph.edges.class_logits[batch_inds]

                # LATENT GRAPH

                # extract graph for frame i, add to result
                g_i = BaseDataElement()
                g_i.nodes = BaseDataElement()
                g_i.edges = BaseDataElement()
                g_i.nodes.feats = graph.nodes.feats[ind]
                g_i.nodes.nodes_per_img = graph.nodes.nodes_per_img[ind]
                g_i.nodes.bboxes = r.pred_instances.bboxes
                g_i.nodes.scores = r.pred_instances.scores
                g_i.nodes.labels = r.pred_instances.labels

                # split edge quantities and add
                epi = graph.edges.edges_per_img.tolist()
                for k in graph.edges.keys():
                    if k in ['batch_index', 'presence_logits', 'edges_per_img']:
                        continue

                    elif k == 'edge_flats':
                        val = graph.edges.get(k).split(epi)[ind][:, 1:]

                    elif not isinstance(graph.edges.get(k), Tensor):
                        # no need to split, just index
                        val = graph.edges.get(k)[ind]

                    else:
                        val = graph.edges.get(k).split(epi)[ind]

                    g_i.edges.set_data({k: val})

                # pool img feats and add to graph
                g_i.img_feats = F.adaptive_avg_pool2d(feats.bb_feats[-1][ind], 1).squeeze()

                # add graph to result
                r.lg = g_i

        # use feats and detections to reconstruct img
        if self.reconstruction_head is not None:
            reconstructed_imgs, _, _ = self.reconstruction_head.predict(detached_results,
                    feats, batch_inputs)

            for r, r_img in zip(results, reconstructed_imgs):
                # renormalize img
                norm_r_img = r_img * Tensor(self.reconstruction_img_stats.std).view(-1, 1, 1).to(r_img.device) / 255 + \
                        Tensor(self.reconstruction_img_stats.mean).view(-1, 1, 1).to(r_img.device) / 255
                r.reconstruction = torch.clamp(norm_r_img, 0, 1)

        if self.ds_head is not None:
            try:
                ds_preds = self.ds_head.predict(graph, feats)
            except AttributeError:
                raise NotImplementedError("Must have graph head in order to do downstream prediction")

            for r, dp in zip(results, ds_preds):
                r.pred_ds = dp

        return results

    def extract_lg(self, batch_inputs: Tensor, batch_data_samples: SampleList,
            force_perturb: bool = False) -> Tuple[BaseDataElement]:
        # run detector to get detections
        with torch.no_grad():
            detector_is_training = self.detector.training
            self.detector.training = False
            results = self.detector.predict(batch_inputs, batch_data_samples)
            detached_results = self.detach_results(results)
            self.detector.training = detector_is_training

        # get bb and fpn features
        feats = self.extract_feat(batch_inputs, detached_results, force_perturb=force_perturb)

        # run graph head
        if self.graph_head is not None:
            if self.detector.training:
                # train graph with gt boxes (only when detector is training)
                graph_losses, feats, graph = self.graph_head.loss_and_predict(
                        detached_results, feats)
                losses.update(graph_losses)
                gt_edges = None

            else:
                feats, graph, gt_edges = self.graph_head.predict(detached_results, feats)

        return feats, graph, detached_results, results, gt_edges

    def detach_results(self, results: SampleList) -> SampleList:
        for i in range(len(results)):
            results[i].pred_instances.bboxes = results[i].pred_instances.bboxes.detach()
            results[i].pred_instances.labels = results[i].pred_instances.labels.detach()

        return results

    def extract_feat(self, batch_inputs: Tensor, results: SampleList, force_perturb: bool = False) -> BaseDataElement:
        feats = BaseDataElement()
        if self.use_gt_dets:
            boxes = [r.gt_instances.bboxes.to(batch_inputs.device) for r in results]
            classes = [r.gt_instances.labels.to(batch_inputs.device) for r in results]
            scores = [torch.ones_like(c) for c in classes]
        else:
            boxes = [r.pred_instances.bboxes for r in results]
            classes = [r.pred_instances.labels for r in results]
            scores = [r.pred_instances.scores for r in results]

        # apply box perturbation
        if (self.training or force_perturb) and self.perturb_factor > 0:
            boxes = self.box_perturbation(boxes, results[0].img_shape)

        # run bbox feat extractor and add instance feats to feats
        if self.roi_extractor is not None:
            if self.trainable_backbone is not None:
                backbone = self.trainable_backbone.backbone
                neck = self.trainable_backbone.neck
            else:
                backbone = self.detector.backbone
                neck = self.detector.neck if self.detector.with_neck else torch.nn.Identity()

            bb_feats = backbone(batch_inputs)
            neck_feats = neck(bb_feats)

            feats.bb_feats = bb_feats
            feats.neck_feats = neck_feats

            # rescale bboxes, convert to rois
            boxes_per_img = [len(b) for b in boxes]
            scale_factor = results[0].scale_factor
            rescaled_boxes = scale_boxes(torch.cat(boxes), scale_factor).split(boxes_per_img)
            rois = bbox2roi(rescaled_boxes)

            # extract roi feats
            roi_input_feats = feats.neck_feats if feats.neck_feats is not None else feats.bb_feats
            if isinstance(self.roi_extractor, SgSingleRoIExtractor) and 'masks' in results[0].pred_instances:
                masks = torch.cat([r.pred_instances.masks for r in results])
                roi_feats = self.roi_extractor(
                    roi_input_feats[:self.roi_extractor.num_inputs], rois, masks=masks,
                )

            else:
                roi_feats = self.roi_extractor(
                    roi_input_feats[:self.roi_extractor.num_inputs], rois
                )

            # pool feats and split into list
            # TODO(adit98) experiment with multiplying by instance mask before mean/sum
            feats.instance_feats = pad_sequence(roi_feats.squeeze(-1).squeeze(-1).split(boxes_per_img),
                    batch_first=True)

        else:
            # instance feats are just queries (run detector.get_queries to get)
            if self.trainable_backbone is not None:
                feats.bb_feats, feats.neck_feats, feats.instance_feats = \
                        self.trainable_backbone.get_queries(batch_inputs, results)
            else:
                feats.bb_feats, feats.neck_feats, feats.instance_feats = \
                        self.detector.get_queries(batch_inputs, results)

        # compute semantic feat
        c = pad_sequence(classes, batch_first=True)
        b = pad_sequence(boxes, batch_first=True)
        s = pad_sequence(scores, batch_first=True)
        b_norm = b / Tensor(results[0].ori_shape).flip(0).repeat(2).to(b.device)
        c_one_hot = F.one_hot(c, num_classes=self.num_classes)
        s = self.semantic_feat_projector(torch.cat([b_norm, c_one_hot, s.unsqueeze(-1)], -1).flatten(end_dim=1))
        feats.semantic_feats = s.view(b_norm.shape[0], b_norm.shape[1], s.shape[-1])

        return feats

    def box_perturbation(self, boxes: List[Tensor], image_shape: Tuple):
        boxes_per_img = [len(b) for b in boxes]
        perturb_factor = min(self.perturb_factor, 1)
        xmin, ymin, xmax, ymax = torch.cat(boxes).unbind(1)

        # compute x and y perturbation ranges
        h = xmax - xmin
        w = ymax - ymin

        # generate random numbers drawn from (-h, h), (-w, w), multiply by perturb factor
        perturb = perturb_factor * (torch.rand(4).to(boxes[0]) * torch.stack([h, w], dim=1).repeat(1, 2) - \
                torch.stack([h, w], dim=1).repeat(1, 2))

        perturbed_boxes = torch.cat(boxes) + perturb

        # ensure boxes are valid (clamp from 0 to img shape)
        perturbed_boxes = torch.maximum(torch.zeros_like(perturbed_boxes), perturbed_boxes)
        stacked_img_shapes = Tensor(image_shape).flip(0).unsqueeze(0).repeat(perturbed_boxes.shape[0],
                2).to(perturbed_boxes)
        perturbed_boxes = torch.minimum(stacked_img_shapes, perturbed_boxes)

        return perturbed_boxes.split(boxes_per_img)

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError
