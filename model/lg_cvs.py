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
from mmengine.structures import BaseDataElement
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import bbox2roi, roi2bbox, scale_boxes
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector
from .predictor_heads.reconstruction import ReconstructionHead
from .predictor_heads.modules.loss import ReconstructionLoss
from .predictor_heads.graph import GraphHead
from .predictor_heads.ds import DSHead

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
            use_pred_boxes_recon_loss: bool = False, reconstruction_head: ConfigType = None,
            reconstruction_loss: ConfigType = None, reconstruction_img_stats: ConfigType = None,
            graph_head: ConfigType = None, ds_head: ConfigType = None, roi_extractor: ConfigType = None,
            trainable_detector_cfg: OptConfigType = None, trainable_backbone_cfg: OptConfigType = None,
            trainable_neck_cfg: OptConfigType = None, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.detector = MODELS.build(detector)
        self.roi_extractor = MODELS.build(roi_extractor) if roi_extractor is not None else None

        # if trainable detector cfg is defined, that is used for trainable backbone
        if trainable_detector_cfg is not None:
            self.trainable_backbone = MODELS.build(trainable_detector_cfg)
        elif trainable_backbone_cfg is not None:
            bb = MODELS.build(trainable_backbone_cfg)
            if trainable_neck_cfg is not None:
                neck = MODELS.build(trainable_neck_cfg)
                self.trainable_backbone = torch.nn.Sequential(OrderedDict([
                        ('backbone', bb),
                        ('neck', neck),
                    ])
                )

            else:
                self.trainable_backbone = torch.nn.Sequential(OrderedDict([
                        ('backbone', bb),
                        ('neck', torch.nn.Identity()),
                    ])
                )
        else:
            self.trainable_backbone = None

        # add obj feat size to recon cfg
        if reconstruction_head is not None:
            reconstruction_head.obj_feat_size = 256 # HACK get this value from cfg
            self.reconstruction_head = MODELS.build(reconstruction_head)
        else:
            self.reconstruction_head = None

        self.reconstruction_loss = MODELS.build(reconstruction_loss) if reconstruction_loss is not None else None
        self.reconstruction_img_stats = reconstruction_img_stats if reconstruction_img_stats is not None else None

        # add roi extractor to graph head
        graph_head.roi_extractor = self.roi_extractor
        self.graph_head = MODELS.build(graph_head) if graph_head is not None else None
        self.ds_head = MODELS.build(ds_head) if ds_head is not None else None
        self.semantic_feat_projector = torch.nn.Linear(num_classes + 4, semantic_feat_size)
        self.use_pred_boxes_recon_loss = use_pred_boxes_recon_loss

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        if self.detector.training:
            losses = self.detector.loss(batch_inputs, batch_data_samples)
        else:
            losses = {}

        # now run detector predict fn, we will pass the output of that to reconstructor
        with torch.no_grad():
            self.detector.training = False
            results = self.detector.predict(batch_inputs, batch_data_samples)
            detached_results = self.detach_results(results)
            self.detector.training = True

        # get bb and fpn features TODO(adit98) see if we can prevent running bb twice
        feats = self.extract_feat(batch_inputs, detached_results)

        # run graph head
        if self.graph_head is not None:
            # TODO(adit98) train graph
            feats, graph = self.graph_head.predict(detached_results, feats)

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
            except AttributeError:
                raise NotImplementedError("Must have graph head in order to do downstream prediction")

        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        # run detector to get detections
        results = self.detector.predict(batch_inputs, batch_data_samples)
        detached_results = copy.deepcopy(self.detach_results(results))

        # get bb and fpn features TODO(adit98) see if we can prevent running bb twice
        feats = self.extract_feat(batch_inputs, detached_results)

        # run graph head
        if self.graph_head is not None:
            feats, graph = self.graph_head.predict(detached_results, feats)

            # TODO(adit98) add graph to result

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

    def detach_results(self, results: SampleList) -> SampleList:
        for i in range(len(results)):
            results[i].pred_instances.bboxes = results[i].pred_instances.bboxes.detach()
            results[i].pred_instances.labels = results[i].pred_instances.labels.detach()

        return results

    def extract_feat(self, batch_inputs: Tensor, results: SampleList) -> BaseDataElement:
        feats = BaseDataElement()

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
            boxes = [r.pred_instances.bboxes for r in results]
            boxes_per_img = [len(b) for b in boxes]
            scale_factor = results[0].scale_factor
            rescaled_boxes = scale_boxes(torch.cat(boxes), scale_factor).split(boxes_per_img)
            rois = bbox2roi(rescaled_boxes)

            # extract roi feats
            roi_input_feats = feats.neck_feats if feats.neck_feats is not None else feats.bb_feats
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
        classes = [r.pred_instances.labels for r in results]
        boxes = [r.pred_instances.bboxes for r in results]
        c = pad_sequence(classes, batch_first=True)
        b = pad_sequence(boxes, batch_first=True)
        b_norm = b / Tensor(results[0].ori_shape).flip(0).repeat(2).to(b.device)
        c_one_hot = F.one_hot(c, num_classes=self.num_classes)
        s = self.semantic_feat_projector(torch.cat([b_norm, c_one_hot], -1))
        feats.semantic_feats = s

        return feats

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError
