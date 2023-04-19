from typing import List, Tuple, Union
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.nn.utils.rnn import pad_sequence
from mmdet.registry import MODELS
from mmengine.structures import BaseDataElement
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import scale_boxes
from mmdet.utils import ConfigType
from mmdet.models.detectors.base import BaseDetector

@MODELS.register_module()
class DeepCVS(BaseDetector):
    """Detector that also outputs downstream predictions.

    Args:
        detector (ConfigType): underlying object detector config
        num_classes: int
        decoder_backbone (ConfigType): backbone for downstream decoder
        reconstruction_head (ConfigType): reconstruction head config
        reconstruction_loss (ConfigType): reconstruction loss config
        reconstruction_img_stats (ConfigType): reconstructed image mean and std
    """
    def __init__(self,
            detector: ConfigType,
            num_classes: int,
            detector_num_classes: int,
            decoder_backbone: ConfigType,
            loss: ConfigType,
            num_nodes: int,
            use_pred_boxes_recon_loss: bool = False,
            reconstruction_head: ConfigType = None,
            reconstruction_loss: ConfigType = None,
            reconstruction_img_stats: ConfigType = None,
            **kwargs):
        super().__init__(**kwargs)
        self.detector = MODELS.build(detector)
        self.detector_num_classes = detector_num_classes
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.decoder_backbone = MODELS.build(decoder_backbone)
        self.decoder_predictor = torch.nn.Linear(self.decoder_backbone.feat_dim,
                self.num_classes)
        self.loss_fn = MODELS.build(loss)

        # add obj feat size to recon cfg
        if reconstruction_head is not None:
            reconstruction_head.obj_feat_size = self.decoder_backbone.feat_dim
            self.reconstruction_head = MODELS.build(reconstruction_head)
        else:
            self.reconstruction_head = None

        self.reconstruction_loss = MODELS.build(reconstruction_loss) if reconstruction_loss is not None else None
        self.reconstruction_img_stats = reconstruction_img_stats if reconstruction_img_stats is not None else None

        self.use_pred_boxes_recon_loss = use_pred_boxes_recon_loss

    def _construct_layout(self, layout_size, classes, boxes, masks=None):
        box_layout = torch.zeros(len(boxes), self.num_nodes, *layout_size).to(boxes[0].device)
        mask_layout = None

        # build box layout from boxes, classes (used with backgroundized img as input to reconstructor)
        for img_id, (label, box) in enumerate(zip(classes, boxes)):
            for instance_id, (l, b) in enumerate(zip(label, box.round().int())):
                box_layout[img_id][instance_id, b[1]:b[3], b[0]:b[2]] = l + 1 # labels are 0-indexed

        if masks is not None:
            mask_layouts = []
            for ind, (label, mask) in enumerate(zip(classes, masks)):
                mask_layouts.append(mask.int() * (label + 1).unsqueeze(-1).unsqueeze(-1))

            mask_layout = pad_sequence(mask_layouts, batch_first=True)

            # to construct layout, one hot encode mask_layout, sum across instance dim
            mask_ohl = F.one_hot(mask_layout, self.detector_num_classes + 1)
            if mask_ohl.shape[1] == 0:
                layout = torch.zeros(*mask_ohl.transpose(1, -1).shape[:-1]).to(mask_ohl.device).float()
            else:
                layout = mask_ohl.transpose(1, -1).max(-1).values.float()

        else:
            box_ohl = F.one_hot(box_layout.long(), self.detector_num_classes + 1)
            layout = box_ohl.transpose(1, -1).max(-1).values.float()

        # one hot layout is based on box layout
        one_hot_layout = (box_layout > 0).int() # stack of instance box masks
        box_ohl = F.one_hot(box_layout.long(), self.detector_num_classes + 1)
        box_layout = box_ohl.transpose(1, -1).max(-1).values.float()

        return box_layout, layout, one_hot_layout

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        # run detector to get detections
        results = self.detector.predict(batch_inputs, batch_data_samples)

        # get feats
        ds_feats = self.extract_feat(batch_inputs, results)

        # reconstruction if recon head is not None
        recon_imgs, _, _ = self.reconstruct(batch_inputs, results, ds_feats)

        ds_preds = self.decoder_predictor(ds_feats)
        for r, dp, r_img in zip(results, ds_preds, recon_imgs):
            r.pred_ds = dp
            if r_img is not None:
                # renormalize img
                norm_r_img = r_img * Tensor(self.reconstruction_img_stats.std).view(-1, 1, 1).to(r_img.device) / 255 + \
                        Tensor(self.reconstruction_img_stats.mean).view(-1, 1, 1).to(r_img.device) / 255
                r.reconstruction = torch.clamp(norm_r_img, 0, 1)

        return results

    def reconstruct(self, batch_inputs: Tensor, results: SampleList, ds_feats: Tensor) -> Tensor:
        if self.reconstruction_head is None:
            return [None] * len(results), None, None

        feats = BaseDataElement()
        feats.bb_feats = ds_feats
        feats.instance_feats = ds_feats.unsqueeze(1).repeat(1, 16, 1)
        feats.semantic_feats = None
        reconstructed_imgs, img_targets, rescaled_results = \
                self.reconstruction_head.predict(results, feats, batch_inputs)

        return reconstructed_imgs, img_targets, rescaled_results

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        losses = {}

        # run detector
        results = self.detector.predict(batch_inputs, batch_data_samples)

        # get feats
        ds_feats = self.extract_feat(batch_inputs, results)

        # reconstruction and loss if recon head is not None
        recon_imgs, img_targets, rescaled_results = self.reconstruct(
                batch_inputs, results, ds_feats)

        if self.reconstruction_loss is not None:
            if self.use_pred_boxes_recon_loss:
                recon_boxes = [r.pred_instances.bboxes for r in rescaled_results]
            else:
                recon_boxes = [r.gt_instances.bboxes for r in rescaled_results]

            recon_losses = self.reconstruction_loss(recon_imgs, img_targets, recon_boxes)
            losses.update(recon_losses)

        # ds prediction and loss
        ds_preds = self.decoder_predictor(ds_feats)

        # get gt
        ds_gt = torch.stack([torch.from_numpy(b.ds) for b in batch_data_samples]).to(
                ds_preds.device).round()

        # update loss
        losses.update({'ds_loss': self.loss_fn(ds_preds, ds_gt)})

        return losses

    def extract_feat(self, batch_inputs: Tensor, results: SampleList):
        # extract quantities (resize)
        detections_size = results[0].ori_shape
        img_size = results[0].batch_input_shape
        scale_factor = (Tensor(img_size) / Tensor(detections_size)).flip(0).to(batch_inputs.device)

        classes = [r.pred_instances.labels for r in results]
        boxes = [scale_boxes(r.pred_instances.bboxes, scale_factor) for r in results]
        if 'masks' in results[0].pred_instances:
            masks = []
            for r in results:
                if r.pred_instances.masks.shape[0] == 0:
                    masks.append(torch.zeros(0, *img_size).to(r.pred_instances.masks.device))
                else:
                    masks.append(TF.resize(r.pred_instances.masks, img_size,
                        interpolation=InterpolationMode.NEAREST))

            _, layout, _ = self._construct_layout(img_size, classes, boxes, masks)

        else:
            layout, _, _  = self._construct_layout(img_size, classes, boxes)

        # concatenate layout and batch inputs
        decoder_input = torch.cat([batch_inputs, layout], 1)

        # ds prediction
        ds_feats = F.adaptive_avg_pool2d(self.decoder_backbone(decoder_input)[-1],
                1).squeeze(-1).squeeze(-1)

        return ds_feats

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError
