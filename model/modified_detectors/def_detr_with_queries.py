from mmdet.registry import MODELS
from mmdet.models.detectors import DeformableDETR
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.structures import InstanceData
import torch
from torch import Tensor
from typing import Tuple, Dict

@MODELS.register_module()
class DeformableDETRWithQueries(DeformableDETR):
    def extract_feat(self, batch_inputs: Tensor, return_all: bool = False) -> Tuple[Tuple[Tensor]]:
        bb_feats = self.backbone(batch_inputs)
        if self.with_neck:
            neck_feats = self.neck(bb_feats)
        else:
            neck_feats = bb_feats

        if return_all:
            return bb_feats, neck_feats
        else:
            return neck_feats

    def get_queries(self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tensor:

        bb_feats, neck_feats = self.extract_feat(batch_inputs, return_all=True)
        head_inputs_dict = self.forward_transformer(neck_feats, batch_data_samples)

        # select top queries
        queries = torch.stack([q[r.pred_instances.selected_inds] for q, r in zip(
            head_inputs_dict['hidden_states'][-1], batch_data_samples)])

        return bb_feats, neck_feats, queries

@MODELS.register_module()
class DeformableDETRHeadWithIndices(DeformableDETRHead):
    def _predict_by_feat_single(self,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = True) -> InstanceData:
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = torch.div(indexes, self.num_classes, rounding_mode='floor')
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.selected_inds = bbox_index

        return results
