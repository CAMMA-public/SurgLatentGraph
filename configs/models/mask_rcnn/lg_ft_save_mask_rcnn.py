import copy

_base_ = 'lg_save_mask_rcnn.py'

# trainable bb, neck
model.trainable_backbone_cfg=copy.deepcopy(_base_.detector.backbone)
model.trainable_backbone_cfg.frozen_stages=_base_.trainable_backbone_frozen_stages
if 'neck' in _base_.detector:
    model.trainable_neck_cfg=copy.deepcopy(_base_.detector.neck)

load_from = _base_.load_from.replace('lg', 'lg_ds')
