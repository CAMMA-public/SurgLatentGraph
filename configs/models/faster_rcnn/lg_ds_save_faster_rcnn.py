_base_ = 'lg_save_faster_rcnn.py'

# trainable bb, neck
model.trainable_backbone_cfg=copy.deepcopy(detector.backbone)
model.trainable_backbone_cfg.frozen_stages=_base_.trainable_backbone_frozen_stages
if 'neck' in detector:
    model.trainable_neck_cfg=copy.deepcopy(detector.neck)
