import os
import copy

# modify base for different detectors
_base_ = [
    '../lg_ds_base.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/mask-rcnn_r50_fpn.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head.num_classes = _base_.num_classes
detector.roi_head.mask_head.num_classes = _base_.num_classes
detector.test_cfg.rcnn.max_per_img = _base_.num_nodes
dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
dp.pad_mask = False
del _base_.model
del detector.data_preprocessor

# extract lg config, set detector
model = copy.deepcopy(_base_.lg_model)
model.data_preprocessor = dp
model.detector = detector
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
model.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.roi_extractor.roi_layer.output_size = 1

# trainable bb, neck
model.trainable_backbone_cfg=copy.deepcopy(detector.backbone)
model.trainable_backbone_cfg.frozen_stages = _base_.trainable_backbone_frozen_stages
if 'neck' in detector:
    model.trainable_neck_cfg=copy.deepcopy(detector.neck)

del _base_.lg_model

# modify load_from
load_from = _base_.load_from.replace('base', 'mask_rcnn')

#remove semantic features
model.ds_head.final_viz_feat_size = 0
model.ds_head.use_img_feats = False
