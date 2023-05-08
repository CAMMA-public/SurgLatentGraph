import os
import copy

# modify base for different detectors
_base_ = [
    '../deepcvs_base.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/cascade-mask-rcnn_r50_fpn.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head = [
        dict(type='Shared2FCBBoxHead', num_classes=6),
        dict(type='Shared2FCBBoxHead', num_classes=6),
        dict(type='Shared2FCBBoxHead', num_classes=6)
]
detector.roi_head.mask_head.num_classes = 6
detector.test_cfg.rcnn.max_per_img = _base_.num_nodes
dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
dp.pad_mask = False
del _base_.model
del detector.data_preprocessor

# extract dc config, set detector
model = copy.deepcopy(_base_.dc_model)
model.data_preprocessor = dp
model.detector = detector
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
del _base_.dc_model

# modify load_from
load_from = _base_.load_from.replace('base', 'cascade_mask_rcnn')
