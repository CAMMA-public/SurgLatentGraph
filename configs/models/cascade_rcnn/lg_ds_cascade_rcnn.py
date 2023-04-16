import os
import copy

# modify base for different detectors
_base_ = [
    '../lg_ds_base.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/cascade-rcnn_r50_fpn.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head = [
        dict(type='Shared2FCBBoxHead', num_classes=6),
        dict(type='Shared2FCBBoxHead', num_classes=6),
        dict(type='Shared2FCBBoxHead', num_classes=6)
]
detector.test_cfg.rcnn.max_per_img = _base_.num_nodes

dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
del _base_.model
del detector.data_preprocessor

# extract lg config, set detector
model = copy.deepcopy(_base_.lg_model)
model.data_preprocessor = dp
model.detector = detector
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
model.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.roi_extractor.roi_layer.output_size = 1
del _base_.lg_model

# modify load_from
load_from = 'weights/lg_cascade_rcnn_no_recon.pth'
