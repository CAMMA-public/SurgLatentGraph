import copy
import os

_base_=['../lg_base_seg.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/cascade-mask-rcnn_r50_fpn.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head = [
        dict(type='Shared2FCBBoxHead', num_classes=_base_.num_classes),
        dict(type='Shared2FCBBoxHead', num_classes=_base_.num_classes),
        dict(type='Shared2FCBBoxHead', num_classes=_base_.num_classes)
]
detector.roi_head.mask_head.num_classes = 6
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
model.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.roi_extractor.roi_layer.output_size = 1
del _base_.lg_model

# modify load_from
load_from = 'weights/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco_20210628_164719-5bdc3824_LG.pth'

# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
)
auto_scale_lr = dict(enable=True)
