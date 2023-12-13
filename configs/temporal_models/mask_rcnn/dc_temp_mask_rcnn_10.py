import copy

_base_ = [
    '../dc_temp_10_base.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/mask-rcnn_r50_fpn.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head.num_classes = _base_.dc_model.detector_num_classes
detector.roi_head.mask_head.num_classes = _base_.dc_model.detector_num_classes
detector.test_cfg.rcnn.max_per_img = _base_.num_nodes
del _base_.model
del detector.data_preprocessor

# init model
model = copy.deepcopy(_base_.dc_model)

# configure lg detector
model.detector = detector
model.reconstruction_img_stats = dict(
    mean=model.data_preprocessor.mean,
    std=model.data_preprocessor.std,
)
del _base_.dc_model

# modify load_from
load_from = _base_.load_from.replace('base', 'mask_rcnn')
