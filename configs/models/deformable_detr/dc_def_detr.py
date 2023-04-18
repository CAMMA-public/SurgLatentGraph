import os
import copy

# modify base for different detectors
_base_ = [
    '../deepcvs_base.py',
    os.path.expandvars('$MMDETECTION/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco_no_base.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.as_two_stage = True
detector.with_box_refine = True
detector.bbox_head.num_classes = 6
detector.test_cfg.max_per_img = _base_.num_nodes

dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
dp.pad_mask = False
del _base_.model
del detector.data_preprocessor

# extract lg config, set detector
model = copy.deepcopy(_base_.dc_model)
model.data_preprocessor = dp
model.detector = detector
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
del _base_.dc_model

# modify load_from
load_from = 'weights/lg_def_detr_no_recon.pth'
