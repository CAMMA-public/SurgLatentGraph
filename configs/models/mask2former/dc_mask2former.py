import os
import copy

# modify base for different detectors
_base_ = [
    '../deepcvs_base.py',
    '../../_base_/mask2former_r50_8xb2-lsj-50e_coco-panoptic_no_base.py',
]

# extract detector, data preprocessor config from base
num_things_classes = 6
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
detector = copy.deepcopy(_base_.model)
detector.panoptic_head.num_things_classes = num_things_classes
detector.panoptic_head.num_stuff_classes = num_stuff_classes
detector.panoptic_head.loss_cls=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=2.0,
    reduction='mean',
    class_weight=[1.0] * num_classes + [0.1],
)
detector.panoptic_fusion_head.num_things_classes = num_things_classes
detector.panoptic_fusion_head.num_stuff_classes = num_stuff_classes
detector.test_cfg.max_per_image = _base_.num_nodes

dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
dp.pad_mask = False
dp.batch_augments = []
del _base_.model
del detector.data_preprocessor

# extract lg config, set detector
model = copy.deepcopy(_base_.dc_model)
model.data_preprocessor = dp
model.detector = detector
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
del _base_.dc_model

# modify load_from
load_from = _base_.load_from.replace('base', 'mask2former')
