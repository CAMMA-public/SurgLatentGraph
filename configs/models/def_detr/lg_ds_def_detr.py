import os
import copy

# modify base for different detectors
_base_ = [
    '../lg_ds_base.py',
    os.path.expandvars('$MMDETECTION/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco_no_base.py'),
]
custom_imports = dict(imports=_base_.custom_imports.imports + ['model.modified_detectors.def_detr_with_queries'],
        allow_failed_imports=False)

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.type = 'DeformableDETRWithQueries'
detector.bbox_head.type = 'DeformableDETRHeadWithIndices'
detector.as_two_stage = True
detector.with_box_refine = True
detector.bbox_head.num_classes = _base_.num_classes
detector.test_cfg.max_per_img = _base_.num_nodes

dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
del _base_.model
del detector.data_preprocessor

# extract lg config, set detector
model = copy.deepcopy(_base_.lg_model)

# trainable detector
model.trainable_detector_cfg = copy.deepcopy(detector)
model.trainable_detector_cfg.backbone.frozen_stages=_base_.trainable_backbone_frozen_stages

model.data_preprocessor = dp
model.detector = detector
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
del _base_.lg_model

# modify load_from
load_from = _base_.load_from.replace('base', 'def_detr')

# modify optim
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'trainable_backbone.backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)
