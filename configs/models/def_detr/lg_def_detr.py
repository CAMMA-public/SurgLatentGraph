import copy
import os

_base_=['../lg_base_box.py',
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
model.data_preprocessor = dp
model.detector = detector
del _base_.lg_model

# modify load_from
load_from = 'weights/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff_LG.pth'

# optimizer
del _base_.optim_wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)

# learning policy
max_epochs = 30
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[22],
        gamma=0.1)
]
