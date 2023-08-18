import os

# dataset, optimizer, and runtime cfgs
_base_ = [
    '../datasets/c80_phase/c80_phase_instance.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/schedules/schedule_1x.py'),
    os.path.expandvars('$MMDETECTION/configs/_base_/default_runtime.py')
]

data_root = _base_.data_root
val_data_prefix = _base_.val_dataloader.dataset.data_prefix.img
test_data_prefix = _base_.test_dataloader.dataset.data_prefix.img

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['model.simple_cvs_predictor', 'evaluator.CocoMetricRGD'], allow_failed_imports=False)

model = dict(
    type='SimpleCVSPredictor',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50'
        )
    ),
    loss=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[1.9209686, 0.19571111, 0.98499113, 0.2993076, 1.9426803, 1,  2.20292951],
    )
    num_classes=7,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
    ),
)

# dataset
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='train_phase/annotation_phase_coco.json',
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='val_phase/annotation_phase_coco.json',
    ),
)
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='test_phase/annotation_phase_coco.json',
    ),
)

# evaluators
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_phase',
        data_root=_base_.data_root,
        data_prefix=_base_.val_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'val_phase/annotation_phase_coco.json'),
        use_pred_boxes_recon=True,
        metric=[],
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_phase',
        data_root=_base_.data_root,
        data_prefix=_base_.test_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'test_phase/annotation_phase_coco.json'),
        metric=[],
        #additional_metrics = ['reconstruction'],
        use_pred_boxes_recon=True,
        outfile_prefix='./results/c80_phase_preds/test'
    ),
]

# optimizer
del _base_.param_scheduler
del _base_.optim_wrapper
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00001),
)
auto_scale_lr = dict(enable=False)

# Running settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='c80_phase/ds_f1', rule='greater'),
)
