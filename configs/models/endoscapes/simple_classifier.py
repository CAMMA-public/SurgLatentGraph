import os

# dataset, optimizer, and runtime cfgs
_base_ = [
    '../datasets/endoscapes/endoscapes_instance.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/schedules/schedule_1x.py'),
    os.path.expandvars('$MMDETECTION/configs/_base_/default_runtime.py')
]

data_root = _base_.data_root
val_data_prefix = _base_.val_dataloader.dataset.data_prefix.img
test_data_prefix = _base_.test_dataloader.dataset.data_prefix.img

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['model.simple_classifier', 'evaluator.CocoMetricRGD'], allow_failed_imports=False)

model = dict(
    type='SimpleClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
            checkpoint='torchvision://resnet50')),
    loss=dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        class_weight=[3.19852941, 4.46153846, 2.79518072],
    ),
    num_classes=3,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
)

# dataset
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        ann_file='train/annotation_ds_coco.json',
        filter_cfg=dict(filter_empty_gt=False),
    ),
)
train_eval_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        ann_file='train/annotation_ds_coco.json',
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        ann_file='val/annotation_ds_coco.json',
    ),
)
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        ann_file='test/annotation_ds_coco.json',
    ),
)

# evaluators
train_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=_base_.data_root,
    data_prefix=_base_.train_eval_dataloader.dataset.data_prefix.img,
    ann_file=os.path.join(_base_.data_root, 'train/annotation_ds_coco.json'),
    use_pred_boxes_recon=True,
    metric=[],
    num_classes=3,
    outfile_prefix='./results/endoscapes_preds/train/r50',
)
val_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=_base_.data_root,
    data_prefix=_base_.val_dataloader.dataset.data_prefix.img,
    ann_file=os.path.join(_base_.data_root, 'val/annotation_ds_coco.json'),
    use_pred_boxes_recon=True,
    metric=[],
    num_classes=3,
    outfile_prefix='./results/endoscapes_preds/val/r50',
)

test_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=_base_.data_root,
    data_prefix=_base_.test_dataloader.dataset.data_prefix.img,
    ann_file=os.path.join(_base_.data_root, 'test/annotation_ds_coco.json'),
    metric=[],
    num_classes=3,
    #additional_metrics = ['reconstruction'],
    use_pred_boxes_recon=True,
    outfile_prefix='./results/endoscapes_preds/test/r50',
)

# optimizer
del _base_.param_scheduler
optim_wrapper = dict(
    _delete_=True,
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
    checkpoint=dict(save_best='endoscapes/ds_average_precision'),
)
