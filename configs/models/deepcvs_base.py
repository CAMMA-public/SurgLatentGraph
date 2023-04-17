import os

# dataset, optimizer, and runtime cfgs
_base_ = [
    '../datasets/endoscapes_instance.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/schedules/schedule_1x.py'),
    os.path.expandvars('$MMDETECTION/configs/_base_/default_runtime.py')
]

data_root = _base_.data_root
val_data_prefix = _base_.val_dataloader.dataset.data_prefix.img
test_data_prefix = _base_.test_dataloader.dataset.data_prefix.img

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['model.deepcvs', 'evaluator.CocoMetricRGD', 'hooks.custom_hooks'],
        allow_failed_imports=False)
num_nodes = 16

dc_model = dict(
    type='DeepCVS',
    num_classes=3,
    detector_num_classes=len(_base_.metainfo.classes),
    num_nodes=num_nodes,
    decoder_backbone=dict(
        type='ResNet',
        in_channels=4+len(_base_.metainfo.classes), # 3 channels + detector_num_classes + 1 (bg)
        depth=18,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
    ),
    loss=dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        class_weight=[3.19852941, 4.46153846, 2.79518072],
    ),
    use_pred_boxes_recon_loss=True,
)

# dataset
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='train/annotation_cvs_coco.json',
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='val/annotation_cvs_coco.json',
    ),
)
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='test/annotation_cvs_coco.json',
    ),
)

# evaluators
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.val_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'val/annotation_cvs_coco.json'),
        use_pred_boxes_recon=True,
        metric=[],
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.test_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'test/annotation_cvs_coco.json'),
        metric=[],
        #additional_metrics = ['reconstruction'],
        use_pred_boxes_recon=True,
        outfile_prefix='./results/endoscapes_preds/test'
    ),
]

# optimizer
del _base_.param_scheduler
del _base_.optim_wrapper
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001),
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
custom_hooks = [dict(type="FreezeDetectorHook")]
default_hooks = dict(
    checkpoint=dict(save_best='endoscapes/ds_average_precision'),
)
