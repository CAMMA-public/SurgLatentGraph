import os

# dataset, optimizer, and runtime cfgs
_base_ = [
    '../datasets/cholecT50/cholecT50_instance.py',
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
        use_sigmoid=True,
        #class_weight=[509.25971751, 62.23765034, 1261.8771661, 522.52123691,
        #    107.07235718, 1497.27899735, 414.56522561, 9.48319277,
        #    4558.98357534, 600.7399111, 200.85322204, 760.01910511,
        #    10.26945419, 219.53186448, 85.42097956, 1120.89562161,
        #    33.28156643, 1.13297828, 65.59468477, 3.73075545,
        #    9.64965248, 185.1038516, 132.09563499, 229.57893881,
        #    1034.23428624, 706.49910163, 619.28051674, 116.0517523,
        #    153.87498791, 19.9279709, 152.70395706, 660.02090042,
        #    543.76118355, 372.26367968, 275.24196753, 732.28250766,
        #    115.05360826, 226.98193047, 4558.98357534, 418.89251046,
        #    490.58330895, 4558.98357534, 50000.       , 1223.40837409,
        #    289.12278457, 694.27648781, 1759.92934145, 3714.93211832,
        #    960.01159736, 2539.51559931, 4093.9055069, 193.12076046,
        #    217.15601218, 551.23016177, 50000, 50000,
        #    1840.65034386, 17.51831417, 6.68426371, 14.72803927,
        #    1.6683155, 11.90194223, 141.40461424, 87.46898612,
        #    326.79270198, 2254.21876106, 257.57589956, 1840.65034386,
        #    82.43792936, 62.04520322, 1929.13200356, 441.95861309,
        #    1618.01454846, 706.49910163, 3134.59276304, 760.01910511,
        #    427.82387102, 1061.59308535, 47.61613105, 27.54731454,
        #    2907.49147626, 745.89304387, 15.15634122, 960.01159736,
        #    732.28250766, 3714.93211832, 1346.55946356, 398.11465218,
        #    120.94841204, 2711.07412066, 334.97597966, 2539.51559931,
        #    125.88039584, 446.88005762, 11.72802889, 76.61473705,
        #    14.0820279, 242.04080732, 135.66816954, 91.24775109
        #],
    ),
    num_classes=100,
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
        ann_file='train/annotation_ds_coco.json',
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='val/annotation_ds_coco.json',
    ),
)
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='test/annotation_ds_coco.json',
    ),
)

# evaluators
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='cholecT50',
        data_root=_base_.data_root,
        data_prefix=_base_.val_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'val/annotation_ds_coco.json'),
        use_pred_boxes_recon=True,
        metric=[],
        num_classes=100,
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='cholecT50',
        data_root=_base_.data_root,
        data_prefix=_base_.test_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'test/annotation_ds_coco.json'),
        metric=[],
        num_classes=100,
        #additional_metrics = ['reconstruction'],
        use_pred_boxes_recon=True,
        outfile_prefix='./results/cholecT50_preds/test'
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
    checkpoint=dict(save_best='cholecT50/ds_average_precision', rule='greater'),
)
