import os

_base_ = [
    os.path.expandvars('$MMDETECTION/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic_no_base.py'),
    '../datasets/endoscapes_detection_panoptic.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/default_runtime.py')
]

# preprocessing
batch_augments = []
data_preprocessor = dict(
    batch_augments = [],
)

# model
num_things_classes = 6
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    panoptic_head = dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1],
        ),
    ),
    panoptic_fusion_head = dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
    ),
    data_preprocessor = dict(
        batch_augments = [],
    )
)

# load weights
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'

# hooks
log_config = dict( # config to register logger hook
    interval=50, # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook', by_epoch=False),
        dict(type='MMDetWandbHook', by_epoch=True, init_kwargs=
            {
                'entity': "adit98",
                'project': "lg-surg",
            }),
    ]
)

# learning policy
max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]
auto_scale_lr = dict(enable=True, base_batch_size=8)
