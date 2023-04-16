import os
import wandb

_base_ = [
    os.path.expandvars('$MMDETECTION/configs/_base_/models/cascade-mask-rcnn_r50_fpn.py'),
    '../datasets/endoscapes_detection.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/schedules/schedule_1x.py'), os.path.expandvars('$MMDETECTION/configs/_base_/default_runtime.py')
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=[dict(type='Shared2FCBBoxHead', num_classes=6),
            dict(type='Shared2FCBBoxHead', num_classes=6),
            dict(type='Shared2FCBBoxHead', num_classes=6)],
        mask_head=dict(type='FCNMaskHead', num_classes=6))
)

# Running settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco_20210628_164719-5bdc3824.pth'

# hooks
log_config = dict( # config to register logger hook
    interval=50, # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
        dict(type='MMDetWandbHook', by_epoch=False, init_kwargs=
            {
                'entity': "adit98",
                'project': "lg-surg",
                'dir': 'work_dirs',
            }),
    ]
)
