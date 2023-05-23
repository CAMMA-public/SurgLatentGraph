import copy

# modify base for different detectors
_base_ = ['../lg_ds_faster_rcnn_no_recon.py']

trainable_backbone_init = 'weights/ssl_weights/no_phase/converted_moco_lap.torch'
_base_.model.trainable_backbone_cfg.frozen_stages = -1
_base_.model.trainable_backbone_cfg.init_cfg.checkpoint = trainable_backbone_init

# neck
#trainable_neck_cfg = copy.deepcopy(_base_.model.detector.neck)
_base_.model.trainable_neck_cfg = dict(
    type='ChannelMapper',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
)

# modify model
_base_.model.trainable_backbone_cfg = trainable_backbone_cfg
_base_.model.trainable_neck_cfg = trainable_neck_cfg

custom_hooks = [dict(type="FreezeDetectorHook")]#, dict(type='CopyDetectorBackbone')]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0003),
)
