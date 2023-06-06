import copy

# modify base for different detectors
_base_ = ['../lg_ds_faster_rcnn_no_recon.py']

# modify model

trainable_backbone_init = 'weights/ssl_weights/no_phase/converted_moco_lap.torch'
_base_.model.trainable_backbone_cfg.frozen_stages = -1
_base_.model.trainable_backbone_cfg.init_cfg.checkpoint = trainable_backbone_init

# neck
_base_.model.trainable_neck_cfg = dict(
    type='ChannelMapper',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
)

custom_hooks = [dict(type="FreezeDetectorHook")]#, dict(type='CopyDetectorBackbone')]
