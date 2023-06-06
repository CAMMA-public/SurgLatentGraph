import copy
import os

_base_='../lg_faster_rcnn.py'

# load ssl backbone weights
_base_.model.detector.backbone.init_cfg.checkpoint='weights/ssl_weights/no_phase/converted_moco_lap.torch'
load_from = None

# training schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]
train_cfg = dict(
    max_epochs=50,
)
