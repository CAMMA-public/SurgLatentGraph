import os

_base_ = ['../simple_classifier.py']

model = dict(
    backbone=dict(
        init_cfg=dict(
            checkpoint='weights/ssl_weights/moco/converted_lap_nissen.torch',
        ),
    ),
)
