import os

_base_ = ['simple_cvs_classifier_with_recon.py']

model = dict(
    backbone=dict(
        init_cfg=dict(
            checkpoint='weights/ssl_weights/no_phase/converted_moco_lap.torch',
        ),
    ),
)
