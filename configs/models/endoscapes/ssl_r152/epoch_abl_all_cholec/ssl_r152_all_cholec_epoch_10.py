import os

_base_ = ['../../simple_classifier_r152.py']

model = dict(
    backbone=dict(
        init_cfg=dict(
            checkpoint='weights/trained_ssl/all_cholec_resnet152/moco/converted_model_phase10.torch'
        ),
    ),
)
