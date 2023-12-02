import os

_base_ = 'endoscapes_vid_instance.py'
num_temp_frames = _base_.num_temp_frames

del _base_.train_pipeline[2]
_base_.train_pipeline[1] = dict(
    type='TransformBroadcaster',
    transforms=[
        dict(type='LoadLG', saved_graph_dir='latent_graphs/endoscapes/base'),
        dict(type='LoadTrackAnnotationsWithDS', with_mask=False),
    ],
)

_base_.eval_pipeline[1] = dict(
    type='TransformBroadcaster',
    transforms=[
        dict(type='LoadLG', saved_graph_dir='latent_graphs/endoscapes/base'),
        dict(type='LoadTrackAnnotationsWithDS', with_mask=False),
    ],
)

train_dataloader=dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        pipeline=_base_.train_pipeline,
    ),
)

val_dataloader=dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        pipeline=_base_.eval_pipeline,
    ),
)

test_dataloader=dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        pipeline=_base_.eval_pipeline,
    ),
)
