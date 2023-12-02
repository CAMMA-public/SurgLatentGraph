import os

_base_ = 'c80_phase_vid_instance_load_graphs.py'
num_temp_frames = _base_.num_temp_frames

_base_.train_pipeline[0] = dict(
    type='AllFramesSample',
    sampling_ratio=4,
)
_base_.train_pipeline[1].transforms[0].load_keyframes_only = True

_base_.eval_pipeline[0] = dict(
    type='AllFramesSample',
    sampling_ratio=4,
)
_base_.eval_pipeline[1].transforms[0].load_keyframes_only = True

train_dataloader=dict(
    batch_size=1,
    dataset=dict(
        pipeline=_base_.train_pipeline,
    ),
)

val_dataloader=dict(
    batch_size=1,
    dataset=dict(
        pipeline=_base_.eval_pipeline,
    ),
)

test_dataloader=dict(
    batch_size=1,
    dataset=dict(
        pipeline=_base_.eval_pipeline,
    ),
)
