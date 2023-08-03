import os

_base_ = 'c80_phase_vid_instance_5.py'
num_temp_frames = _base_.num_temp_frames

train_pipeline = [
    dict(
        type='AllFramesSample',
        sampling_ratio=6,
    ),
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadLG', saved_graph_dir='latent_graphs/c80_phase_faster_rcnn',
                skip_keys=['boxesA', 'boxesB'], load_keyframes_only=True),
            dict(type='LoadTrackAnnotationsWithDS', with_mask=False),
        ]
    ),
    dict(
        type='PackTrackInputs',
        meta_keys=('ds', 'is_det_keyframe', 'is_ds_keyframe', 'lg'),
    ),
]

eval_pipeline = [
    dict(
        type='AllFramesSample',
        sampling_ratio=6,
    ),
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadLG', saved_graph_dir='latent_graphs/c80_phase_faster_rcnn',
                skip_keys=['boxesA', 'boxesB'], load_keyframes_only=True),
            dict(type='LoadTrackAnnotationsWithDS', with_mask=False),
        ],
    ),
    dict(
        type='PackTrackInputs',
        meta_keys=('ds', 'is_det_keyframe', 'is_ds_keyframe', 'lg'),
    ),
]

train_dataloader=dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        pipeline=train_pipeline,
    ),
)

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        pipeline=eval_pipeline,
    ),
)

test_dataloader=dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        pipeline=eval_pipeline,
    ),
)
