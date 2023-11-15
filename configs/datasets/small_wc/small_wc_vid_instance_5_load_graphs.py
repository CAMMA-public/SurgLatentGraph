import os

_base_ = 'small_wc_vid_instance_5.py'
num_temp_frames = _base_.num_temp_frames

train_pipeline = [
    dict(
        type='UniformRefFrameSampleWithPad',
        num_ref_imgs=num_temp_frames-1,
        frame_range=[1-num_temp_frames, 0],
        filter_key_img=True,
    ),
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadLG', saved_graph_dir='',
                skip_keys=['boxesA', 'boxesB']),
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
        type='UniformRefFrameSampleWithPad',
        num_ref_imgs=num_temp_frames-1,
        frame_range=[1-num_temp_frames, 0],
        filter_key_img=True,
    ),
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadLG', saved_graph_dir='',
                skip_keys=['boxesA', 'boxesB']),
            dict(type='LoadTrackAnnotationsWithDS', with_mask=False),
        ]),
    dict(
        type='PackTrackInputs',
        meta_keys=('ds', 'is_det_keyframe', 'is_ds_keyframe', 'lg'),
    ),
]

train_dataloader=dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        pipeline=train_pipeline,
    ),
)

val_dataloader=dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        pipeline=eval_pipeline,
    ),
)

test_dataloader=dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        pipeline=eval_pipeline,
    ),
)
