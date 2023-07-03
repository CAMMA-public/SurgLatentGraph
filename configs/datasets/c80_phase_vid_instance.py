import os

_base_ = [os.path.expandvars('$MMDETECTION/configs/_base_/datasets/youtube_vis.py')]
custom_imports = dict(imports=['datasets.custom_loading'], allow_failed_imports=False)

dataset_type = 'VideoDatasetWithDS'
data_root = 'data/mmdet_datasets/cholec80/'
metainfo = {
    'classes': ('abdominal_wall', 'liver', 'gastrointestinal_wall', 'fat', 'grasper',
        'connective_tissue', 'blood', 'cystic_duct', 'hook', 'gallbladder', 'hepatic_vein',
        'liver_ligament'),
}
num_temp_frames = 5

train_data_prefix = 'train_phase'
val_data_prefix = 'val_phase'
test_data_prefix = 'test_phase'

train_pipeline = [
    dict(
        type='UniformRefFrameSample',
        num_ref_imgs=4,
        frame_range=[-4, 0],
        filter_key_img=True,
    ),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotationsWithDS', with_mask=True),
            dict(type='Resize', scale=(399, 224), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]
    ),
    dict(
        type='PackTrackInputs',
        meta_keys=('ds', 'is_det_keyframe'),
    ),
]

eval_pipeline = [
    dict(
        type='UniformRefFrameSample',
        num_ref_imgs=num_temp_frames-1,
        frame_range=[1-num_temp_frames, 0],
        filter_key_img=False,
    ),
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(399, 224), keep_ratio=True),
            dict(type='LoadTrackAnnotations', with_mask=True),
        ]),
    dict(
        type='PackTrackInputs',
        meta_keys=('ds'),
    ),
]

train_dataloader=dict(
    _delete_=True,
    batch_size=4,
    num_workers=0,
    #persistent_workers=True,
    sampler=dict(type='TrackCustomKeyframeSampler'),
    batch_sampler=dict(type='TrackAspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_phase/annotation_coco_vid.json',
        data_prefix=dict(img_path=train_data_prefix),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
    ),
)

val_dataloader=dict(
    sampler=dict(_delete_=True, type='TrackCustomKeyframeSampler'),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='val_phase/annotation_coco_vid.json',
        data_prefix=dict(img_path=val_data_prefix),
        test_mode=True,
        pipeline=eval_pipeline,
    )
)

test_dataloader=dict(
    sampler=dict(_delete_=True, type='TrackCustomKeyframeSampler'),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='test_phase/annotation_coco_vid.json',
        data_prefix=dict(img_path=test_data_prefix),
        test_mode=True,
        pipeline=eval_pipeline,
    )
)
