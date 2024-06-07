import os

_base_ = os.path.expandvars('$MMDETECTION/configs/_base_/datasets/youtube_vis.py')

dataset_type = 'VideoDatasetWithDS'
data_root='data/mmdet_datasets/sages'
metainfo = {
    'classes': ('cystic_plate', 'calot_triangle', 'cystic_artery', 'cystic_duct',
        'gallbladder', 'tool'),
    'palette': [(255, 255, 100), (102, 178, 255), (255, 0, 0), (0, 102, 51), (51, 255, 103), (255, 151, 53)]
}
num_temp_frames = 5

train_data_prefix = 'train'
val_data_prefix = 'val'
test_data_prefix = 'test'

# aug
rand_aug_surg = [
        #[dict(type='ShearX', level=8)],
        #[dict(type='ShearY', level=8)],
        #[dict(type='Rotate', level=8)],
        #[dict(type='TranslateX', level=8)],
        #[dict(type='TranslateY', level=8)],
        [dict(type='AutoContrast', level=8)],
        [dict(type='Equalize', level=8)],
        [dict(type='Contrast', level=8)],
        [dict(type='Color', level=8)],
        [dict(type='Brightness', level=8)],
        [dict(type='Sharpness', level=8)],
]

train_pipeline = [
    dict(
        type='UniformRefFrameSampleWithPad',
        num_ref_imgs=num_temp_frames-1,
        frame_range=[1-num_temp_frames, 0],
        filter_key_img=True,
    ),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotationsWithDS', with_mask=False),
            dict(type='Resize', scale=(399, 224), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomAffine', max_rotate_degree=15, max_translate_ratio=0.05,
                max_shear_degree=5, scaling_ratio_range=(1, 1)),
        ]
    ),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='RandAugment',
                aug_space=rand_aug_surg,
            ),
        ],
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
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(399, 224), keep_ratio=True),
            dict(type='LoadTrackAnnotationsWithDS', with_mask=False),
        ]),
    dict(
        type='PackTrackInputs',
        meta_keys=('ds', 'is_det_keyframe', 'is_ds_keyframe', 'lg'),
    ),
]

train_dataloader=dict(
    _delete_=True,
    batch_size=20,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='TrackCustomKeyframeSampler'),
    batch_sampler=dict(type='TrackAspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotation_coco_vid.json',
        data_prefix=dict(img_path=train_data_prefix),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
        metainfo=metainfo,
    ),
)

val_dataloader=dict(
    batch_size=20,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_delete_=True, type='TrackCustomKeyframeSampler'),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annotation_coco_vid.json',
        data_prefix=dict(img_path=val_data_prefix),
        test_mode=True,
        pipeline=eval_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
        metainfo=metainfo,
    )
)

test_dataloader=dict(
    batch_size=20,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_delete_=True, type='TrackCustomKeyframeSampler'),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='test/annotation_coco_vid.json',
        data_prefix=dict(img_path=test_data_prefix),
        test_mode=True,
        pipeline=eval_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
        metainfo=metainfo,
    )
)
