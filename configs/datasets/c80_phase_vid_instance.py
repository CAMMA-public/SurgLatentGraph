import os

_base_ = [os.path.expandvars('$MMTRACKING/configs/_base_/datasets/youtube_vis.py')]
custom_imports = dict(imports=['datasets.custom_loading'], allow_failed_imports=False)

dataset_type = 'VideoDatasetWithDS'
data_root = 'data/mmdet_datasets/cholec80/'
num_temp_frames = 5
train_data_prefix = 'train_phase'
val_data_prefix = 'val_phase'
test_data_prefix = 'test_phase'

train_dataloader=dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='train_phase/annotation_coco_vid.json',
        data_prefix=dict(img=train_data_prefix),
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=num_temp_frames,
            frame_range=[-num_temp_frames, 0],
            filter_key_img=True,
            method='uniform',
        )
    )
)

val_dataloader=dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='val_phase/annotation_coco_vid.json',
        data_prefix=dict(img=val_data_prefix),
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=num_temp_frames,
            frame_range=[-num_temp_frames, 0],
            filter_key_img=True,
            method='uniform',
        ),
        test_mode=True,
    )
)

test_dataloader=dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='test_phase/annotation_coco_vid.json',
        data_prefix=dict(img=test_data_prefix),
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=num_temp_frames,
            frame_range=[-num_temp_frames, 0],
            filter_key_img=True,
            method='uniform',
        ),
        test_mode=True,
    )
)
