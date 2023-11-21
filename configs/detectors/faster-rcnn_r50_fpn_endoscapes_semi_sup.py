import copy

_base_ = 'faster-rcnn_r50_fpn_endoscapes.py'

# define aug spaces for surg
color_space_surg = [
        [dict(type='AutoContrast', level=8)],
        [dict(type='Equalize', level=8)],
        [dict(type='Contrast', level=8)],
        [dict(type='Color', level=8)],
        [dict(type='Brightness', level=8)],
        [dict(type='Sharpness', level=8)],
]

geom_space_surg = [
        [dict(type='ShearX', level=8)],
        [dict(type='ShearY', level=8)],
        [dict(type='Rotate', level=8)],
        [dict(type='TranslateX', level=8)],
        [dict(type='TranslateY', level=8)],
]

# define pipelines
branch_field = ['sup', 'unsup_teacher', 'unsup_student']
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsWithDS', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(399, 224), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=_base_.rand_aug_surg),
    dict(type='Color', min_mag=0.6, max_mag=1.4),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor', 'flip', 'flip_direction',
                       'homography_matrix', 'ds')
        ),
    ),
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type='Resize', scale=(399, 224), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type='Resize', scale=(399, 224), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space_surg, aug_num=1),
            dict(type='RandAugment', aug_space=geom_space_surg, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(399, 224), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
    ),
]

labeled_dataset = dict(
    type='CocoDatasetWithDS',
    data_root=_base_.data_root,
    ann_file='train/annotation_coco.json',
    data_prefix=dict(img='train'),
    pipeline=sup_pipeline,
    metainfo=_base_.metainfo,
    filter_cfg=dict(filter_empty_gt=True),
)

unlabeled_dataset = dict(
    type='CocoDatasetWithDS',
    data_root=_base_.data_root,
    ann_file='unlabeled/annotation_coco.json',
    data_prefix=dict(img='unlabeled/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    metainfo=_base_.metainfo,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=10,
    num_workers=5,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=10,
        source_ratio=[2, 8]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset])
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDatasetWithDS',
        data_root=_base_.data_root,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
    )
)

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDatasetWithDS',
        data_root=_base_.data_root,
        ann_file='test/annotation_coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
    )
)

# define semi-sup model
detector = _base_.model
model = dict(
    _delete_=True,
    type='SoftTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=4.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher')
)

# evaluation
val_evaluator = dict(metric=['bbox'])
test_evaluator = dict(metric=['bbox'])
eval_interval = 1000

# hooks
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=eval_interval))
log_processor = dict(by_epoch=False)

# ema update
custom_hooks = [dict(type='MeanTeacherHook')]

# test teacher and student
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=20000, val_interval=eval_interval)
val_cfg = dict(type='TeacherStudentValLoop')

# lr
auto_scale_lr = dict(enable=True)
