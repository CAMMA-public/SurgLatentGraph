import os

_base_ = os.path.expandvars('$MMDETECTION/configs/_base_/datasets/coco_panoptic.py')

# Modify dataset related settings

data_root='data/endoscapes_mmdet'
metainfo = {
    'classes': ('cystic_plate', 'calot_triangle', 'cystic_artery', 'cystic_duct',
        'gallbladder', 'tool'),
    'palette': [(255, 255, 100), (102, 178, 255), (255, 0, 0), (0, 102, 51), (51, 255, 103), (255, 151, 53)]
}

rand_aug_surg = [
        [dict(type='ShearX', level=8)],
        [dict(type='ShearY', level=8)],
        [dict(type='Rotate', level=8)],
        [dict(type='TranslateX', level=8)],
        [dict(type='TranslateY', level=8)],
        [dict(type='Rotate', level=8)],
        [dict(type='AutoContrast', level=8)],
        [dict(type='Equalize', level=8)],
        [dict(type='Contrast', level=8)],
        [dict(type='Color', level=8)],
        [dict(type='Brightness', level=8)],
        [dict(type='Sharpness', level=8)],
    ]
train_pipeline = [
        dict(type='LoadImageFromFile', file_client_args={{_base_.file_client_args}}),
        dict(type='LoadPanopticAnnotations',
            with_bbox=True,
            with_mask=True,
            with_seg=True,
            file_client_args={{_base_.file_client_args}},
        ),
        dict(
            type='Resize',
            scale=(399, 224),
            keep_ratio=True,
        ),
        dict(
            type='RandomFlip',
            prob=0.5,
        ),
        dict(
            type='RandAugment',
            aug_space=rand_aug_surg,
        ),
        dict(
            type='Color',
            min_mag = 0.6,
            max_mag = 1.4,
        ),
        #dict(
        #    type='RandomErasing',
        #    n_patches=(0, 1),
        #    ratio=(0.3, 1),
        #),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
        dict(type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                'flip', 'flip_direction', 'homography_matrix')
        ),
        #dict(type='PackDetInputs',
        #    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
        #        'homography_matrix')
        #),
]

eval_pipeline = [
        dict(type='LoadImageFromFile', file_client_args={{_base_.file_client_args}}),
        dict(
            type='Resize',
            scale=(399, 224),
            keep_ratio=True,
        ),
        dict(type='LoadPanopticAnnotations',
            file_client_args={{_base_.file_client_args}},
        ),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
        ),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_panopt_coco.json',
        data_prefix=dict(img='train/', seg='panoptic_masks/'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_panopt_coco.json',
        data_prefix=dict(img='val/', seg='panoptic_masks/'),
        test_mode=True,
        pipeline=eval_pipeline,
    )
)

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/annotation_panopt_coco.json',
        data_prefix=dict(img='test/', seg='panoptic_masks/'),
        test_mode=True,
        pipeline=eval_pipeline,
    )
)

# metric
val_evaluator = [
    #dict(
    #    type='CocoPanopticMetric',
    #    ann_file=os.path.join(data_root, 'val/annotation_panopt_coco.json'),
    #    seg_prefix=os.path.join(data_root, 'panoptic_masks'),
    #    file_client_args={{_base_.file_client_args}}
    #),
    dict(
        type='CocoMetric',
        ann_file=os.path.join(data_root, 'val/annotation_coco.json'),
        metric=['bbox', 'segm'],
    )
]

test_evaluator = [
    #dict(
    #    type='CocoPanopticMetric',
    #    ann_file=os.path.join(data_root, 'test/annotation_panopt_coco.json'),
    #    seg_prefix=os.path.join(data_root, 'panoptic_masks'),
    #    file_client_args={{_base_.file_client_args}}
    #),
    dict(
        type='CocoMetric',
        ann_file=os.path.join(data_root, 'test/annotation_coco.json'),
        metric=['bbox', 'segm'],
        outfile_prefix='./work_dirs/coco_panoptic/test'
    )
]
