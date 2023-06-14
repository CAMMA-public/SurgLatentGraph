import os

_base_ = os.path.expandvars('$MMDETECTION/configs/_base_/datasets/coco_instance.py')
custom_imports = dict(imports=['datasets.custom_loading'], allow_failed_imports=False)

# Modify dataset related settings

data_root='data/mmdet_datasets/cholec80'
metainfo = {
    'classes': ('abdominal_wall', 'liver', 'gastrointestinal_wall', 'fat', 'grasper',
        'connective_tissue', 'blood', 'cystic_duct', 'hook', 'gallbladder', 'hepatic_vein',
        'liver_ligament'),
}
#metainfo = {
#    'classes': ('cystic_plate', 'calot_triangle', 'cystic_artery', 'cystic_duct',
#        'gallbladder', 'tool'),
#    'palette': [(255, 255, 100), (102, 178, 255), (255, 0, 0), (0, 102, 51), (51, 255, 103), (255, 151, 53)]
#}

rand_aug_surg = [
        [dict(type='ShearX', level=8)],
        [dict(type='ShearY', level=8)],
        [dict(type='Rotate', level=8)],
        [dict(type='TranslateX', level=8)],
        [dict(type='TranslateY', level=8)],
        [dict(type='AutoContrast', level=8)],
        [dict(type='Equalize', level=8)],
        [dict(type='Contrast', level=8)],
        [dict(type='Color', level=8)],
        [dict(type='Brightness', level=8)],
        [dict(type='Sharpness', level=8)],
]

train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotationsWithDS',
            with_bbox=True,
            with_mask=True,
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
                'flip', 'flip_direction', 'homography_matrix', 'ds')
        ),
]

eval_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='Resize',
            scale=(399, 224),
            keep_ratio=True,
        ),
        dict(type='LoadAnnotationsWithDS',
            with_bbox=True,
            with_mask=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'ds'),
        ),
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CocoDatasetWithDS',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_phase/annotation_coco.json',
        data_prefix=dict(img='train_phase/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CocoDatasetWithDS',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_phase/annotation_coco.json',
        data_prefix=dict(img='val_phase/'),
        pipeline=eval_pipeline))

test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CocoDatasetWithDS',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test_phase/annotation_coco.json',
        data_prefix=dict(img='test_phase/'),
        pipeline=eval_pipeline))

# metric
val_evaluator = dict(ann_file=os.path.join(data_root, 'val_phase/annotation_coco.json'), format_only=False,)
test_evaluator = dict(ann_file=os.path.join(data_root, 'test_phase/annotation_coco.json'), format_only=False,)
evaluation = dict(metric=['bbox', 'segm'])