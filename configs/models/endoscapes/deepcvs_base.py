import os

# dataset, optimizer, and runtime cfgs
_base_ = [
    '../datasets/endoscapes/endoscapes_instance.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/schedules/schedule_1x.py'),
    os.path.expandvars('$MMDETECTION/configs/_base_/default_runtime.py')
]

data_root = _base_.data_root
val_data_prefix = _base_.val_dataloader.dataset.data_prefix.img
test_data_prefix = _base_.test_dataloader.dataset.data_prefix.img

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['model.deepcvs', 'evaluator.CocoMetricRGD',
    'hooks.custom_hooks', 'model.predictor_heads.reconstruction', 'model.predictor_heads.modules.loss'],
    allow_failed_imports=False)

# additional params
num_nodes = 16
layout_noise_dim = 256
bottleneck_feat_size = 1024
recon_input_dim = bottleneck_feat_size + layout_noise_dim

dc_model = dict(
    type='DeepCVS',
    num_classes=3,
    detector_num_classes=len(_base_.metainfo.classes),
    num_nodes=num_nodes,
    decoder_backbone=dict(
        type='ResNet',
        in_channels=4+len(_base_.metainfo.classes), # 3 channels + detector_num_classes + 1 (bg)
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet18'
        ),
    ),
    loss=dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        class_weight=[3.19852941, 4.46153846, 2.79518072],
    ),
    use_pred_boxes_recon_loss=True,
    reconstruction_head=dict(
        type='ReconstructionHead',
        bg_img_dim=layout_noise_dim,
        num_classes=len(_base_.metainfo.classes),
        num_nodes=num_nodes,
        bottleneck_feat_size=bottleneck_feat_size,
        decoder_cfg=dict(
            type='DecoderNetwork',
            dims=(recon_input_dim, 1024, 512, 256, 128, 64),
            spade_blocks=True,
            source_image_dims=layout_noise_dim,
            normalization='batch',
            activation='leakyrelu-0.2',
        ),
        aspect_ratio=[2, 3],
        use_seg_recon=True,
        use_pred_boxes_whiteout=True,
        img_feat_size=512,
    ),
    reconstruction_loss=dict(
        type='ReconstructionLoss',
        l1_weight=0.15,
        ssim_weight=0.0,
        deep_loss_weight=0.6,
        perceptual_weight=1.0,
        box_loss_weight=0.75,
        recon_loss_weight=1.0,
        use_content=True,
        use_style=False,
        use_ssim=False,
        use_l1=True,
        #deep_loss_backbone='resnet50',
        #load_backbone_weights='weights/converted_moco.torch',
    ),
)

# dataset
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        ann_file='train/annotation_ds_coco.json',
        filter_cfg=dict(filter_empty_gt=False),
    ),
)
train_eval_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        ann_file='train/annotation_ds_coco.json',
    ),
    drop_last=False,
)
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        ann_file='val/annotation_ds_coco.json',
    ),
)
test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        ann_file='test/annotation_ds_coco.json',
    ),
)

# evaluators
train_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=_base_.data_root,
    data_prefix=_base_.train_eval_dataloader.dataset.data_prefix.img,
    ann_file=os.path.join(_base_.data_root, 'train/annotation_ds_coco.json'),
    use_pred_boxes_recon=True,
    metric=[],
    num_classes=3,
    outfile_prefix='./results/endoscapes_preds/train/deepcvs',
)
val_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=_base_.data_root,
    data_prefix=_base_.val_dataloader.dataset.data_prefix.img,
    ann_file=os.path.join(_base_.data_root, 'val/annotation_ds_coco.json'),
    use_pred_boxes_recon=True,
    metric=[],
    num_classes=3,
    outfile_prefix='./results/endoscapes_preds/val/deepcvs',
)

test_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=_base_.data_root,
    data_prefix=_base_.test_dataloader.dataset.data_prefix.img,
    ann_file=os.path.join(_base_.data_root, 'test/annotation_ds_coco.json'),
    metric=[],
    num_classes=3,
    #additional_metrics = ['reconstruction'],
    use_pred_boxes_recon=True,
    outfile_prefix='./results/endoscapes_preds/test/deepcvs',
)

# optimizer
del _base_.param_scheduler
del _base_.optim_wrapper
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00001),
)
auto_scale_lr = dict(enable=False)

# Running settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# hooks
custom_hooks = [dict(type="FreezeHook")]
default_hooks = dict(
    checkpoint=dict(save_best='endoscapes/ds_average_precision'),
)

# loading
load_from = 'weights/endoscapes/lg_base.pth'
