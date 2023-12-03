import os

_base_ = ['simple_classifier.py']

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['model.predictor_heads.reconstruction',
    'model.predictor_heads.modules.loss'], allow_failed_imports=False)

model = dict(
    img_decoder=dict(
        type='DecoderNetwork',
        dims=(1024, 512, 256, 128),
        spade_blocks=True,
        normalization='batch',
        activation='leakyrelu-0.2',
    ),
    aspect_ratio=[4, 4],
    reconstruction_loss=dict(
        type='ReconstructionLoss',
        l1_weight=0.15,
        ssim_weight=0.0,
        deep_loss_weight=0.6,
        perceptual_weight=1.0,
        box_loss_weight=0.0,
        recon_loss_weight=1.0,
        use_content=True,
        use_style=False,
        use_ssim=False,
        use_l1=True,
        #deep_loss_backbone='resnet50',
        #load_backbone_weights='weights/converted_moco.torch',
    ),
    reconstruction_img_stats=dict(
        mean=_base_.model.data_preprocessor.mean,
        std=_base_.model.data_preprocessor.std,
    ),
)

# evaluators
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.val_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'val/annotation_ds_coco.json'),
        use_pred_boxes_recon=True,
        metric=[],
        num_classes=3,
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.test_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'test/annotation_ds_coco.json'),
        metric=[],
        num_classes=3,
        #additional_metrics = ['reconstruction'],
        use_pred_boxes_recon=True,
        outfile_prefix='./results/endoscapes_preds/test/r50'
    ),
]
