import os
import copy

_base_ = ['lg_base_box.py']

# import freeze hook
orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['hooks.custom_hooks'], allow_failed_imports=False)

# feat sizes
ds_input_feat_size = 128 # downproject each node and edge feature to this size for ds pred

# recon params
bottleneck_feat_size = 64
layout_noise_dim = 32
recon_input_dim = bottleneck_feat_size + layout_noise_dim + _base_.semantic_feat_size

# model
lg_model = _base_.lg_model
lg_model.trainable_backbone=True
lg_model.use_pred_boxes_recon_loss=True
lg_model.ds_head=dict(
    type='DSHead',
    num_classes=3,
    gnn_cfg=dict(
        type='GNNHead',
        num_layers=3,
        arch='tripleconv',
        add_self_loops=False,
        use_reverse_edges=False,
        norm='graph',
        skip_connect=True,
    ),
    img_feat_key='bb',
    img_feat_size=2048,
    graph_feat_input_dim=_base_.viz_feat_size+_base_.semantic_feat_size,
    graph_feat_projected_dim=ds_input_feat_size,
    loss_consensus='mode',
    loss='bce',
    loss_weight=1.0,
    num_predictor_layers=3,
    weight=[3.19852941, 4.46153846, 2.79518072],
)
lg_model.reconstruction_head = dict(
    type='ReconstructionHead',
    layout_noise_dim=layout_noise_dim,
    num_classes=_base_.num_classes,
    num_nodes=_base_.num_nodes,
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
)
lg_model.reconstruction_loss=dict(
    type='ReconstructionLoss',
    l1_weight=0.15,
    ssim_weight=0.0,
    deep_loss_weight=0.6,
    perceptual_weight=1.0,
    box_loss_weight=0.5,
    recon_loss_weight=1.0,
    use_content=True,
    use_style=False,
    use_ssim=False,
    use_l1=True,
    #deep_loss_backbone='resnet50',
    #load_backbone_weights='weights/converted_moco.torch',
)

# dataset
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='train/annotation_cvs_coco.json',
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='val/annotation_cvs_coco.json',
    ),
)
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        ann_file='test/annotation_cvs_coco.json',
    ),
)

# metric (in case we need to change dataset)
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.val_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'val/annotation_cvs_coco.json'),
        use_pred_boxes_recon=True,
        metric=['bbox'],
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.test_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'test/annotation_cvs_coco.json'),
        metric=['bbox'],
        #additional_metrics = ['reconstruction'],
        use_pred_boxes_recon=True,
        outfile_prefix='./results/endoscapes_preds/test'
    ),
]

# optimizer
del _base_.param_scheduler
del _base_.optim_wrapper
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00001),
    #clip_grad=dict(max_norm=0.1, norm_type=2),
)
auto_scale_lr = dict(enable=False)

# hooks
custom_hooks = [dict(type="CopyDetectorBackbone"), dict(type="FreezeDetectorHook")]
default_hooks = dict(
    checkpoint=dict(save_best='endoscapes/ds_average_precision'),
)
