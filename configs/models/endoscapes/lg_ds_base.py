import os
import copy

_base_ = ['lg_base_box.py']

# import freeze hook
orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['hooks.custom_hooks', 'visualizer.LatentGraphVisualizer'], allow_failed_imports=False)
# recon params
bottleneck_feat_size = 64
bg_img_dim = 256
recon_input_dim = bottleneck_feat_size + bg_img_dim

# model
lg_model = _base_.lg_model
lg_model.perturb_factor = 0.125
lg_model.ds_head = dict(
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
    input_sem_feat_size=_base_.semantic_feat_size,
    input_viz_feat_size=_base_.viz_feat_size,
    final_sem_feat_size=512,
    final_viz_feat_size=512,
    use_img_feats=True,
    loss_consensus='mode',
    loss=dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        class_weight=[3.19852941, 4.46153846, 2.79518072],
    ),
    loss_weight=1.0,
    num_predictor_layers=3,
    semantic_loss_weight=1.0,
    viz_loss_weight=0.3,
    img_loss_weight=0.3,
)
lg_model.reconstruction_head = dict(
    type='ReconstructionHead',
    bg_img_dim=bg_img_dim,
    num_classes=_base_.num_classes,
    num_nodes=_base_.num_nodes,
    bottleneck_feat_size=bottleneck_feat_size,
    decoder_cfg=dict(
        type='DecoderNetwork',
        dims=(recon_input_dim, 1024, 512, 256, 128, 64),
        spade_blocks=True,
        source_image_dims=bg_img_dim,
        normalization='batch',
        activation='leakyrelu-0.2',
    ),
    aspect_ratio=[2, 3],
    use_seg_recon=True,
)
lg_model.reconstruction_loss=dict(
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
)
trainable_backbone_frozen_stages = 1

lg_model.force_train_graph_head = True

# dataset
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='train/annotation_ds_coco.json',
        filter_cfg=dict(filter_empty_gt=False),
    ),
)
train_eval_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        ann_file='train/annotation_ds_coco.json',
        test_mode=True,
    ),
)
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='val/annotation_ds_coco.json',
    ),
)
test_dataloader = dict(
    batch_size=32,
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
    outfile_prefix='./results/endoscapes_preds/train/lg',
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
    outfile_prefix='./results/endoscapes_preds/val/lg',
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
    outfile_prefix='./results/endoscapes_preds/test/lg',
)

# optimizer
del _base_.param_scheduler
del _base_.optim_wrapper
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00001),
    clip_grad=dict(max_norm=10, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'semantic_feat_projector': dict(lr_mult=10),
        }
    ),
)
auto_scale_lr = dict(enable=False)

# hooks
custom_hooks = [dict(type="CopyDetectorBackbone"), dict(type="FreezeHook")]
default_hooks = dict(
    checkpoint=dict(save_best='endoscapes/ds_average_precision'),
    visualization=dict(draw=False),
)

# loading
load_from = 'weights/endoscapes/lg_base.pth'

# visualization
visualizer = dict(
    type='LatentGraphVisualizer',
    dataset='endoscapes',
    data_prefix='test',
    draw=True,
)
