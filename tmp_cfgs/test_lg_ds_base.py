import os
import copy

_base_ = ['test_lg_base_box.py']

# recon params
bottleneck_feat_size = 64
layout_noise_dim = 32
recon_input_dim = bottleneck_feat_size + layout_noise_dim + _base_.semantic_feat_size

# save graphs or no
save_graphs = False

# model
lg_model = _base_.lg_model
lg_model.perturb_factor = 0.125
lg_model.use_pred_boxes_recon_loss = True
lg_model.graph_head.compute_gt_eval = save_graphs
lg_model.ds_head = dict(
    type='DSHead',
    num_classes=7,
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
    final_sem_feat_size=256,
    final_viz_feat_size=256,
    use_img_feats=True,
    loss_consensus='none',
    loss=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[1.61803561, 0.18816378, 1, 0.24091337, 1.85450955, 0.98427673, 2.12283346]
    ),
    num_predictor_layers=3,
)
lg_model.reconstruction_head = dict(
    type='ReconstructionHead',
    layout_noise_dim=layout_noise_dim,
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
    box_loss_weight=0.75,
    recon_loss_weight=0.1,
    use_content=True,
    use_style=False,
    use_ssim=False,
    use_l1=True,
    #deep_loss_backbone='resnet50',
    #load_backbone_weights='weights/converted_moco.torch',
)
trainable_backbone_frozen_stages = 1

# optimizer
del _base_.param_scheduler
del _base_.optim_wrapper
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00001),
    clip_grad=dict(max_norm=10, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'semantic_feat_projector': dict(lr_mult=10),
            'reconstruction_head': dict(lr_mult=10),
        }
    ),
)
auto_scale_lr = dict(enable=False)

# hooks
custom_hooks = [dict(type="CopyDetectorBackbone"), dict(type="FreezeDetectorHook")]
default_hooks = dict(
    checkpoint=dict(save_best='c80_phase/ds_f1', rule='greater'),
)

# loading
load_from = 'weights/c80_phase/lg_base_no_recon.pth'
_base_.default_hooks.visualization.update(dict(draw=False))
