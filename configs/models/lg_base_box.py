import os

# dataset, optimizer, and runtime cfgs
_base_ = [
    '../datasets/endoscapes_instance.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/schedules/schedule_1x.py'),
    os.path.expandvars('$MMDETECTION/configs/_base_/default_runtime.py')
]

data_root = _base_.data_root
val_data_prefix = _base_.val_dataloader.dataset.data_prefix.img
test_data_prefix = _base_.test_dataloader.dataset.data_prefix.img

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['model.lg_detector', 'evaluator.CocoMetricRGD'], allow_failed_imports=False)

# feat sizes
viz_feat_size = 256
semantic_feat_size = 64

# recon params
bottleneck_feat_size = 64
layout_noise_dim = 32
recon_input_dim = bottleneck_feat_size + layout_noise_dim + semantic_feat_size

# num nodes in graph
num_nodes = 16
num_classes = len(_base_.metainfo.classes)

lg_model=dict(
    type='LGDetector',
    trainable_backbone=False,
    num_classes=num_classes,
    semantic_feat_size=semantic_feat_size,
    #reconstruction_head=dict(
    #    type='ReconstructionHead',
    #    layout_noise_dim=layout_noise_dim,
    #    num_classes=num_classes,
    #    num_nodes=num_nodes,
    #    bottleneck_feat_size=bottleneck_feat_size,
    #    decoder_cfg=dict(
    #        type='DecoderNetwork',
    #        dims=(recon_input_dim, 1024, 512, 256, 128, 64),
    #        spade_blocks=True,
    #        source_image_dims=layout_noise_dim,
    #        normalization='batch',
    #        activation='leakyrelu-0.2',
    #    ),
    #    aspect_ratio=[2, 3],
    #    use_seg_recon=True,
    #),
    #reconstruction_loss=dict(
    #    type='ReconstructionLoss',
    #    l1_weight=0.15,
    #    ssim_weight=0.0,
    #    deep_loss_weight=0.6,
    #    perceptual_weight=1.0,
    #    box_loss_weight=0.75,
    #    recon_loss_weight=1.0,
    #    use_content=True,
    #    use_style=False,
    #    use_ssim=False,
    #    use_l1=True,
    #    #deep_loss_backbone='resnet50',
    #    #load_backbone_weights='weights/converted_moco.torch',
    #),
    graph_head=dict(
        type='GraphHead',
        edges_per_node=4,
        viz_feat_size=viz_feat_size,
        semantic_feat_size=semantic_feat_size,
        gnn_cfg=dict(
            type='GNNHead',
            num_layers=3,
            arch='tripleconv',
            add_self_loops=False,
            use_reverse_edges=False,
            norm='graph',
            skip_connect=True,
        ),
        num_edge_classes=3,
    ),
)

# metric
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=data_root,
        data_prefix=val_data_prefix,
        ann_file=os.path.join(data_root, 'val/annotation_coco.json'),
        metric=['bbox'],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=data_root,
        data_prefix=test_data_prefix,
        ann_file=os.path.join(data_root, 'test/annotation_coco.json'),
        metric=['bbox'],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
        outfile_prefix='./work_dirs/coco_instance/test'
    ),
]

# Running settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# hooks
log_config = dict( # config to register logger hook
    interval=50, # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
        dict(type='MMDetWandbHook', by_epoch=False, init_kwargs=
            {
                'entity': "adit98",
                'project': "lg-surg",
                'dir': 'work_dirs',
            }),
    ]
)

default_hooks = dict(
    checkpoint=dict(save_best='bbox_mAP'),
)
