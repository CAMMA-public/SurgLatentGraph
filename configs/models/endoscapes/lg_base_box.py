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
custom_imports = dict(imports=orig_imports + ['model.lg', 'evaluator.CocoMetricRGD'], allow_failed_imports=False)

# feat sizes
viz_feat_size = 256
semantic_feat_size = 512

# num nodes in graph
num_nodes = 16
num_classes = len(_base_.metainfo.classes)

lg_model=dict(
    type='LGDetector',
    num_classes=num_classes,
    viz_feat_size = viz_feat_size,
    semantic_feat_size=semantic_feat_size,
    graph_head=dict(
        type='GraphHead',
        edges_per_node=4,
        gnn_cfg=dict(
            type='GNNHead',
            num_layers=3,
            arch='tripleconv',
            add_self_loops=False,
            use_reverse_edges=False,
            norm='graph',
            skip_connect=True,
        ),
        presence_loss_cfg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
        ),
        presence_loss_weight=0.5,
        classifier_loss_cfg=dict(
            type='CrossEntropyLoss',
        ),
        classifier_loss_weight=0.5,
        num_edge_classes=3,
    ),
)

# metric
val_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=data_root,
    data_prefix=val_data_prefix,
    ann_file=os.path.join(data_root, 'val/annotation_coco.json'),
    metric=['bbox'],
    use_pred_boxes_recon=False,
    num_classes=-1, # ds_num_classes
)

test_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=data_root,
    data_prefix=test_data_prefix,
    ann_file=os.path.join(data_root, 'test/annotation_coco.json'),
    metric=['bbox'],
    use_pred_boxes_recon=False,
    num_classes=-1, # ds num classes
    outfile_prefix='./results/endoscapes_preds/test/lg',
    classwise=True,
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[8, 16],
        gamma=0.1)
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
    checkpoint=dict(save_best='endoscapes/bbox_mAP'),
)

# visualizer
visualization = _base_.default_hooks.visualization
visualization.update(dict(draw=False, show=False, score_thr=0.2))
