import os
import copy

_base_ = ['lg_base_seg.py']

# import freeze hook
orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['hooks.custom_hooks'], allow_failed_imports=False)

# feat sizes
ds_input_feat_size = 128 # downproject each node and edge feature to this size for ds pred

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
#lg_model.reconstruction_head=None
lg_model.reconstruction_head.use_seg_recon = False
lg_model.reconstruction_head.use_pred_boxes_whiteout = True
lg_model.reconstruction_loss.box_loss_weight = 0.5

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
        metric=['bbox', 'segm'],
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.test_dataloader.dataset.data_prefix.img,
        ann_file=os.path.join(_base_.data_root, 'test/annotation_cvs_coco.json'),
        metric=['bbox', 'segm'],
        #additional_metrics = ['reconstruction'],
        use_pred_boxes_recon=True,
        outfile_prefix='./work_dirs/coco_instance/test'
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
