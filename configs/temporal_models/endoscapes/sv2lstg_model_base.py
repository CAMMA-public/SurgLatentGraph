import copy

_base_ = '../models/lg_ds_base.py'

# delete dataset related fields
del _base_.train_dataloader
del _base_.train_eval_dataloader
del _base_.val_dataloader
del _base_.test_dataloader
del _base_.train_pipeline
del _base_.test_pipeline
del _base_.eval_pipeline
del _base_.metainfo
del _base_.val_data_prefix
del _base_.test_data_prefix
del _base_.dataset_type
del _base_.data_root
del _base_.rand_aug_surg
del _base_.backend_args

model_imports = copy.deepcopy(_base_.custom_imports.imports)
del _base_.custom_imports
custom_imports = dict(imports=model_imports + ['evaluator.CocoMetricRGD', 'model.sv2lstg',
        'datasets.custom_loading', 'hooks.custom_hooks', 'runner.custom_loops',
        'model.saved_lg_preprocessor'], allow_failed_imports=False)

lg_model = _base_.lg_model
lg_model.force_encode_semantics = True
lg_model.force_train_graph_head = True
lg_model.perturb_factor = 0

# define additional params in ds head
lg_model.ds_head.type = 'STDSHead'
lg_model.ds_head.num_temp_frames = 5
lg_model.ds_head.gnn_cfg.num_layers = 5
lg_model.ds_head.use_node_positional_embedding = True
lg_model.ds_head.use_temporal_model = True
lg_model.ds_head.temporal_arch = 'transformer'
lg_model.ds_head.pred_per_frame = True
lg_model.ds_head.edited_graph_loss_weight = 1
lg_model.ds_head.loss.reduction = 'none'

# remove unnecessary parts of lg_model (only need detector and graph head)
st_ds_head = copy.deepcopy(lg_model['ds_head'])
del lg_model.ds_head
del lg_model.reconstruction_head

# define model
sv2lstg_model = dict(
    type='SV2LSTG',
    clip_size=5,
    lg_detector=lg_model,
    ds_head=st_ds_head,
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
        pad_size_divisor=1,
    ),
    use_spat_graph=True,
    use_viz_graph=True,
    viz_feat_size=512,
    semantic_feat_size=512,
)

# visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    _delete_=True,
    type='TrackLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Running settings (modify train_cfg here)
train_cfg = dict(
    max_epochs=10,
)

val_cfg = dict(type='ValLoopKeyframeEval')
test_cfg = dict(type='TestLoopKeyframeEval')

# Hooks
del _base_.custom_hooks
default_hooks = dict(visualization=dict(type='TrackVisualizationHook', draw=False))
custom_hooks = [
    dict(type="FreezeLGDetector", finetune_backbone=True),
    dict(type="CopyDetectorBackbone", temporal=True)
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=0.00001),
    clip_grad=dict(max_norm=10, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'semantic_feat_projector': dict(lr_mult=10),
        }
    )
)

# evaluators
train_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/train/sv2lstg',
)
val_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/val/sv2lstg',
)
test_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/test/sv2lstg',
)
