import os
import copy

_base_ = [
    '../../configs/datasets/c80_phase/c80_phase_vid_instance_5_load_graphs.py',
    'sv2lstg_faster_rcnn_base.py',
]
orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['evaluator.CocoMetricRGD', 'model.sv2lstg',
    'hooks.custom_hooks', 'runner.custom_loops', 'model.saved_lg_preprocessor'], allow_failed_imports=False)

lg_model = copy.deepcopy(_base_.model)
lg_model.num_classes = len(_base_.metainfo.classes)
lg_model.detector.roi_head.bbox_head.num_classes = len(_base_.metainfo.classes)

# load and modify ds head
ds_head = copy.deepcopy(lg_model.ds_head)
ds_head['type'] = 'STDSHead'
ds_head['gnn_cfg']['num_layers'] = 8
ds_head['causal'] = True
ds_head['num_temp_frames'] = _base_.num_temp_frames
ds_head['use_temporal_model'] = True
ds_head['temporal_arch'] = 'tcn'
ds_head['final_viz_feat_size'] = 256
ds_head['final_sem_feat_size'] = 256

# remove unnecessary parts of lg_model (only need detector and graph head)
del lg_model.data_preprocessor
del lg_model.ds_head
del lg_model.reconstruction_head
del _base_.load_from

model = dict(
    _delete_=True,
    type='SV2LSTG',
    lg_detector=lg_model,
    ds_head=ds_head,
    data_preprocessor=dict(
        type='SavedLGPreprocessor',
    ),
    edge_max_temporal_range=10,
    use_spat_graph=True,
    use_viz_graph=True,
    learn_sim_graph=False,
    pred_per_frame=True,
    per_video=True,
)

# metric
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_phase',
        data_root=_base_.data_root,
        data_prefix=_base_.val_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'val_phase/annotation_ds_coco.json'),
        metric=[],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_phase',
        data_root=_base_.data_root,
        data_prefix=_base_.test_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'test_phase/annotation_ds_coco.json'),
        metric=[],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
        outfile_prefix='./results/c80_phase_preds/test/lg_cvs',
    ),
]

# Running settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30)
val_cfg = dict(type='ValLoopKeyframeEval')
test_cfg = dict(type='TestLoopKeyframeEval')

# Hooks
del _base_.custom_hooks

# visualizer
default_hooks = dict(
    visualization=dict(type='TrackVisualizationHook', draw=False),
    logger=dict(interval=5),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# optimizer
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=0.0003),
    clip_grad=dict(max_norm=10, norm_type=2),
)

# logging
