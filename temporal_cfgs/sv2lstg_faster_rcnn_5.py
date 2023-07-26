import os
import copy

_base_ = [
    '../configs/datasets/endoscapes_vid_instance_5.py',
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
ds_head['gnn_cfg']['num_layers'] = 5
ds_head['num_temp_frames'] = _base_.num_temp_frames
#ds_head['loss']['class_weight'] = [3.42870491, 4.77537741, 2.97358185]
ds_head['use_node_positional_embedding'] = True
ds_head['use_temporal_model'] = True
ds_head['temporal_arch'] = 'gru'

# remove unnecessary parts of lg_model (only need detector and graph head)
del lg_model.data_preprocessor
del lg_model.ds_head
del lg_model.reconstruction_head

# set init cfg for lg_model
lg_model.init_cfg = dict(
    type='Pretrained',
    checkpoint='weights/lg_ds_faster_rcnn.pth',
    #checkpoint=_base_.load_from,
)
del _base_.load_from


model = dict(
    _delete_=True,
    type='SV2LSTG',
    lg_detector=lg_model,
    ds_head=ds_head,
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=1,
    ),
    use_spat_graph=True,
    use_viz_graph=True,
    pred_per_frame=True,
)

# metric
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.val_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'val/annotation_ds_coco.json'),
        metric=[],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='endoscapes',
        data_root=_base_.data_root,
        data_prefix=_base_.test_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'test/annotation_ds_coco.json'),
        metric=[],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
        outfile_prefix='./results/endoscapes_preds/test/lg_cvs',
    ),
]

# Running settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1)
val_cfg = dict(type='ValLoopKeyframeEval')
test_cfg = dict(type='TestLoopKeyframeEval')

# Hooks
del _base_.custom_hooks
custom_hooks = [dict(type="FreezeLGDetector", finetune_backbone=True)]#, dict(type="CopyDetectorBackbone", temporal=True)]

# visualizer
default_hooks = dict(
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# optimizer
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=0.00001),
    clip_grad=dict(max_norm=10, norm_type=2),
)
