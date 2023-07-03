import os
import copy

_base_ = [
    '../configs/datasets/c80_phase_vid_instance.py',
    'test_lg_ds_faster_rcnn.py',
]
orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['evaluator.CocoMetricRGD', 'model.sv2lstg', 'hooks.custom_hooks'],
        allow_failed_imports=False)

lg_model = copy.deepcopy(_base_.model)
lg_model.num_classes = len(_base_.metainfo.classes)
lg_model.detector.roi_head.bbox_head.num_classes = len(_base_.metainfo.classes)
ds_head = copy.deepcopy(lg_model.ds_head)

# remove unnecessary parts of lg_model (only need detector and graph head)
del lg_model.data_preprocessor
del lg_model.ds_head
del lg_model.reconstruction_head

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
)

# metric
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_phase',
        data_root=_base_.data_root,
        data_prefix=_base_.val_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'val_phase/annotation_coco.json'),
        metric=['bbox'],
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
        ann_file=os.path.join(_base_.data_root, 'test_phase/annotation_coco.json'),
        metric=['bbox'],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
        outfile_prefix='./results/c80_preds/test',
        save_graphs=True,
    ),
]

# Running settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hooks
del _base_.custom_hooks
custom_hooks = [dict(type="FreezeLGDetector")]
#custom_hooks = [dict(type="CopyDetectorBackbone"), dict(type="FreezeDetector")]

# visualizer
default_hooks = dict(
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')
