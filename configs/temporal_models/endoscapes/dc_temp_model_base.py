import copy

_base_ = '../models/deepcvs_base.py'

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
custom_imports = dict(imports=model_imports + ['evaluator.CocoMetricRGD',
    'model.deepcvs_temporal', 'datasets.custom_loading', 'hooks.custom_hooks',
    'runner.custom_loops'], allow_failed_imports=False)

_base_.dc_model.type = 'DeepCVSTemporal'
_base_.dc_model.clip_size = 5
_base_.dc_model.temporal_arch = 'transformer'
_base_.dc_model.data_preprocessor = dict(
    type='TrackDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_mask=False,
    pad_size_divisor=1,
)

# turn off reconstruction
del _base_.dc_model.reconstruction_head
del _base_.dc_model.reconstruction_loss

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

# evaluators
train_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/train/dc_temp',
)
val_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/val/dc_temp',
)
test_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/test/dc_temp',
)
