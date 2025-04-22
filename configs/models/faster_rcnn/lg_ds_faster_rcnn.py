import os
import copy
import argparse

# modify base for different detectors
_base_ = [
    '../lg_ds_base.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/faster-rcnn_r50_fpn.py'),
]

# Make sure custom modules are imported
custom_imports = dict(
    imports=['datasets.custom_loading', 'model.lg', 'evaluator.CocoMetricRGD', 'hooks.custom_hooks', 'model.corruption_preprocessor'],
    allow_failed_imports=False
)

# Try to get corruption type from command line arguments
def get_corruption_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corruption', type=str, default='none', help='Corruption type to apply')
    
    args, _ = parser.parse_known_args()
    return args.corruption

corruption_type = get_corruption_arg()

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head.num_classes = _base_.num_classes
detector.test_cfg.rcnn.max_per_img = _base_.num_nodes
dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
del _base_.model
del detector.data_preprocessor

# Wrap the data preprocessor with our corruption preprocessor
corruption_dp = dict(
    type='CorruptionDataPreprocessor',
    base_preprocessor=dp,
    corruption_type=corruption_type,
)

# extract lg config, set detector
model = copy.deepcopy(_base_.lg_model)
model.data_preprocessor = corruption_dp  # Use corruption preprocessor
model.detector = detector
model.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.roi_extractor.roi_layer.output_size = 1
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
model.sem_feat_use_masks = False

# trainable bb, neck
model.trainable_backbone_cfg=copy.deepcopy(detector.backbone)
model.trainable_backbone_cfg.frozen_stages=_base_.trainable_backbone_frozen_stages
if 'neck' in detector:
    model.trainable_neck_cfg=copy.deepcopy(detector.neck)

del _base_.lg_model

# modify load_from
load_from = _base_.load_from.replace('base', 'faster_rcnn')
