import copy
import os
import time
import argparse

_base_=['../lg_base_box.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/faster-rcnn_r50_fpn.py'),
]

# Make sure custom hooks are imported
custom_imports = dict(
    imports=['datasets.custom_loading', 'model.lg', 'evaluator.CocoMetricRGD', 'hooks.custom_hooks', 'model.corruption_preprocessor'],
    allow_failed_imports=False
)

# Try to get corruption type from command line arguments
# def get_corruption_arg():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--corruption', type=str, default='none', help='Corruption type to apply')
    
#     args, _ = parser.parse_known_args()
#     return args.corruption

# corruption_type = get_corruption_arg()
corruption_type = 'none'  # Default to 'none' if not specified

# Set unique output directory based on timestamp
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
output_dir = f'results/endoscapes_lg_faster_rcnn_{timestamp}'
ckpt_dir = f'{output_dir}/checkpoints'

# Create the checkpoint directory if it doesn't exist
os.makedirs(ckpt_dir, exist_ok=True)

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
model.data_preprocessor = corruption_dp
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
model.detector = detector
model.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.roi_extractor.roi_layer.output_size = 1
del _base_.lg_model

# modify load_from
load_from = 'weights/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c_LG.pth'

# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
)
auto_scale_lr = dict(enable=True)

# Configure custom directory for outputs
work_dir = output_dir

# Configure checkpoint saving
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
        out_dir=ckpt_dir
    ),
    logger=dict(
        type='LoggerHook',
        interval=50,
        out_dir=output_dir
    )
)

# Configure test evaluation output path
test_evaluator = dict(
    outfile_prefix=f'{output_dir}/test_results'
)
