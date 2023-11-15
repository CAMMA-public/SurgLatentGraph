import copy

# modify base for different detectors
_base_ = [
    '../lg_ds_base.py', 'sam_detector.py',
]
custom_imports = dict(imports=_base_.custom_imports.imports + ['model.modified_detectors.sam_detector'],
        allow_failed_imports=False)

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.num_classes = _base_.num_classes
detector.num_nodes = _base_.num_nodes
detector.cluster_info_path = 'sam_queries/endoscapes/cluster_info.pickle'

# set dp stats, detector pixel stats
dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
dp.mean = [123.675, 116.28, 103.53]
dp.std = [58.395, 57.12, 57.375]
detector.pixel_mean = [0, 0, 0]
detector.pixel_std = [1, 1, 1]

del _base_.model
del detector.data_preprocessor

# extract lg config, set detector
model = copy.deepcopy(_base_.lg_model)
model.data_preprocessor = dp
model.detector = detector
model.reconstruction_img_stats = dict(mean=dp.mean, std=dp.std)
model.ds_head.use_img_feats = True
#model.ds_head.img_feat_size = 256
#model.ds_head.final_sem_feat_size = 0
model.sem_feat_use_class_logits = False

# trainable bb, neck
model.trainable_backbone_cfg = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
            checkpoint='weights/ssl_weights/no_phase/converted_moco.torch'),
)
model.trainable_neck_cfg = dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
)

# roi extractor
model.roi_extractor = dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=1, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
)

del _base_.lg_model

# modify load_from
load_from = _base_.load_from.replace('base', 'sam')

# optim settings
optim_wrapper = dict(
    optimizer=dict(lr=0.00001),
    paramwise_cfg=None,
)

custom_hooks = [dict(type="FreezeDetectorHook")]
