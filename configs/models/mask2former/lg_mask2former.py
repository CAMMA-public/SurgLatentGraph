import copy
import os

_base_=['../lg_base_seg.py',
    os.path.expandvars('$MMDETECTION/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic_no_base.py'),
]
custom_imports = dict(imports=_base_.custom_imports.imports + ['model.modified_detectors.mask2former_with_queries'],
        allow_failed_imports=False)

# extract detector, data preprocessor config from base
num_things_classes = _base_.num_classes
num_stuff_classes = 0
detector = copy.deepcopy(_base_.model)
detector.type = 'Mask2FormerWithQueries'
detector.panoptic_head.type = 'Mask2FormerHeadWithQueries'
detector.panoptic_head.num_things_classes = num_things_classes
detector.panoptic_head.num_stuff_classes = num_stuff_classes
detector.panoptic_head.loss_cls=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=2.0,
    reduction='mean',
    class_weight=[1.0] * _base_.num_classes + [0.1],
)
detector.panoptic_fusion_head.type = 'MaskFormerFusionHeadWithIndices'
detector.panoptic_fusion_head.num_things_classes = num_things_classes
detector.panoptic_fusion_head.num_stuff_classes = num_stuff_classes
detector.test_cfg.max_per_image = _base_.num_nodes

dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
dp.pad_mask = False
dp.batch_augments = []
del _base_.model
del detector.data_preprocessor

# extract lg config, set detector
model = copy.deepcopy(_base_.lg_model)
model.data_preprocessor = dp
model.detector = detector
del _base_.lg_model

# modify load_from
load_from = 'weights/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721_LG.pth'

# optimizer
del _base_.optim_wrapper
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_epochs = 30
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[22],
        gamma=0.1)
]
