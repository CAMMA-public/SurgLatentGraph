import copy

_base_ = [
    '../sv2lstg_5_base.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/faster-rcnn_r50_fpn.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head.num_classes = _base_.num_classes
detector.test_cfg.rcnn.max_per_img = _base_.num_nodes
del _base_.model
del detector.data_preprocessor

# init model
model = copy.deepcopy(_base_.sv2lstg_model)

# configure lg detector
model.lg_detector.detector = detector
model.lg_detector.sem_feat_use_masks = False
model.sem_feat_use_masks = False

# trainable bb, neck
model.lg_detector.trainable_backbone_cfg=copy.deepcopy(detector.backbone)
model.lg_detector.trainable_backbone_cfg.frozen_stages = _base_.trainable_backbone_frozen_stages
if 'neck' in detector:
    model.lg_detector.trainable_neck_cfg=copy.deepcopy(detector.neck)

# weight init
model.lg_detector.init_cfg = dict(
    type='Pretrained',
    checkpoint=_base_.load_from.replace('base', 'faster_rcnn'),
)


model.lg_detector.reconstruction_img_stats = dict(
    mean=model.data_preprocessor.mean,
    std=model.data_preprocessor.std,
)
model.lg_detector.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.lg_detector.roi_extractor.roi_layer.output_size = 1

# remove unnecessary components
del _base_.sv2lstg_model
del _base_.load_from

del _base_.optim_wrapper.paramwise_cfg.custom_keys['semantic_feat_projector']
