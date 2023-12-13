import copy

_base_ = [
    '../sv2lstg_load_graphs_10_base.py',
    os.path.expandvars('$MMDETECTION/configs/_base_/models/mask-rcnn_r50_fpn.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head.num_classes = _base_.num_classes
detector.roi_head.mask_head.num_classes = _base_.num_classes
detector.test_cfg.rcnn.max_per_img = _base_.num_nodes
del _base_.model
del detector.data_preprocessor

# init model
model = copy.deepcopy(_base_.sv2lstg_model)

# configure lg detector
model.lg_detector.detector = detector

# weight init
model.lg_detector.init_cfg = dict(
    type='Pretrained',
    checkpoint=_base_.load_from.replace('base', 'mask_rcnn'),
)


model.lg_detector.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.lg_detector.roi_extractor.roi_layer.output_size = 1

# remove unnecessary components
del _base_.sv2lstg_model
del _base_.load_from

# set saved graph dir in pipelines
saved_graph_dir = _base_.train_dataloader.dataset.pipeline[1].transforms[0].saved_graph_dir.replace('base', 'mask_rcnn')
_base_.train_dataloader.dataset.pipeline[1].transforms[0].saved_graph_dir = saved_graph_dir
_base_.val_dataloader.dataset.pipeline[1].transforms[0].saved_graph_dir = saved_graph_dir
_base_.test_dataloader.dataset.pipeline[1].transforms[0].saved_graph_dir = saved_graph_dir
