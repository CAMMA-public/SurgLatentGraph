import copy

_base_ = [
    '../sv2lstg_load_graphs_15_base.py',
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

# weight init
model.lg_detector.init_cfg = dict(
    type='Pretrained',
    checkpoint=_base_.load_from.replace('base', 'faster_rcnn'),
)


model.lg_detector.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.lg_detector.roi_extractor.roi_layer.output_size = 1

# remove unnecessary components
del _base_.sv2lstg_model
del _base_.load_from

# set saved graph dir in pipelines
saved_graph_dir = _base_.train_dataloader.dataset.pipeline[1].transforms[0].saved_graph_dir.replace('base', 'faster_rcnn')
_base_.train_dataloader.dataset.pipeline[1].transforms[0].saved_graph_dir = saved_graph_dir
_base_.val_dataloader.dataset.pipeline[1].transforms[0].saved_graph_dir = saved_graph_dir
_base_.test_dataloader.dataset.pipeline[1].transforms[0].saved_graph_dir = saved_graph_dir
