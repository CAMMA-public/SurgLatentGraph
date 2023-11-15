import copy
import os

_base_ = ['../lg_base_seg.py', 'sam_detector.py']
custom_imports = dict(imports=_base_.custom_imports.imports + ['model.modified_detectors.sam_detector', 'hooks.custom_hooks'],
        allow_failed_imports=False)

# define data preprocessor, detector
dp = _base_.model.data_preprocessor
dp.mean = [123.675, 116.28, 103.53]
dp.std = [58.395, 57.12, 57.375]
detector = _base_.model
detector.num_classes = _base_.num_classes
detector.num_nodes = _base_.num_nodes
detector.pixel_mean = [0, 0, 0]
detector.pixel_std = [1, 1, 1]
detector.cluster_info_path = 'sam_queries/endoscapes/cluster_info.pickle'
del _base_.model

# extract lg config, set detector
model = copy.deepcopy(_base_.lg_model)
model.data_preprocessor = dp
model.detector = detector
del _base_.lg_model

# set graph head to use pred dets
model.force_train_graph_head = True
model.graph_head.gt_use_pred_detections = True
model.graph_head.presence_loss_weight = 1
model.graph_head.classifier_loss_weight = 1

# val/test cfgs
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# modify batch sizes
train_dataloader = dict(
    batch_size=8,
)
val_dataloader = dict(
    batch_size=8,
)
test_dataloader = dict(
    #dataset=dict(
    #    ann_file='train/annotation_coco.json',
    #    data_prefix=dict(img='train/'),
    #),
    batch_size=8,
)

visualization = _base_.default_hooks.visualization
visualization.update(dict(draw=False))

custom_hooks = [dict(type="FreezeDetectorHook")]
