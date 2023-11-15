import os
import copy

# modify base for different detectors
_base_ = [
    'lg_ds_mask2former.py',
]

# trainable bb, neck
_base_.model.trainable_detector_cfg = None

# remove visual features
_base_.model.ds_head.final_viz_feat_size = 0
_base_.model.ds_head.use_img_feats = False

_base_.model.sem_feat_use_masks = False

# train graph head since we are changing semantic feat projector arch, use pred detections rather than gt
#model.force_train_graph_head = True
#model.graph_head.gt_use_pred_detections = True

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=1)

train_dataloader = dict(batch_size=32)
val_dataloader = dict(batch_size=32)
test_dataloader = dict(batch_size=32)
