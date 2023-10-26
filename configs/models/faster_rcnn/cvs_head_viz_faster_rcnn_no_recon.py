# modify base for different detectors
_base_ = ['cvs_head_viz_faster_rcnn']

model = dict(
    reconstruction_head=None,
    reconstruction_loss=None,
)

