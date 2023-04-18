_base_ = ['dc_cascade_rcnn.py']

model = dict(
    reconstruction_head = None,
    reconstruction_loss = None,
)
