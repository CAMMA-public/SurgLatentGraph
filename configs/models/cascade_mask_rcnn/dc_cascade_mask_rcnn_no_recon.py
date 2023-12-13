_base_ = ['dc_cascade_mask_rcnn.py']

model = dict(
    reconstruction_head = None,
    reconstruction_loss = None,
)
