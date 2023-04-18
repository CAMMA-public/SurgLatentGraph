_base_ = ['dc_mask_rcnn.py']

model = dict(
    reconstruction_head = None,
    reconstruction_loss = None,
)
