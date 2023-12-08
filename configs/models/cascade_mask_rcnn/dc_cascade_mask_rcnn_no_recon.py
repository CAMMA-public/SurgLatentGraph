_base_ = ['dc_cascade_rcnn_no_recon.py']

model = dict(
    reconstruction_head = None,
    reconstruction_loss = None,
)
