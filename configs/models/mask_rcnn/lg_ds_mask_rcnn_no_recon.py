# modify base for different detectors
_base_ = ['lg_ds_mask_rcnn.py']

model = dict(
    reconstruction_head=None,
    reconstruction_loss=None,
)

# modify load_from
load_from = 'weights/lg_mask_rcnn_no_recon.pth'
