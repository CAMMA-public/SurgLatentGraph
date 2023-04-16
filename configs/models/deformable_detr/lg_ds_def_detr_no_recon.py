# modify base for different detectors
_base_ = ['../lg_ds_def_detr.py']
model = dict(
    reconstruction_head=None,
    reconstruction_loss=None,
)

# modify load_from
load_from = 'weights/lg_def_detr_no_recon.pth'
