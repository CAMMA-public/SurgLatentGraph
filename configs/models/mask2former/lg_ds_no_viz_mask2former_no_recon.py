# modify base for different detectors
_base_ = ['lg_ds_no_viz_mask2former.py']

model = dict(
    reconstruction_head=None,
    reconstruction_loss=None,
)
