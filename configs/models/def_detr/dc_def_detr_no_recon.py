_base_ = ['dc_def_detr_no_recon.py']

model = dict(
    reconstruction_head = None,
    reconstruction_loss = None,
)
