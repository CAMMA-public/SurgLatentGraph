# modify base for different detectors
_base_ = ['cvs_head_sem_img_mask_rcnn.py']

model = dict(
    reconstruction_head=None,
    reconstruction_loss=None,
)

