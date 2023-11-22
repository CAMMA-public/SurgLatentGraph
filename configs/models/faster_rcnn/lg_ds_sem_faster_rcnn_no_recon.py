# modify base for different detectors
_base_ = ['lg_ds_sem_faster_rcnn.py']

model = dict(
    reconstruction_head=None,
    reconstruction_loss=None,
)
