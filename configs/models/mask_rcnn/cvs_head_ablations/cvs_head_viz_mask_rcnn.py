_base_ = '../lg_ds_mask_rcnn.py'

model = dict(
    # remove semantic features in ds head
    ds_head=dict(
        final_sem_feat_size=0,
    ),
)
