_base_ = '../lg_ds_mask_rcnn.py'

model = dict(
    # remove img features in ds head
    ds_head=dict(
        use_img_feats=False,
    ),
)
