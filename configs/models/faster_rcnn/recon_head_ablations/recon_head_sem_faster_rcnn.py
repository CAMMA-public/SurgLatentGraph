_base_ = '../lg_ds_faster_rcnn.py'

model = dict(
    reconstruction_head=dict(
        # turn off graph visual and img feats in reconstruction head
        use_img=False,
        use_visual=False,
    ),
)
