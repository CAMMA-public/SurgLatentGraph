_base_ = '../lg_ds_faster_rcnn.py'

model = dict(
    reconstruction_head=dict(
        # turn off graph visual and semantic feats in reconstruction head
        use_visual=False,
        use_semantics=False,
    ),
)
