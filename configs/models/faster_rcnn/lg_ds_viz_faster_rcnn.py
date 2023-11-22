_base_ = 'lg_ds_faster_rcnn.py'

model = dict(
    # remove semantic features
    semantic_feat_size=0,
    ds_head=dict(
        # remove semantic features
        final_sem_feat_size=0,
        # turn off perturbation
        semantic_loss_weight=0,
        viz_loss_weight=0,
        img_loss_weight=0,
    ),
)
