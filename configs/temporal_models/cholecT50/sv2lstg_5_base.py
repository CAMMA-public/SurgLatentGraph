_base_ = [
    'sv2lstg_model_base.py',
    '../datasets/cholecT50/cholecT50_vid_instance.py',
]

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=640,
    ),
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=6400,
    val_interval=640,
)
