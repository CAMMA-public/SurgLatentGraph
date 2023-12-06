_base_ = [
    'sv2lstg_model_base.py',
    '../datasets/endoscapes/endoscapes_vid_instance.py',
]

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=175,
    ),
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=3500,
    val_interval=175,
)
