_base_ = [
    'sv2lstg_model_base.py',
    '../datasets/cholecT50/cholecT50_vid_instance_load_graphs.py',
]

_base_.sv2lstg_model.data_preprocessor = dict(type='SavedLGPreprocessor')

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=200,
    ),
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=4000,
    val_interval=200,
)
