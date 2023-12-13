_base_ = [
    'sv2lstg_model_base.py',
    '../datasets/cholecT50/cholecT50_vid_instance_load_graphs.py',
]

num_temp_frames = 10
_base_.train_dataloader.dataset.pipeline[0].num_ref_imgs = num_temp_frames - 1
_base_.train_dataloader.dataset.pipeline[0].frame_range = [1 - num_temp_frames, 0]
_base_.val_dataloader.dataset.pipeline[0].num_ref_imgs = num_temp_frames - 1
_base_.val_dataloader.dataset.pipeline[0].frame_range = [1 - num_temp_frames, 0]
_base_.test_dataloader.dataset.pipeline[0].num_ref_imgs = num_temp_frames - 1
_base_.test_dataloader.dataset.pipeline[0].frame_range = [1 - num_temp_frames, 0]

_base_.sv2lstg_model.clip_size = num_temp_frames
_base_.sv2lstg_model.data_preprocessor = dict(type='SavedLGPreprocessor')

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=100,
    ),
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=100,
)
