_base_ = [
    'sv2lstg_model_base.py',
    '../datasets/endoscapes/endoscapes_vid_instance.py',
]

num_temp_frames = 15
_base_.train_dataloader.dataset.pipeline[0].num_ref_imgs = num_temp_frames - 1
_base_.train_dataloader.dataset.pipeline[0].frame_range = [1 - num_temp_frames, 0]
_base_.val_dataloader.dataset.pipeline[0].num_ref_imgs = num_temp_frames - 1
_base_.val_dataloader.dataset.pipeline[0].frame_range = [1 - num_temp_frames, 0]
_base_.test_dataloader.dataset.pipeline[0].num_ref_imgs = num_temp_frames - 1
_base_.test_dataloader.dataset.pipeline[0].frame_range = [1 - num_temp_frames, 0]

_base_.sv2lstg_model.clip_size = num_temp_frames

train_dataloader = dict(batch_size=5)
val_dataloader = dict(batch_size=5)
test_dataloader = dict(batch_size=5)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=250,
    ),
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=5000,
    val_interval=250,
)
