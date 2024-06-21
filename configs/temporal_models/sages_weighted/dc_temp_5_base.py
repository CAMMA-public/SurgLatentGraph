_base_ = [
    'dc_temp_model_base.py',
    '../datasets/sages/sages_weighted_vid_instance.py',
]

train_dataloader = dict(batch_size=16)
val_dataloader = dict(batch_size=16)
test_dataloader = dict(batch_size=16)
