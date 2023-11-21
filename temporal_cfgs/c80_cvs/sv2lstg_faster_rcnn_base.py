import copy

_base_ = '../../configs/models/faster_rcnn/lg_ds_faster_rcnn.py'

# delete dataset related fields
del _base_.val_evaluator
del _base_.test_evaluator
del _base_.train_dataloader
del _base_.val_dataloader
del _base_.test_dataloader
del _base_.train_pipeline
del _base_.test_pipeline
del _base_.eval_pipeline
del _base_.metainfo
del _base_.val_data_prefix
del _base_.test_data_prefix
del _base_.dataset_type
del _base_.data_root
del _base_.backend_args

model_imports = copy.deepcopy(_base_.custom_imports)
del _base_.custom_imports
del _base_.rand_aug_surg
