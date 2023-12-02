_base_ = [
    'sv2lstg_model_base.py',
    '../datasets/cholecT50/cholecT50_vid_instance_load_graphs.py',
]

_base_.sv2lstg_model.data_preprocessor = dict(type='SavedLGPreprocessor')
