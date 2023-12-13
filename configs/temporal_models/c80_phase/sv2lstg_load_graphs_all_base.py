_base_ = [
    'sv2lstg_model_base.py',
    '../datasets/c80_phase/c80_phase_vid_instance_load_all.py',
]

_base_.sv2lstg_model.data_preprocessor = dict(type='SavedLGPreprocessor')
_base_.sv2lstg_model.ds_head.gnn_cfg.num_layers = 8
_base_.sv2lstg_model.ds_head.temporal_arch = 'tcn'
_base_.sv2lstg_model.ds_head.causal = True
_base_.sv2lstg_model.per_video = True
_base_.sv2lstg_model.edge_max_temporal_range = 10

# Hooks
del _base_.custom_hooks
custom_hooks = [dict(type='ClearGPUMem')]

optim_wrapper = dict(
    _delete=True,
    optimizer=dict(lr=0.001),
    clip_grad=dict(max_norm=10, norm_type=2),
)

train_cfg = dict(
    max_epochs=10,
)
