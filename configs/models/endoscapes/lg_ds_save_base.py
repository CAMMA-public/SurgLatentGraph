_base_ = ['lg_ds_base.py']

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['visualizer.LatentGraphVisualizer'],
        allow_failed_imports=False)

visualizer = dict(
    type='LatentGraphVisualizer',
    save_graphs=True,
)

default_hooks = dict(
    visualization=dict(
        draw=True,
    ),
)
