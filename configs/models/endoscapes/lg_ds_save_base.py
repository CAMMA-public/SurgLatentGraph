_base_ = ['lg_ds_base.py']

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['visualizer.LatentGraphVisualizer'],
        allow_failed_imports=False)

visualizer = dict(
    type='LatentGraphVisualizer',
    prefix='endoscapes_base',
    save_graphs=True,
    draw=False,
)

default_hooks = dict(
    visualization=dict(
        draw=True,
    ),
)

lg_model = dict(
    graph_head=dict(
        compute_gt_eval=False,
    ),
    reconstruction_head=None,
)

test_dataloader = dict(
    dataset=dict(
        ann_file='annotation_coco_all_frames.json',
        data_prefix=dict(img='all/'),
    )
)

test_evaluator = dict(save_lg=True)
