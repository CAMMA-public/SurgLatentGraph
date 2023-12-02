_base_ = ['lg_ds_base.py']

visualizer = dict(
    type='LatentGraphVisualizer',
    dataset='endoscapes',
    save_graphs=True,
    draw=False,
)

default_hooks = dict(
    visualization=dict(
        draw=True,
    ),
)

lg_model = dict(
    reconstruction_head=None,
    ds_head=None,
    force_encode_semantics=True,
)

test_dataloader = dict(
    dataset=dict(
        ann_file='all/annotation_coco.json',
        data_prefix=dict(img='all/'),
    )
)

test_evaluator = dict(save_lg=True)
