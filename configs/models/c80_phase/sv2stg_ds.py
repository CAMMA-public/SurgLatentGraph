import os
import copy

_base_ = ['lg_base_box.py']

# import freeze hook
orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['hooks.custom_hooks'], allow_failed_imports=False)

# save graphs or no
save_graphs = False

# lg_model from base
lg_model = _base_.lg_model
lg_model.perturb_factor = 0.125 # box perturbation
lg_model.graph_head.compute_gt_eval = save_graphs

# now define model
model = dict(
    type='SV2STG',
    lg_detector=lg_model,
    ds_head=dict(
        type='DSHead',
        num_classes=7,
        gnn_cfg=dict(
            type='GNNHead',
            num_layers=3,
            arch='tripleconv',
            add_self_loops=False,
            use_reverse_edges=False,
            norm='graph',
            skip_connect=True,
        ),
        img_feat_key='bb',
        img_feat_size=2048,
        input_sem_feat_size=_base_.semantic_feat_size,
        input_viz_feat_size=_base_.viz_feat_size,
        final_sem_feat_size=256,
        final_viz_feat_size=256,
        use_img_feats=True,
        loss_consensus='none',
        loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.61803561, 0.18816378, 1, 0.24091337, 1.85450955, 0.98427673, 2.12283346],
        ),
        num_predictor_layers=3,
    )
)
