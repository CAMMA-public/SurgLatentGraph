import os

_base_ = ['lg_base_box.py']

# metric
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_cvs',
        data_root=_base_.data_root,
        data_prefix=_base_.val_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'val_cvs/annotation_coco.json'),
        metric=['bbox', 'segm'],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
        num_classes=-1, # ds_num_classes
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_cvs',
        data_root=_base_.data_root,
        data_prefix=_base_.test_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'test_cvs/annotation_coco.json'),
        metric=['bbox', 'segm'],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
        num_classes=-1, # ds num classes
        outfile_prefix='./results/c80_cvs_preds/test',
        save_graphs=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(save_best='c80_cvs/segm_mAP'),
)
