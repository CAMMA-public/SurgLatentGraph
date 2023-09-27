import os

_base_ = ['lg_base_box.py']

# metric
val_evaluator = dict(
    type='CocoMetricRGD',
    prefix='italy',
    data_root=_base_.data_root,
    data_prefix=_base_.val_data_prefix,
    ann_file=os.path.join(_base_.data_root, 'val/annotation_coco.json'),
    metric=['bbox', 'segm'],
    additional_metrics=['reconstruction'],
    use_pred_boxes_recon=False,
)

test_evaluator = dict(
    type='CocoMetricRGD',
    prefix='italy',
    data_root=_base_.data_root,
    data_prefix=_base_.test_data_prefix,
    ann_file=os.path.join(_base_.data_root, 'test/annotation_coco.json'),
    metric=['bbox', 'segm'],
    additional_metrics=['reconstruction'],
    use_pred_boxes_recon=False,
    outfile_prefix='./results/italy_preds/test/lg_cvs',
    classwise = True,
)

default_hooks = dict(
    checkpoint=dict(save_best='italy/segm_mAP'),
)
