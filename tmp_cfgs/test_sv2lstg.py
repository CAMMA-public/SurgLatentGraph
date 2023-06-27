import os

_base_ = [
    'configs/datasets/c80_phase_vid_instance.py',
    'configs/models/faster_rcnn/lg_ds_faster_rcnn.py',
]
orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['evaluator.CocoMetricRGD'],
        allow_failed_imports=False)

lg_model = copy.deepcopy(_base_.model)
ds_head = copy.deepcopy(lg_model.ds_head)

model = dict(
    _delete_=True,
    type='SV2LSTG',
    lg_detector=lg_model,
    ds_head=ds_head,
)

# metric
val_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_phase',
        data_root=_base_.data_root,
        data_prefix=_base_.val_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'val_phase/annotation_coco.json'),
        metric=['bbox'],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
    )
]

test_evaluator = [
    dict(
        type='CocoMetricRGD',
        prefix='c80_phase',
        data_root=_base_.data_root,
        data_prefix=_base_.test_data_prefix,
        ann_file=os.path.join(_base_.data_root, 'test_phase/annotation_coco.json'),
        metric=['bbox'],
        additional_metrics=['reconstruction'],
        use_pred_boxes_recon=False,
        outfile_prefix='./results/c80_preds/test',
        save_graphs=True,
    ),
]

# Running settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
