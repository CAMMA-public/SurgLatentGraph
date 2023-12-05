import os

_base_ = ['lg_base_box.py']

# dataset
train_dataloader = dict(
    dataset=dict(
        ann_file='train_seg/annotation_coco.json',
        data_prefix=dict(img='train_seg'),
    )
)
val_dataloader = dict(
    dataset=dict(
        ann_file='val_seg/annotation_coco.json',
        data_prefix=dict(img='val_seg'),
    )
)
test_dataloader = dict(
    dataset=dict(
        ann_file='test_seg/annotation_coco.json',
        data_prefix=dict(img='test_seg'),
    )
)

# metric
val_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=_base_.data_root,
    data_prefix='val_seg',
    ann_file=os.path.join(_base_.data_root, 'val_seg/annotation_coco.json'),
    metric=['bbox', 'segm'],
    additional_metrics=['reconstruction'],
    use_pred_boxes_recon=False,
    num_classes=-1, # ds_num_classes
)

test_evaluator = dict(
    type='CocoMetricRGD',
    prefix='endoscapes',
    data_root=_base_.data_root,
    data_prefix='test_seg',
    ann_file=os.path.join(_base_.data_root, 'test_seg/annotation_coco.json'),
    #data_prefix='test',
    #ann_file=os.path.join(_base_.data_root, 'test/annotation_coco.json'),
    metric=['bbox', 'segm'],
    additional_metrics=['reconstruction'],
    use_pred_boxes_recon=False,
    num_classes=-1, # ds num classes
    outfile_prefix='./results/endoscapes_preds/test/lg',
    classwise=True,
)

default_hooks = dict(
    checkpoint=dict(save_best='endoscapes/segm_mAP'),
)

# training schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,
        by_epoch=True,
        milestones=[32, 54],
        gamma=0.1)
]

#optim_wrapper = dict(
#    optimizer=dict(lr=0.001),
#    paramwise_cfg=dict(
#        custom_keys={
#            'mask_head': dict(lr_mult=10),
#        }
#    ),
#)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=60,
    val_interval=3)
