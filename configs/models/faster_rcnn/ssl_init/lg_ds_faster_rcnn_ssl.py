import copy

# modify base for different detectors
_base_ = ['../lg_ds_faster_rcnn.py']

# modify model

trainable_backbone_init = 'weights/ssl_weights/no_phase/converted_moco_lap.torch'
model = dict(
    trainable_backbone_cfg=dict(
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=trainable_backbone_init,
        ),
    ),
    ds_head=dict(final_viz_feat_size=0),
)

# neck
#_base_.model.trainable_neck_cfg = dict(
#    type='ChannelMapper',
#    in_channels=[256, 512, 1024, 2048],
#    out_channels=256,
#)

custom_hooks = [dict(type="FreezeDetectorHook", freeze_graph_head=False)]#, dict(type='CopyDetectorBackbone')]

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=0.00001),
    clip_grad=dict(max_norm=10, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            #'trainable_backbone.backbone': dict(lr_mult=0.0333),
            #'trainable_backbone.neck': dict(lr_mult=10),
            'semantic_feat_projector': dict(lr_mult=10),
            'reconstruction_head': dict(lr_mult=10),
        }
    ),
)
