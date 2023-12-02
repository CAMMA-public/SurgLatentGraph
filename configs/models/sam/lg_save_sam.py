_base_ = ['lg_sam.py']

orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['visualizer.SAMQueryVisualizer'],
        allow_failed_imports=False)

# set number of objects per img
model = dict(
    detector=dict(
        num_nodes=128,
    ),
    #trainable_backbone_cfg = dict(
    #    type='ResNet',
    #    depth=50,
    #    num_stages=4,
    #    out_indices=(0, 1, 2, 3),
    #    frozen_stages=-1,
    #    norm_cfg=dict(type='BN', requires_grad=True),
    #    norm_eval=True,
    #    style='pytorch',
    #    init_cfg=dict(type='Pretrained', checkpoint='weights/ssl_weights/no_phase/converted_moco.torch'),
    #    #init_cfg=dict(type='Pretrained', checkpoint='weights/endoscapes/lg_faster_rcnn_no_recon_bb.pth'),
    #),
    ##trainable_neck_cfg = dict(
    ##    type='FPN',
    ##    in_channels=[256, 512, 1024, 2048],
    ##    out_channels=256,
    ##    num_outs=5,
    ##),
    #trainable_neck_cfg = dict(
    #    type='ChannelMapper',
    #    in_channels=[256, 512, 1024, 2048],
    #    out_channels=256,
    #),
    #roi_extractor=dict(
    #    type='SingleRoIExtractor',
    #    roi_layer=dict(type='RoIAlign', output_size=1, sampling_ratio=0),
    #    out_channels=256,
    #    featmap_strides=[4, 8, 16, 32],
    #),
)

visualizer = dict(
    type='SAMQueryVisualizer',
    prefix='endoscapes',
    draw=False,
)

default_hooks = dict(
    visualization=dict(
        draw=True,
    ),
)

test_dataloader = dict(
    dataset=dict(
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/'),
    )
)
