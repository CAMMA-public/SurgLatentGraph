_base_ = 'simple_classifier.py'

model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(type='Pretrained',
            checkpoint='torchvision://resnet152',
        ),
    ),
)

# evaluators
train_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/train/r152',
)
val_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/val/r152',
)

test_evaluator = dict(
    outfile_prefix='./results/endoscapes_preds/test/r152',
)
