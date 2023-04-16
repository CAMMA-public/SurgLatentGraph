from mmdet.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class FreezeDetectorHook(Hook):
    def before_train_iter(self, runner, **kwargs):
        model = runner.model
        for p in model.detector.parameters():
            p.requires_grad = False
        for m in model.detector.modules():
            m.eval()
        model.detector = model.detector.eval()

@HOOKS.register_module()
class CopyDetectorBackbone(Hook):
    def before_train(self, runner) -> None:
        # copy weights for backbone, neck if it exists
        if not runner._resume and runner.model.training:
            if 'DeformableDETR' in type(runner.model.detector).__name__:
                try:
                    runner.model.trainable_backbone.load_state_dict(runner.model.detector.state_dict())
                    print()
                    print("SUCCESSFULLY LOADED TRAINABLE BACKBONE WEIGHTS")
                    print()
                except AttributeError as e:
                    print(e)

            elif 'Mask2Former' in type(runner.model.detector).__name__:
                try:
                    runner.model.trainable_backbone.load_state_dict(runner.model.detector.state_dict())
                    print()
                    print("SUCCESSFULLY LOADED TRAINABLE BACKBONE WEIGHTS")
                    print()
                except AttributeError as e:
                    print(e)

            else:
                try:
                    runner.model.trainable_backbone.backbone.load_state_dict(runner.model.detector.backbone.state_dict())
                    runner.model.trainable_backbone.neck.load_state_dict(runner.model.detector.neck.state_dict())
                    print()
                    print("SUCCESSFULLY LOADED TRAINABLE BACKBONE WEIGHTS")
                    print()
                except AttributeError as e:
                    print(e)
