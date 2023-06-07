from mmdet.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class FreezeDetectorHook(Hook):
    def __init__(self, freeze_graph_head=False):
        self.freeze_graph_head = freeze_graph_head

    def before_train_iter(self, runner, **kwargs):
        model = runner.model
        for p in model.detector.parameters():
            p.requires_grad = False
        for m in model.detector.modules():
            m.eval()

        model.detector.eval()

        # also freeze graph head if it exists
        if self.freeze_graph_head:
            try:
                for p in model.graph_head.parameters():
                    p.requires_grad = False
                for m in model.graph_head.modules():
                    m.eval()

                model.graph_head.eval()

            except AttributeError as e:
                print(e)

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
                    try:
                        runner.model.trainable_backbone.neck.load_state_dict(runner.model.detector.neck.state_dict())
                    except RuntimeError as re:
                        print("SKIPPING LOADING OF NECK WEIGHTS")

                    print()
                    print("SUCCESSFULLY LOADED TRAINABLE BACKBONE WEIGHTS")
                    print()
                except AttributeError as e:
                    print(e)
