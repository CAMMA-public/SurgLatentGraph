from mmdet.registry import HOOKS
from mmengine.hooks import Hook
from prettytable import PrettyTable

@HOOKS.register_module()
class FreezeDetectorHook(Hook):
    def __init__(self, train_ds_only=False):
        self.train_ds_only = train_ds_only

    def before_train_iter(self, runner, **kwargs):
        model = runner.model
        for p in model.detector.parameters():
            p.requires_grad = False
        for m in model.detector.modules():
            m.eval()

        model.detector.eval()

        # if only training ds, freeze everything but ds_head
        if self.train_ds_only:
            for name, p in model.named_parameters():
                if 'ds_head' in name:
                    continue

                p.requires_grad = False

            for name, m in model.graph_head.named_modules():
                if 'ds_head' in name:
                    continue

                m.eval()

@HOOKS.register_module()
class CountTrainableParameters(Hook):
    def before_train(self, runner) -> None:
        self.count_parameters(runner.model)

    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params

        print(table)
        print(f"Total Trainable Params: {total_params}")

@HOOKS.register_module()
class FreezeLGDetector(Hook):
    def __init__(self, finetune_backbone=False):
        self.finetune_backbone = finetune_backbone

    def before_train_iter(self, runner, **kwargs):
        model = runner.model
        if self.finetune_backbone:
            # only freeze detector in lg detector
            for p in model.lg_detector.detector.parameters():
                p.requires_grad = False
            for m in model.lg_detector.detector.modules():
                m.eval()

            model.lg_detector.detector.eval()

        else:
            for p in model.lg_detector.parameters():
                p.requires_grad = False
            for m in model.lg_detector.modules():
                m.eval()

        model.lg_detector.eval()

@HOOKS.register_module()
class CopyDetectorBackbone(Hook):
    def __init__(self, temporal=False):
        self.temporal = temporal

    def before_train(self, runner) -> None:
        # copy weights for backbone, neck if it exists
        if not runner._resume and runner.model.training:
            # NOTE hack
            if self.temporal:
                det = runner.model.lg_detector.detector
                lg_model = runner.model.lg_detector
            else:
                det = runner.model.detector
                lg_model = runner.model

            if 'DeformableDETR' in type(det).__name__:
                try:
                    lg_model.trainable_backbone.load_state_dict(det.state_dict())
                    print()
                    print("SUCCESSFULLY LOADED TRAINABLE BACKBONE WEIGHTS")
                    print()
                except AttributeError as e:
                    print(e)

            elif 'Mask2Former' in type(det).__name__:
                try:
                    lg_model.trainable_backbone.load_state_dict(det.state_dict())
                    print()
                    print("SUCCESSFULLY LOADED TRAINABLE BACKBONE WEIGHTS")
                    print()
                except AttributeError as e:
                    print(e)

            else:
                try:
                    lg_model.trainable_backbone.backbone.load_state_dict(det.backbone.state_dict())
                    try:
                        lg_model.trainable_backbone.neck.load_state_dict(det.neck.state_dict())
                    except RuntimeError as re:
                        print("SKIPPING LOADING OF NECK WEIGHTS")

                    print()
                    print("SUCCESSFULLY LOADED TRAINABLE BACKBONE WEIGHTS")
                    print()
                except AttributeError as e:
                    print(e)
