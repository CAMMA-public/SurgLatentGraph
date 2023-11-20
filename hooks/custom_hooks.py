from mmdet.registry import HOOKS
from mmengine.hooks import Hook
from prettytable import PrettyTable
import torch

@HOOKS.register_module()
class FreezeHook(Hook):
    def __init__(self, freeze_detector: bool = True, freeze_graph_head: bool = False,
            freeze_projectors: bool = False):
        self.freeze_detector = freeze_detector
        self.freeze_graph_head = freeze_graph_head
        self.freeze_projectors = freeze_projectors
        if self.freeze_detector:
            print("FREEZING DETECTOR")
        if self.freeze_graph_head:
            print("FREEZING GRAPH HEAD")
        if self.freeze_projectors:
            print("FREEZING PROJECTORS")

    def before_train_iter(self, runner, **kwargs):
        model = runner.model
        if self.freeze_detector:
            for p in model.detector.parameters():
                p.requires_grad = False
            for m in model.detector.modules():
                m.eval()

            model.detector.eval()

        if self.freeze_graph_head:
            for name, p in model.graph_head.named_parameters():
                p.requires_grad = False
            for name, m in model.graph_head.named_modules():
                m.eval()

            model.graph_head.eval()

        if self.freeze_projectors:
            for name, p in model.named_parameters():
                if 'semantic_feat_projector' in name:
                    p.requires_grad = False
            for name, m in model.named_modules():
                if 'semantic_feat_projector' in name:
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

@HOOKS.register_module()
class ClearGPUMem(Hook):
    def after_train_iter(self, runner, **kwargs) -> None:
        torch.cuda.empty_cache()

    def after_val_iter(self, runner, **kwargs) -> None:
        torch.cuda.empty_cache()

    def after_test_iter(self, runner, **kwargs) -> None:
        torch.cuda.empty_cache()
