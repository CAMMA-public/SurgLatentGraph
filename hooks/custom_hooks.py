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

        # also freeze graph head (but not edge_sem_feat_projector) if it exists
        if self.freeze_graph_head:
            try:
                for name, p in model.graph_head.named_parameters():
                    if not 'edge_semantic_feat_projector' in name:
                        p.requires_grad = False
                for name, m in model.graph_head.named_modules():
                    if not 'edge_semantic_feat_projector' in name:
                        m.eval()

            except AttributeError as e:
                print(e)

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
