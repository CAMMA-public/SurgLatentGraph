from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList, DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models import BaseDetector
from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

@MODELS.register_module()
class SimpleCVSPredictor(BaseDetector, metaclass=ABCMeta):
    def __init__(self,
            backbone: ConfigType,
            loss: ConfigType,
            num_classes: int,
            neck: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.loss_fn = MODELS.build(loss)
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = torch.nn.Identity()

        self.predictor = torch.nn.Linear(self.backbone.feat_dim, num_classes)

    def _forward(self, batch_inputs: Tensor,
            batch_data_samples: SampleList = None):
        raise NotImplementedError

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        feats = self.extract_feat(batch_inputs)

        # TODO reconstruction if recon head is not None

        ds_preds = self.predictor(feats)
        for r, dp in zip(batch_data_samples, ds_preds):
            r.pred_ds = dp

        return batch_data_samples

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        feats = self.extract_feat(batch_inputs)
        ds_preds = self.predictor(feats)
        ds_gt = torch.stack([torch.from_numpy(b.ds) for b in batch_data_samples]).to(
                ds_preds.device).round()
        loss = {'ds_loss': self.loss_fn(ds_preds, ds_gt)}

        return loss

    def extract_feat(self, batch_inputs: Tensor):
        feats = F.adaptive_avg_pool2d(self.neck(self.backbone(batch_inputs))[-1],
                1).squeeze(-1).squeeze(-1)

        return feats
