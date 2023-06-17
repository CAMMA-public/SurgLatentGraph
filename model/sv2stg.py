import torch
from typing import List, Tuple, Union
from torch import Tensor
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from mmengine.structures import SampleList
from mmdet.structures.bbox import bbox2roi, roi2bbox, scale_boxes
from mmdet.models.detectors.base import BaseDetector
from mmengine.structures import BaseDataElement
from mmdet.registry import MODELS

@MODELS.register_module()
class SV2STG(BaseDetector):
    def __init__(self, lg_detector: BaseDetector, ds_head: ConfigType):
        self.lg_detector = lg_detector
        self.ds_head = MODELS.build(ds_head)

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        # ... existing code ...
        breakpoint()

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        B, T = batch_inputs.shape[:2]
        feats, graph = self.lg_detector.extract_lg(batch_inputs.flatten(end_dim=1),
                batch_data_samples)
        breakpoint()

    def build_st_graph(self, feats: BaseDataElement, graph: BaseDataElement):
        raise NotImplementedError

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError
