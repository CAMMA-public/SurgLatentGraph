from mmdet.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from typing import Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
import torch
import os
from segment_anything.utils.amg import coco_encode_rle, mask_to_rle_pytorch, area_from_rle

@VISUALIZERS.register_module()
class SAMQueryVisualizer(DetLocalVisualizer):
    def __init__(self, name: str, prefix: str = 'endoscapes', draw: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
        self.draw = draw

    def add_datasample(self, name: str, image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            out_file: Optional[str] = None, **kwargs):

        if self.draw:
            super().add_datasample(name, image, data_sample, out_file=out_file, **kwargs)

        save_dir = os.path.join('sam_queries', self.prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        query_filename = str(data_sample.img_id) + '.npz'

        mask_anns = self.encode_masks(data_sample.pred_instances.masks)
        data_subsample = dict(
                img_path=data_sample.img_path,
                bboxes=data_sample.pred_instances.bboxes.cpu().numpy(),
                feats=data_sample.pred_instances.feats.cpu().numpy(),
                graph_feats=data_sample.pred_instances.graph_feats.cpu().numpy(),
                masks=mask_anns,
        )
        np.savez(os.path.join(save_dir, query_filename), data_subsample)

    def encode_masks(self, raw_masks):
        mask_rles = mask_to_rle_pytorch(raw_masks)
        coco_rles = [coco_encode_rle(m) for m in mask_rles]
        mask_anns = []
        for idx in range(len(coco_rles)):
            ann = {
                "segmentation": coco_rles[idx],
            }
            mask_anns.append(ann)

        return mask_anns
