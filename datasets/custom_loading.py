from mmdet.datasets.transforms import LoadAnnotations
from mmdet.datasets import CocoDataset
from mmdet.registry import TRANSFORMS, DATASETS
from mmtrack.datasets import BaseVideoDataset
from mmtrack.datasets.transforms import LoadTrackAnnotations
from mmtrack.datasets.samplers import VideoSampler, EntireVideoBatchSampler
from mmcv.transforms import BaseTransform
from typing import List, Union
import numpy as np


@TRANSFORMS.register_module()
class LoadAnnotationsWithDS(LoadAnnotations):
    def _load_ds(self, results: dict) -> None:
        """Private function to load downstream annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_ds = results.get('ds')
        results['ds'] = np.array(gt_ds)

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        self._load_ds(results)
        return results

@TRANSFORMS.register_module()
class LoadTrackAnnotationsWithDS(LoadTrackAnnotations):
    def _load_ds(self, results: dict) -> None:
        """Private function to load downstream annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        breakpoint()
        gt_ds = results.get('ds')
        results['ds'] = np.array(gt_ds)

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        self._load_ds(results)
        return results

@DATASETS.register_module()
class CocoDatasetWithDS(CocoDataset):
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        data_info = super().parse_data_info(raw_data_info)

        # get ds labels
        data_info['ds'] = raw_data_info['raw_img_info']['ds']

        return data_info

@DATASETS.register_module()
class VideoDatasetWithDS(BaseVideoDataset):
    # TODO(adit98) figure out how to load from annot file
    METAINFO = {
        'CLASSES':
        ('abdominal_wall', 'liver', 'gastrointestinal_wall', 'fat', 'grasper', 'connective_tissue',
        'blood', 'cystic_duct', 'hook', 'gallbladder', 'hepatic_vein', 'liver_ligament')
    }

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        data_info = super().parse_data_info(raw_data_info)

        # get ds labels
        if 'ds' in raw_data_info['raw_img_info']:
            data_info['ds'] = raw_data_info['raw_img_info']['ds']

        data_info['is_det_keyframe'] = raw_data_info['raw_img_info']['is_det_keyframe']

        return data_info
