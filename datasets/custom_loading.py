from mmdet.datasets.transforms import LoadAnnotations
from mmdet.datasets import CocoDataset
from mmdet.registry import TRANSFORMS, DATASETS
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

@DATASETS.register_module()
class CocoDatasetWithDS(CocoDataset):
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        data_info = super().parse_data_info(raw_data_info)

        # get ds labels
        data_info['ds'] = raw_data_info['raw_img_info']['ds']

        return data_info
