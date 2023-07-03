from mmdet.datasets.transforms import LoadAnnotations
from mmdet.datasets import CocoDataset
from mmdet.registry import TRANSFORMS, DATASETS, DATA_SAMPLERS
from mmdet.datasets import BaseVideoDataset
from mmdet.datasets.transforms import LoadTrackAnnotations
from mmdet.datasets.samplers import TrackImgSampler
from mmcv.transforms import BaseTransform
from mmengine.dataset import ClassBalancedDataset, ConcatDataset
from mmengine.dist import get_dist_info, sync_random_seed
from typing import List, Union, Sized, Optional
import numpy as np
import math

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
        'classes':
        ('abdominal_wall', 'liver', 'gastrointestinal_wall', 'fat', 'grasper', 'connective_tissue',
        'blood', 'cystic_duct', 'hook', 'gallbladder', 'hepatic_vein', 'liver_ligament')
    }
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        data_info = super().parse_data_info(raw_data_info)

        # get ds labels
        data_info['ds'] = raw_data_info['raw_img_info']['ds']

        return data_info

    def get_keyframes_per_video(self, idx):
        """Get all keyframes in one video.

        Args:
            idx (int): Index of video.

        Returns:
            List[int]: a list of all keyframes in the video
        """
        data_info = self.get_data_info(idx)['images']
        keyframe_ids = [x['frame_id'] for x in data_info if x['is_ds_keyframe']]

        return keyframe_ids

@DATA_SAMPLERS.register_module()
class TrackCustomKeyframeSampler(TrackImgSampler):
    def __init__(
        self,
        dataset: Sized,
        seed: Optional[int] = None,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        if seed is None:
            self.seed = sync_random_seed()
        else:
            self.seed = seed

        self.dataset = dataset
        self.indices = []
        # Hard code here to handle different dataset wrapper
        if isinstance(self.dataset, ConcatDataset):
            cat_datasets = self.dataset.datasets
            assert isinstance(
                cat_datasets[0], VideoDatasetWithDS
            ), f'expected VideoDatasetWithDS, but got {type(cat_datasets[0])}'
            self.test_mode = cat_datasets[0].test_mode
            assert not self.test_mode, "'ConcatDataset' should not exist in "
            'test mode'
            for dataset in cat_datasets:
                num_videos = len(dataset)
                for video_ind in range(num_videos):
                    self.indices.extend([
                        (video_ind, frame_ind) for frame_ind in range(
                            dataset.get_len_per_video(video_ind))
                    ])
        elif isinstance(self.dataset, ClassBalancedDataset):
            ori_dataset = self.dataset.dataset
            assert isinstance(
                ori_dataset, VideoDatasetWithDS
            ), f'expected VideoDatasetWithDS, but got {type(ori_dataset)}'
            self.test_mode = ori_dataset.test_mode
            assert not self.test_mode, "'ClassBalancedDataset' should not "
            'exist in test mode'
            video_indices = self.dataset.repeat_indices
            for index in video_indices:
                self.indices.extend([(index, frame_ind) for frame_ind in range(
                    ori_dataset.get_len_per_video(index))])
        else:
            assert isinstance(
                self.dataset, VideoDatasetWithDS
            ), 'TrackCustomKeyframeSampler is only supported in VideoDatasetWithDS or '
            'dataset wrapper: ClassBalancedDataset and ConcatDataset, but '
            f'got {type(self.dataset)} '
            self.test_mode = self.dataset.test_mode
            num_videos = len(self.dataset)

            if self.test_mode:
                # in test mode, the images belong to the same video must be put
                # on the same device.
                if num_videos < self.world_size:
                    raise ValueError(f'only {num_videos} videos loaded,'
                                     f'but {self.world_size} gpus were given.')
                chunks = np.array_split(
                    list(range(num_videos)), self.world_size)
                for videos_inds in chunks:
                    indices_chunk = []
                    for video_ind in videos_inds:
                        indices_chunk.extend([
                            (video_ind, frame_ind) for frame_ind in self.dataset.get_keyframes_per_video(video_ind)
                        ])
                    self.indices.append(indices_chunk)
            else:
                for video_ind in range(num_videos):
                    self.indices.extend([
                        (video_ind, frame_ind) for frame_ind in self.dataset.get_keyframes_per_video(video_ind)
                    ])

        if self.test_mode:
            self.num_samples = len(self.indices[self.rank])
            self.total_size = sum(
                [len(index_list) for index_list in self.indices])
        else:
            self.num_samples = int(
                math.ceil(len(self.indices) * 1.0 / self.world_size))
            self.total_size = self.num_samples * self.world_size
