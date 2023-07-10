from mmdet.datasets import CocoDataset
from mmdet.registry import TRANSFORMS, DATASETS, DATA_SAMPLERS
from mmdet.datasets import BaseVideoDataset
from mmdet.datasets.samplers import TrackImgSampler
from mmdet.datasets.transforms import LoadAnnotations, UniformRefFrameSample, LoadTrackAnnotations
from mmengine.dataset import ClassBalancedDataset, ConcatDataset
from mmengine.dist import get_dist_info, sync_random_seed
from typing import List, Union, Sized, Optional, Any
import numpy as np
import math
import random
from collections import defaultdict

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

    def prepare_data(self, idx) -> Any:
        """Get date processed by ``self.pipeline``. Note that ``idx`` is a
        video index in default since the base element of video dataset is a
        video. However, in some cases, we need to specific both the video index
        and frame index. For example, in traing mode, we may want to sample the
        specific frames and all the frames must be sampled once in a epoch; in
        test mode, we may want to output data of a single image rather than the
        whole video for saving memory.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        if isinstance(idx, tuple):
            assert len(idx) == 2, 'The length of idx must be 2: '
            '(video_index, frame_index)'
            video_idx, frame_idx = idx[0], idx[1]
        else:
            video_idx, frame_idx = idx, None

        data_info = self.get_data_info(video_idx)
        if self.test_mode:
            # Support two test_mode: frame-level and video-level
            final_data_info = defaultdict(list)
            if frame_idx is None:
                frames_idx_list = list(range(data_info['video_length']))
            else:
                frames_idx_list = [frame_idx]
                final_data_info['key_frame_id'] = frame_idx

            for index in frames_idx_list:
                frame_ann = data_info['images'][index]
                frame_ann['video_id'] = data_info['video_id']
                # Collate data_list (list of dict to dict of list)
                for key, value in frame_ann.items():
                    final_data_info[key].append(value)
                # copy the info in video-level into img-level
                # TODO: the value of this key is the same as that of
                # `video_length` in test mode
                final_data_info['ori_video_length'].append(
                    data_info['video_length'])

            final_data_info['video_length'] = data_info['video_length']
            return self.pipeline(final_data_info)

        else:
            # Specify `key_frame_id` for the frame sampling in the pipeline
            if frame_idx is not None:
                data_info['key_frame_id'] = frame_idx
            return self.pipeline(data_info)

    @property
    def num_total_keyframes(self):
        """Get the number of all the keyframes in this video dataset."""
        return sum(
            [len([x for x in self.get_data_info(i)['images'] if x['is_ds_keyframe']]) \
                    for i in range(len(self))])

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

@TRANSFORMS.register_module()
class UniformRefFrameSampleWithPad(UniformRefFrameSample):
    def sampling_frames(self, video_length: int, key_frame_id: int):
        """Sampling frames.

        Args:
            video_length (int): The length of the video.
            key_frame_id (int): The key frame id.

        Returns:
            list[int]: The sampled frame indices.
        """
        if video_length > 1:
            left = max(0, key_frame_id + self.frame_range[0])
            right = min(key_frame_id + self.frame_range[1], video_length - 1)
            frame_ids = list(range(0, video_length))

            valid_ids = frame_ids[left:right + 1]
            if self.filter_key_img and key_frame_id in valid_ids:
                valid_ids.remove(key_frame_id)

            # custom logic to pad with initial frame when we don't have enough history
            if len(valid_ids) == 0:
                valid_ids = [key_frame_id] * self.num_ref_imgs
            elif len(valid_ids) < self.num_ref_imgs:
                len_diff = self.num_ref_imgs - len(valid_ids)
                valid_ids = [valid_ids[0]] * len_diff + valid_ids

            ref_frame_ids = random.sample(valid_ids, self.num_ref_imgs)

        else:
            ref_frame_ids = [key_frame_id] * self.num_ref_imgs

        sampled_frames_ids = [key_frame_id] + ref_frame_ids
        sampled_frames_ids = sorted(sampled_frames_ids)

        key_frames_ind = sampled_frames_ids.index(key_frame_id)
        key_frame_flags = [False] * len(sampled_frames_ids)
        key_frame_flags[key_frames_ind] = True
        return sampled_frames_ids, key_frame_flags
