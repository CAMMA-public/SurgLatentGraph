from mmdet.models.data_preprocessors import TrackDataPreprocessor
from typing import Dict
from mmdet.registry import MODELS

@MODELS.register_module()
class SavedLGPreprocessor(TrackDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> Dict:
        data = self.cast_data(data)

        # now make sure lg is cast
        for b_id in range(len(data['data_samples'])):
            for f_id in range(len(data['data_samples'][b_id].video_data_samples)):
                cast_lg = data['data_samples'][b_id].video_data_samples[f_id].metainfo['lg'].to(self.device)
                data['data_samples'][b_id].video_data_samples[f_id].set_metainfo({'lg': cast_lg})

        imgs, data_samples = data['inputs'], data['data_samples']

        # just return them
        return dict(inputs=imgs, data_samples=data_samples)
