from mmdet.models.data_preprocessors import TrackDataPreprocessor
from typing import Dict
from mmdet.registry import MODELS
import imagesize

@MODELS.register_module()
class SavedLGPreprocessor(TrackDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> Dict:
        data = self.cast_data(data)

        # now make sure lg is cast
        for b_id in range(len(data['data_samples'])):
            for f_id in range(len(data['data_samples'][b_id].video_data_samples)):
                cast_lg = data['data_samples'][b_id].video_data_samples[f_id].metainfo['lg'].to(self.device)
                if 'img_shape' in cast_lg:
                    data['data_samples'][b_id].video_data_samples[f_id].set_metainfo({'img_shape': tuple(cast_lg['img_shape'].tolist())})
                else:
                    im_shape = imagesize.get(data['data_samples'][b_id].video_data_samples[f_id].img_path)[::-1]
                    data['data_samples'][b_id].video_data_samples[f_id].set_metainfo({'img_shape': im_shape})

                data['data_samples'][b_id].video_data_samples[f_id].set_metainfo({'lg': cast_lg})

        imgs, data_samples = data['inputs'], data['data_samples']

        # just return them
        return dict(inputs=imgs, data_samples=data_samples)
