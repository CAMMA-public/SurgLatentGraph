from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList, DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models import BaseDetector
from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Dict, List, Tuple, Union

@MODELS.register_module()
class SimpleCVSPredictor(BaseDetector, metaclass=ABCMeta):
    def __init__(self,
            backbone: ConfigType,
            loss: ConfigType,
            num_classes: int,
            aspect_ratio: Union[Tuple, List] = (4, 4),
            bottleneck_feat_size: int = 64,
            img_decoder: OptConfigType = None,
            reconstruction_loss: OptConfigType = None,
            reconstruction_img_stats: ConfigType = None,
            neck: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else torch.nn.Identity()
        self.predictor = torch.nn.Linear(self.backbone.feat_dim, num_classes)

        if img_decoder is not None:
            self.bottleneck = torch.nn.Linear(self.backbone.feat_dim, bottleneck_feat_size)
            img_decoder.dims = (bottleneck_feat_size + 1, *img_decoder.dims) # append dummy mask to recon input
            img_decoder.source_image_dims = 1 # dummy mask dim
            self.img_decoder = MODELS.build(img_decoder) if img_decoder is not None else None
            self.aspect_ratio = aspect_ratio
            num_upsampling_stages = len(img_decoder.dims) - 1
            self.reconstruction_size = (Tensor(aspect_ratio) * (2 ** num_upsampling_stages)).int()
            self.reconstruction_loss = MODELS.build(reconstruction_loss)
        else:
            self.img_decoder = None

        self.loss_fn = MODELS.build(loss)
        self.reconstruction_loss = MODELS.build(reconstruction_loss) \
                if reconstruction_loss is not None else None
        self.reconstruction_img_stats = reconstruction_img_stats

    def _forward(self, batch_inputs: Tensor,
            batch_data_samples: SampleList = None):
        raise NotImplementedError

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        feats = self.extract_feat(batch_inputs)

        # ds preds
        ds_preds = self.predictor(F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1))

        # reconstruction if recon head is not None
        recon_imgs, _ = self.reconstruct(batch_inputs, feats)

        for r, dp, r_img in zip(batch_data_samples, ds_preds, recon_imgs):
            r.pred_ds = dp
            if r_img is not None:
                # renormalize img
                norm_r_img = r_img * Tensor(self.reconstruction_img_stats.std).view(-1, 1, 1).to(r_img.device) / 255 + \
                        Tensor(self.reconstruction_img_stats.mean).view(-1, 1, 1).to(r_img.device) / 255
                r.reconstruction = torch.clamp(norm_r_img, 0, 1)

        return batch_data_samples

    def reconstruct(self, batch_inputs: Tensor, feats: Tensor) -> Tensor:
        if self.img_decoder is None:
            return [None] * len(batch_inputs), None

        # resize feats to aspect ratio, bottleneck
        recon_feats = self.bottleneck(TF.resize(feats, self.reconstruction_size.tolist()).transpose(1, -1)).transpose(1, -1)

        # create dummy layout and concat with recon feats for SPADE compatibility
        dummy_layout = torch.zeros(feats.shape[0], 1, *self.reconstruction_size).to(recon_feats.device)
        recon_input = torch.cat([recon_feats, dummy_layout], dim=1)

        # reconstruct img
        recon_img = self.img_decoder(recon_input)

        # resize gt
        img_targets = TF.resize(batch_inputs, self.reconstruction_size.tolist())

        return recon_img, img_targets

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        feats = self.extract_feat(batch_inputs)

        # ds preds and loss
        ds_preds = self.predictor(F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1))
        ds_gt = torch.stack([torch.from_numpy(b.ds) for b in batch_data_samples]).to(
                ds_preds.device).round()
        loss = {'ds_loss': self.loss_fn(ds_preds, ds_gt)}

        # reconstruction if recon head is not None
        recon_imgs, img_targets = self.reconstruct(batch_inputs, feats)
        if self.reconstruction_loss is not None:
            loss.update(self.reconstruction_loss(recon_imgs, img_targets))

        return loss

    def extract_feat(self, batch_inputs: Tensor):
        feats = self.neck(self.backbone(batch_inputs))[-1]

        return feats
