from typing import List, Tuple, Union
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.nn.utils.rnn import pad_sequence
from mmdet.registry import MODELS
from mmengine.structures import BaseDataElement
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import scale_boxes
from mmdet.utils import ConfigType
from mmdet.models.detectors.base import BaseDetector
from .predictor_heads.modules.utils import *
from .predictor_heads.modules.layers import PositionalEncoding
from .predictor_heads.modules.mstcn import MultiStageModel as MSTCN
from .deepcvs import DeepCVS

@MODELS.register_module()
class DeepCVSTemporal(DeepCVS):
    """Temporal extension of DeepCVS. replace decoder predictor with transformer."""
    def __init__(self, clip_size: int, temporal_arch: str = 'transformer',
            causal: bool = False, per_video: bool = False, **kwargs):

        # init DeepCVS
        loss = kwargs['loss']
        super().__init__(**kwargs)

        self.clip_size = clip_size
        self.temporal_arch = temporal_arch
        self.causal = causal

        # replace decoder predictor
        if isinstance(loss, list):
            raise NotImplementedError

        self._create_temporal_model()

    def _create_temporal_model(self):
        if self.temporal_arch.lower() == 'transformer':
            norm = torch.nn.BatchNorm1d(self.decoder_backbone.feat_dim)
            pe = PositionalEncoding(d_model=self.decoder_backbone.feat_dim, batch_first=True)
            decoder_layer = torch.nn.TransformerDecoderLayer(
                    d_model=self.decoder_backbone.feat_dim, nhead=8, batch_first=True)
            temp_model = torch.nn.TransformerDecoder(decoder_layer, num_layers=3,
                    norm=norm)
            self.decoder_predictor = CustomSequential(pe, DuplicateItem(), temp_model,
                    torch.nn.Linear(self.decoder_backbone.feat_dim, self.num_classes))

        elif self.temporal_arch.lower() == 'tcn':
            self.decoder_predictor = MSTCN(2, 8, 32, self.decoder_backbone.feat_dim,
                    self.num_classes, self.causal)

        else:
            if self.temporal_arch.lower() == 'gru':
                temp_model = torch.nn.GRU(self.decoder_backbone.feat_dim, 2048, 2, batch_first=True)
            elif self.temporal_arch.lower() == 'lstm':
                temp_model = torch.nn.LSTM(self.decoder_backbone.feat_dim, 2048, 2, batch_first=True)
            else:
                raise NotImplementedError("Temporal architecture " + self.temporal_arch + " not implemented.")

            self.decoder_predictor = torch.nn.Sequential(temp_model, SelectItem(0),
                    torch.nn.Linear(2048, self.num_classes))

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        losses = {}

        ds_preds, recon_outputs = self.forward(batch_inputs, batch_data_samples)
        recon_imgs, img_targets, rescaled_results = recon_outputs

        if self.reconstruction_loss is not None:
            if self.use_pred_boxes_recon_loss:
                recon_boxes = [r.pred_instances.bboxes for r in rescaled_results]
            else:
                recon_boxes = [r.gt_instances.bboxes for r in rescaled_results]

            recon_losses = self.reconstruction_loss(recon_imgs, img_targets, recon_boxes)
            losses.update(recon_losses)

        # get gt
        ds_gt = torch.stack([torch.stack([torch.from_numpy(b.ds) for b in vds]) \
                for vds in batch_data_samples]).to(ds_preds.device).float().flatten(end_dim=1)
        ds_preds = ds_preds.flatten(end_dim=1)

        if self.loss_consensus == 'mode':
            ds_gt = ds_gt.round().long()
        elif self.loss_consensus == 'prob':
            # interpret GT as probability of 1 and randomly generate gt
            random_probs = torch.rand_like(ds_gt) # random probability per label per example
            ds_gt = torch.le(random_probs, ds_gt).long().to(ds_gt.device)
        else:
            ds_gt = ds_gt.long()

        if isinstance(self.loss_fn, torch.nn.ModuleList):
            raise NotImplementedError

        else:
            ds_loss = self.loss_fn(ds_preds, ds_gt)

        # update loss
        losses.update({'ds_loss': ds_loss})

        return losses

    def forward(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Tensor:
        if self.per_video:
            filtered_batch_data_samples = [[b for ind, b in enumerate(bds) \
                    if ind in bds.key_frames_inds] for bds in batch_data_samples]
        else:
            filtered_batch_data_samples = batch_data_samples

        flat_batch_inputs = batch_inputs.flatten(end_dim=1)
        flat_batch_data_samples = [x for y in batch_data_samples for x in y]

        breakpoint()

        return super().forward(flat_batch_inputs, flat_batch_data_samples)
