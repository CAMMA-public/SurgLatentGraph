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
        self.per_video = per_video

        # replace decoder predictor
        if isinstance(loss, list):
            raise NotImplementedError

        self._create_temporal_model()

    def _create_temporal_model(self):
        if self.temporal_arch.lower() == 'transformer':
            norm = torch.nn.LayerNorm(self.decoder_backbone.feat_dim)
            pe = PositionalEncoding(d_model=self.decoder_backbone.feat_dim, batch_first=True)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=self.decoder_backbone.feat_dim, nhead=8, batch_first=True)
            temp_model = torch.nn.TransformerEncoder(encoder_layer, num_layers=6,
                    norm=norm)
            self.decoder_predictor = CustomSequential(pe, temp_model,
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
        if self.per_video:
            filtered_batch_data_samples = [[b for ind, b in enumerate(bds) \
                    if ind in bds.key_frames_inds] for bds in batch_data_samples]
        else:
            filtered_batch_data_samples = batch_data_samples

        ds_preds, recon_outputs, _ = self._forward(batch_inputs, filtered_batch_data_samples)
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

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        if self.per_video:
            filtered_batch_data_samples = [[b for ind, b in enumerate(bds) \
                    if ind in bds.key_frames_inds] for bds in batch_data_samples]
        else:
            filtered_batch_data_samples = batch_data_samples

        ds_preds, recon_outputs, results = self._forward(batch_inputs,
                filtered_batch_data_samples)
        recon_imgs, _, _ = recon_outputs

        # only keep keyframes
        if self.per_video:
            # upsample predictions if needed
            if len(filtered_batch_data_samples[0]) < len(batch_data_samples[0]):
                # compute upsample factor
                upsample_factor = int(np.ceil(len(batch_data_samples[0]) / len(filtered_batch_data_samples[0])))

                # pad predictions
                ds_preds = ds_preds.repeat_interleave(upsample_factor, dim=1)[:,
                        :len(batch_data_samples[0])]

                # pad results and add ds_preds
                padded_results = []
                dummy_instance_data = InstanceData(bboxes=torch.zeros(0, 4),
                        scores=torch.zeros(0), labels=torch.zeros(0)).to(ds_preds.device)

                for ind, (b, ds) in enumerate(zip(batch_data_samples[0], ds_preds.flatten(end_dim=1))):
                    if ind % upsample_factor == 0:
                        r = results[ind // upsample_factor]

                    else:
                        r = DetDataSample(metainfo=b.metainfo,
                                pred_instances=dummy_instance_data)

                    r.pred_ds = ds
                    padded_results.append(r)

            else:
                for r in results:
                    for r, p in zip(results, ds_preds.view(-1, ds_preds.shape[-1])):
                        r.pred_ds = p

        else:
            results = results[self.clip_size-1::self.clip_size]
            ds_preds = ds_preds[:, -1] # keep only last frame preds
            for r, p in zip(results, ds_preds):
                r.pred_ds = p

        return results

    def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Tensor:
        flat_batch_inputs = batch_inputs.flatten(end_dim=1)
        flat_batch_data_samples = [x for y in batch_data_samples for x in y]

        with torch.no_grad():
            # run detector to get detections
            results = self.detector.predict(flat_batch_inputs, flat_batch_data_samples)

        # get feats
        ds_feats, ds_bb_feats = self.extract_feat(flat_batch_inputs, results)

        # reconstruction if recon head is not None
        recon_outputs = self.reconstruct(flat_batch_inputs, results, ds_feats, ds_bb_feats)

        # ds prediction
        if isinstance(self.decoder_predictor, torch.nn.ModuleList):
            raise NotImplementedError
        else:
            ds_preds = self.decoder_predictor(ds_feats.view(
                -1, self.clip_size, *ds_feats.shape[1:]))

        return ds_preds, recon_outputs, results
