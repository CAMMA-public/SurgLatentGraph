from mmdet.registry import MODELS
from mmdet.models.detectors import Mask2Former
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.models.seg_heads.panoptic_fusion_heads import MaskFormerFusionHead
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.mask import mask2bbox
from mmengine.structures import InstanceData
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, List

@MODELS.register_module()
class Mask2FormerWithQueries(Mask2Former):
    def extract_feat(self, batch_inputs: Tensor, return_all: bool = False) -> Tuple[Tuple[Tensor]]:
        bb_feats = self.backbone(batch_inputs)
        if self.with_neck:
            neck_feats = self.neck(bb_feats)
        else:
            neck_feats = bb_feats

        if return_all:
            return bb_feats, neck_feats
        else:
            return neck_feats

    def get_queries(self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tensor:

        bb_feats, neck_feats = self.extract_feat(batch_inputs, return_all=True)
        _, _, all_queries = self.panoptic_head(neck_feats, batch_data_samples, return_queries=True)

        # select top queries
        queries = torch.stack([q[r.pred_instances.selected_inds] for q, r in zip(
            all_queries[-1], batch_data_samples)])

        return bb_feats, neck_feats, queries

@MODELS.register_module()
class Mask2FormerHeadWithQueries(Mask2FormerHead):
    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList,
                return_queries: bool = False) -> Tuple[List[Tensor]]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        # keep queries
        queries_list = []

        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        queries_list.append(query_feat) # add query feat to llist

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            queries_list.append(query_feat) # add query feat to llist

        if return_queries:
            return cls_pred_list, mask_pred_list, queries_list
        else:
            return cls_pred_list, mask_pred_list

@MODELS.register_module()
class MaskFormerFusionHeadWithIndices(MaskFormerFusionHead):
    def instance_postprocess(self, mask_cls: Tensor,
                             mask_pred: Tensor) -> InstanceData:
        """Instance segmengation postprocess.
        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.
        Returns:
            :obj:`InstanceData`: Instance segmentation results.
                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels_per_image = labels[top_indices]

        query_indices = torch.div(top_indices, self.num_classes, rounding_mode='floor')
        mask_pred = mask_pred[query_indices]

        # extract things
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)

        results = InstanceData()
        results.bboxes = bboxes
        results.labels = labels_per_image
        results.scores = det_scores
        results.masks = mask_pred_binary
        results.selected_inds = query_indices

        return results
