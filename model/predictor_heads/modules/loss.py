from mmdet.registry import MODELS
from typing import List, Tuple, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode
from torchvision.models import resnet50, vgg16
from torchmetrics.functional import multiscale_structural_similarity_index_measure as ms_ssim, structural_similarity_index_measure as ssim

@MODELS.register_module()
class ReconstructionLoss(nn.Module):
    def __init__(self, l1_weight: float, deep_loss_weight: float, ssim_weight: float,
            perceptual_weight: float, box_loss_weight: float, recon_loss_weight: float,
            use_content: bool, use_style: bool, use_ssim: bool, use_l1: bool,
            deep_loss_backbone: str = 'vgg', load_backbone_weights: str = None):
        super(ReconstructionLoss, self).__init__()

        # store imnet mean and std
        self.imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        # set loss term flags
        self.use_content = use_content
        self.use_style = use_style
        self.use_deep = use_content or use_style
        self.use_ssim = use_ssim
        self.use_l1 = use_l1

        # overall weights in final loss
        self.l1_weight = np.clip(l1_weight, 0, 1)
        self.deep_loss_weight = np.clip(deep_loss_weight, 0, 1)
        self.ssim_weight = np.clip(ssim_weight, 0, 1)
        self.box_loss_weight = box_loss_weight
        self.recon_loss_weight = recon_loss_weight

        # weight of content vs style terms in deep loss term
        self.perceptual_weight = np.clip(perceptual_weight, 0, 1)
        if use_content and not use_style:
            self.perceptual_weight = 1
        elif use_style and not use_content:
            self.perceptual_weight = 0

        # fix weights based on loss terms in use
        if not self.use_deep:
            self.deep_loss_weight = 0
        if not self.use_l1:
            self.l1_weight = 0
        if not self.use_ssim:
            self.ssim_weight = 0

        # reweight
        weights = Tensor([self.l1_weight, self.ssim_weight, self.deep_loss_weight])
        self.l1_weight, self.ssim_weight, self.deep_loss_weight = weights / torch.sum(weights)

        # initialize backbone for deep loss
        if 'vgg' in deep_loss_backbone:
            self.deep_loss_backbones = []
            backbone = vgg16(pretrained=True).features
            # conv1_2, conv2_2, conv3_2, conv4_2, and conv5_2
            self.deep_loss_backbones += [backbone[:3], backbone[3:8],
                    backbone[8:13], backbone[13:20],
                    backbone[20:27]]
        elif 'resnet50' in deep_loss_backbone:
            backbone = resnet50(pretrained=True)
            if load_backbone_weights is not None:
                backbone.load_state_dict(torch.load(load_backbone_weights), strict=False)

            self.deep_loss_backbones = []
            self.deep_loss_backbones.append(
                    torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            )
            self.deep_loss_backbones.append(backbone.layer1)
            self.deep_loss_backbones.append(backbone.layer2)
            self.deep_loss_backbones.append(backbone.layer3)
            self.deep_loss_backbones.append(backbone.layer4)

        else:
            raise NotImplementedError(deep_loss_backbone)

    def convert_img(self, img):
        converted_img = (img * self.imnet_std.to(img.device)) + self.imnet_mean.to(img.device)

        # clip between 0 and 1
        converted_img = torch.clamp(converted_img, 0, 1)

        return converted_img

    def forward(self, reconstructed_imgs, orig_imgs, boxes=None):
        # if boxes is supplied, crop patches and evaluate additional loss
        # NOTE expects boxes in xyxy format
        if boxes is not None:
            box_ssim_loss = []
            box_l1_loss = []
            box_deep_loss = []
            all_pred_patches, all_gt_patches, fixed_boxes = self.crop_boxes(reconstructed_imgs,
                    orig_imgs, boxes, min_size=12)

            pred_patch_batch, gt_patch_batch, mask_batch = [], [], []
            for pred_img, gt_img in zip(all_pred_patches, all_gt_patches):
                for p_patch, gt_patch in zip(pred_img, gt_img):
                    if self.use_ssim:
                        box_ssim_loss.append(1 - ssim(p_patch.unsqueeze(0), gt_patch.unsqueeze(0)))
                    else:
                        box_ssim_loss.append(torch.zeros(1).squeeze().to(orig_imgs.device))

                    if self.use_l1:
                        box_l1_loss.append(F.mse_loss(p_patch, gt_patch))
                    else:
                        box_l1_loss.append(torch.zeros(1).squeeze().to(orig_imgs.device))

            box_ssim_loss = sum(box_ssim_loss) / (len(box_ssim_loss) + 1e-5)
            box_l1_loss = sum(box_l1_loss) / (len(box_l1_loss) + 1e-5)
            total_box_loss = self.ssim_weight * box_ssim_loss + \
                    self.l1_weight * box_l1_loss

        else:
            box_ssim_loss = torch.zeros(1).to(orig_imgs.device).squeeze()
            box_l1_loss = torch.zeros(1).to(orig_imgs.device).squeeze()
            total_box_loss = torch.zeros(1).to(orig_imgs.device).squeeze()

        if self.use_ssim:
            ssim_loss = 1 - ssim(reconstructed_imgs, orig_imgs)
        else:
            ssim_loss = torch.zeros(1).to(orig_imgs.device).squeeze()

        if self.use_l1:
            l1_loss = F.mse_loss(reconstructed_imgs, orig_imgs)
        else:
            l1_loss = torch.zeros(1).to(orig_imgs.device).squeeze()

        if self.use_deep:
            content_loss, style_loss = self.deep_loss(reconstructed_imgs, orig_imgs)

            # iterate through content, style and compute weighted mean
            box_deep_loss = []
            deep_loss = []
            if boxes is not None:
                # construct fg_mask for deep loss based on boxes
                fg_masks = torch.stack([self.boxes_to_mask(b, reconstructed_imgs.shape[-2:]) for b in fixed_boxes])

            for c, s in zip(content_loss, style_loss):
                if boxes is not None:
                    new_shape = c.shape[-2:]

                    # resize masks
                    resized_masks = TF.resize(fg_masks, new_shape, interpolation=InterpolationMode.NEAREST)

                # mask each loss
                overall_box_loss = torch.zeros(1).squeeze().to(orig_imgs.device)
                overall_loss = torch.zeros(1).squeeze().to(orig_imgs.device)

                if c is not None:
                    if boxes is not None and len(torch.cat(boxes)) > 0:
                        denom = resized_masks.flatten(start_dim=1).sum(1) + 1e-3
                        overall_box_loss += ((c.mean(dim=1, keepdims=True) * resized_masks).flatten(start_dim=1).sum(1) / denom).mean()

                    overall_loss += c.mean()

                if s is not None:
                    #if boxes is not None and len(torch.cat(boxes)) > 0:
                    #    denom = resized_masks.flatten(start_dim=1).sum(1) + 1e-3
                    #    overall_box_loss += ((s.mean(dim=1, keepdims=True) * resized_masks).flatten(start_dim=1).sum(1) / denom).mean()
                    overall_loss += s.mean()

                deep_loss.append(overall_loss)
                box_deep_loss.append(overall_box_loss)

            # collate the losses
            deep_loss = torch.stack(deep_loss).mean()
            box_deep_loss = torch.stack(box_deep_loss).mean()

        else:
            deep_loss = torch.zeros(1).to(orig_imgs.device).squeeze()
            box_deep_loss = torch.zeros(1).to(orig_imgs.device).squeeze()

        total_img_loss = ssim_loss * self.ssim_weight + l1_loss * self.l1_weight + \
                deep_loss * self.deep_loss_weight
        total_box_loss += self.deep_loss_weight * box_deep_loss

        total_loss = self.recon_loss_weight * ((1 - self.box_loss_weight) * total_img_loss + self.box_loss_weight * total_box_loss)

        loss_dict = dict(
                loss_reconst_overall=total_loss,
        )

        return loss_dict

    def boxes_to_mask(self, boxes, image_shape):
        fg_mask = torch.zeros([3, *image_shape]).to(boxes.device)
        for b in boxes:
            fg_mask[:, b[1]:b[3], b[0]:b[2]] = 1

        return fg_mask

    def get_gram_matrix(self, x, detach=False):
        # gram matrix per layer
        assert isinstance(x, torch.Tensor)
        b, ch, h, w = x.size()
        features = x.view(b, ch, h*w)
        gram = torch.bmm(features, features.transpose(1, 2)) / (ch*w*h)
        if detach:
            gram = gram.detach()

        return gram

    def deep_loss(self, reconstructed_imgs, orig_imgs):
        # prediction features at different depths of pretrained network
        features_pred = []

        # target features at different depths of pretrained network
        features_target = []

        # populate lists
        feats = None
        for b in self.deep_loss_backbones:
            b = b.to(orig_imgs.device)
            feat_pred, feat_target = feats if feats is not None else [reconstructed_imgs,
                    orig_imgs]

            # update feats
            feats = [b(feat_pred), b(feat_target)]

            # store in respective features list
            features_pred.append(feats[0])
            features_target.append(feats[1])

        # compute losses (no reduction, so it is pixel-wise)
        content_losses, style_losses = [], []

        if self.use_content:
            # compute content loss (direct comparison of features)
            for f, f_ in zip(features_target, features_pred):
                loss_content = torch.abs(f - f_) * self.perceptual_weight
                content_losses.append(loss_content)

        else:
            content_losses.append(None)

        if self.use_style:
            # compute style loss (comparison of features distribution)
            gram_target = [self.get_gram_matrix(l, detach=True) for l in features_target]
            gram_pred = [self.get_gram_matrix(l) for l in features_pred]
            for g, g_ in zip(gram_target, gram_pred):
                loss_style = torch.abs(g - g_) * (1 - self.perceptual_weight)
                style_losses.append(loss_style)

        else:
            style_losses.append(None)

        return content_losses, style_losses

    def train(self, mode=True):
        super(ReconstructionLoss, self).train(mode)

        # always set deep loss backbones to eval
        for b in self.deep_loss_backbones:
            for m in b.modules():
                m.eval()

    def crop_boxes(self, pred_imgs, gt_imgs, boxes, min_size=12):
        boxes_per_img = [len(b) for b in boxes]
        new_image_shape = pred_imgs.shape[-2:]

        # resize boxes based on reconstructed img size
        boxes = torch.cat(boxes).round()
        if boxes.shape[0] == 0:
            return [[]], [[]], boxes

        # pad small boxes
        small_box_inds_x = torch.nonzero((boxes[:, 2] - boxes[:, 0]) < min_size).flatten()
        small_box_inds_y = torch.nonzero((boxes[:, 3] - boxes[:, 1]) < min_size).flatten()
        if small_box_inds_x.shape[0] > 0:
            # pad to at least min_size x min_size box
            x1 = boxes[small_box_inds_x, 0]
            x2 = boxes[small_box_inds_x, 2]
            h_pad = min_size - (x2 - x1)
            new_x2 = torch.minimum(x2 + h_pad, torch.ones(1).to(boxes) * new_image_shape[1])
            extra_h_pad = min_size - (new_x2 - x1)
            new_x1 = torch.maximum(x1 - extra_h_pad, torch.zeros_like(x1))

            boxes[small_box_inds_x, 0] = new_x1
            boxes[small_box_inds_x, 2] = new_x2

        if small_box_inds_y.shape[0] > 0:
            # pad to at least min_sizexmin_size box
            y1 = boxes[small_box_inds_y, 1]
            y2 = boxes[small_box_inds_y, 3]
            v_pad = min_size - (y2 - y1)
            new_y2 = torch.minimum(y2 + v_pad, torch.ones(1).to(boxes) * new_image_shape[0])
            extra_v_pad = min_size - (new_y2 - y1)
            new_y1 = torch.maximum(y1 - extra_v_pad, torch.zeros_like(y1))

            boxes[small_box_inds_y, 1] = new_y1
            boxes[small_box_inds_y, 3] = new_y2

        # cast and split
        boxes = boxes.int().split(boxes_per_img)

        # get all gt and pred patches (List[List[Tensor]])
        all_pred_patches = [[img[:, b[1]:b[3], b[0]:b[2]] for b in boxes[ind]] \
                for ind, img in enumerate(pred_imgs)]
        all_gt_patches = [[img[:, b[1]:b[3], b[0]:b[2]] for b in boxes[ind]] \
                for ind, img in enumerate(gt_imgs)]

        return all_pred_patches, all_gt_patches, boxes
