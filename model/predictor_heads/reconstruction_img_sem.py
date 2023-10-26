from mmdet.registry import MODELS
from abc import ABCMeta
from mmdet.utils import ConfigType, OptConfigType, InstanceList, OptMultiConfig
from mmengine.model import BaseModule, constant_init
from mmengine.structures import BaseDataElement
from mmdet.structures import SampleList
from mmdet.structures.bbox import scale_boxes
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torchvision.transforms import functional as TF, InterpolationMode
from .modules.decoder import SPADEResnetBlock, CRNBlock
from .modules.layers import get_normalization_2d, get_activation

@MODELS.register_module()
class ReconstructionHead(BaseModule, metaclass=ABCMeta):
    """Reconstruction Head from object-centric features

    Args:
        img_decoder_cfg (ConfigType):  decoder
        img_aspect_ratio (tuple[int]): aspect ratio of reconstructed image
    """
    def __init__(self, decoder_cfg: ConfigType, aspect_ratio: Union[Tuple, List],
            obj_viz_feat_size: int, bottleneck_feat_size: int, num_classes: int,
            bool_visual: bool,
            bool_semantics: bool,
            bool_img: bool,
            use_seg_recon: bool = False, num_nodes: int = 8,
            use_pred_boxes_whiteout: bool = False, layout_noise_dim: int = 32,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.obj_viz_feat_size = obj_viz_feat_size
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.use_seg_recon = use_seg_recon
        self.img_decoder = MODELS.build(decoder_cfg)
        num_upsampling_stages = len(decoder_cfg.dims) - 1
        self.reconstruction_size = (Tensor(aspect_ratio) * (2 ** num_upsampling_stages)).int()
        self.bottleneck = torch.nn.Linear(obj_viz_feat_size, bottleneck_feat_size)
        self.img_conv = torch.nn.Sequential(
            torch.nn.Conv2d(4 + num_classes, layout_noise_dim, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(layout_noise_dim),
            torch.nn.ReLU(),
        )
        self.use_pred_boxes_whiteout = use_pred_boxes_whiteout
        self.bool_visual=bool_visual
        self.bool_semantics=bool_semantics
        self.bool_img=bool_img

    def predict(self, results: Tuple, feats: BaseDataElement, imgs: Tensor) -> Tensor:
        # first, rescale results
        results = self._rescale_results(results)

        # extract visual and semantic feats
        classes = [r.pred_instances.labels for r in results]
        boxes = [r.pred_instances.bboxes for r in results]
        viz_feats = feats.instance_feats
        semantic_feats = feats.semantic_feats
        img_feats = feats.bb_feats

        if 'gt_instances' in results[0]:
            gt_classes = [r.gt_instances.labels for r in results]
            gt_boxes = [r.gt_instances.bboxes for r in results]

        if 'masks' in results[0].pred_instances:
            masks = [r.pred_instances.masks for r in results]
            layouts = self._construct_layout(classes, boxes, masks)

            if 'gt_instances' in results[0]:
                gt_masks = [r.gt_instances.masks.to_tensor(masks[0].dtype, masks[0].device) for r in results]
                gt_layouts = self._construct_layout(gt_classes, gt_boxes, gt_masks)

        else:
            layouts = self._construct_layout(classes, boxes)
            if 'gt_instances' in results[0]:
                gt_layouts = self._construct_layout(gt_classes, gt_boxes)
            else:
                gt_layouts = None

        # node features is (f, s)
        node_features = torch.cat([viz_feats, semantic_feats], dim=-1) if semantic_feats is not None else viz_feats

        # build input to img decoder using input image, layout
        reconstruction_input = self._construct_reconstruction_input(imgs,
                node_features, layouts, gt_layouts, img_feats)

        # finally, reconstruct img
        reconstructed_imgs = self.img_decoder(reconstruction_input)

        # resize imgs to get image targets
        img_targets = TF.resize(imgs, reconstructed_imgs.shape[-2:])

        return reconstructed_imgs, img_targets, results

    def _construct_layout(self, classes, boxes, masks=None):
        box_layout = torch.zeros(len(boxes), self.num_nodes, *self.reconstruction_size).to(boxes[0].device)
        mask_layout = None

        # build box layout from boxes, classes (used with backgroundized img as input to reconstructor)
        for img_id, (label, box) in enumerate(zip(classes, boxes)):
            for instance_id, (l, b) in enumerate(zip(label, box.round().int())):
                box_layout[img_id][instance_id, b[1]:b[3], b[0]:b[2]] = l + 1 # labels are 0-indexed

        if masks is not None:
            mask_layouts = []
            for ind, (label, mask) in enumerate(zip(classes, masks)):
                mask_layouts.append(mask.int() * (label + 1).unsqueeze(-1).unsqueeze(-1))

            mask_layout = pad_sequence(mask_layouts, batch_first=True)

            #if self.use_seg_recon:
            #    if mask_layout.shape[1] == 0:
            #        # deal with no predicted instances
            #        one_hot_layout = torch.zeros(mask_layout.shape[0], 1, *mask_layout.shape[2:]).to(mask_layout.device)
            #    else:
            #        one_hot_layout = (mask_layout > 0).int() # stack of instance masks
            #else:
            #    one_hot_layout = (box_layout > 0).int() # stack of instance box masks

            # to construct layout, one hot encode mask_layout, sum across instance dim
            mask_ohl = F.one_hot(mask_layout, self.num_classes + 1)
            if mask_ohl.shape[1] == 0:
                layout = torch.zeros(*mask_ohl.transpose(1, -1).shape[:-1]).to(mask_ohl.device).float()
            else:
                layout = mask_ohl.transpose(1, -1).max(-1).values.float()

        else:
            box_ohl = F.one_hot(box_layout.long(), self.num_classes + 1)
            layout = box_ohl.transpose(1, -1).max(-1).values.float()

        # one hot layout is based on box layout
        one_hot_layout = (box_layout > 0).int() # stack of instance box masks

        box_ohl = F.one_hot(box_layout.long(), self.num_classes + 1)
        box_layout = box_ohl.transpose(1, -1).max(-1).values.float()

        return box_layout, layout, one_hot_layout

    def _construct_reconstruction_input(self, images, node_features, layouts,
            gt_layouts, img_feats):
        
        bool_visual = self.bool_visual
        bool_semantics = self.bool_semantics
        bool_img = self.bool_img
        device = torch.device("cuda:0")

        f = self.bottleneck(node_features[..., :self.obj_viz_feat_size])

        if bool_visual and bool_semantics:
            node_features = torch.cat([f, node_features[..., self.obj_viz_feat_size:]], -1)
            

        elif not(bool_visual) and bool_semantics:
            node_features = node_features[..., self.obj_viz_feat_size:]
            upsample_conv = nn.Conv2d(node_features.shape[-1], 320, kernel_size=1).to(device)
            if node_features.shape[0]!=0 or node_features.shape[1]!=0 or node_features.shape[2]!=0:
                qw=1
            else:
                node_features = upsample_conv(node_features.permute(2, 0, 1).unsqueeze(0)).permute(2, 3, 1, 0).squeeze()
        
            # if node_features.shape[0] != 16 or node_features.shape[1] != 16 or node_features.shape[2] != 256:
            #     # breakpoint()
            #     padding = (0, 256 - node_features.shape[2], 0, 16 - node_features.shape[1], 0, 16 - node_features.shape[0])
            #     node_features = torch.nn.functional.pad(node_features, padding, "constant", 0)
                
            # node_features = upsample_conv(node_features.permute(2, 0, 1).unsqueeze(0)).permute(2, 3, 1, 0).squeeze()

        
        elif bool_visual and not(bool_semantics):
            node_features = node_features[..., :self.obj_viz_feat_size]
            upsample_conv = nn.Conv2d(node_features.shape[-1], 320, kernel_size=1).to(device)
            if node_features.shape[0]!=0 or node_features.shape[1]!=0 or node_features.shape[2]!=0:
                qw=1
            else:
                node_features = upsample_conv(node_features.permute(2, 0, 1).unsqueeze(0)).permute(2, 3, 1, 0).squeeze()
        
        else:
            node_features=[]
            print("ERROR in else")

        

        # convert one-hot layout over proposals to class-wise segmentation mask "layout"
        _, _, one_hot_layout = layouts
        denom = one_hot_layout.sum(1)
        denom[denom == 0] = 1
        feature_layout = torch.sum(one_hot_layout.unsqueeze(2)[:, :node_features.shape[1]] * \
                node_features.unsqueeze(-1).unsqueeze(-1), dim=1) / denom.unsqueeze(1)

        
        # TODO(adit98) add img features here to feature layout here
        feats1 = img_feats[3]
        new_height, new_width = 64, 96
        output_tensor_2048 = F.interpolate(feats1, size=(new_height, new_width), mode='bilinear', align_corners=False)# Upsample using bilinear interpolation, size is now (_, 2048, 64,96) from (_, 2048, 7, 13)
        downsample_conv = nn.Conv2d(2048, 320, kernel_size=1).to(device)# Define a convolutional layer to reduce the number of channels to 256
        img_feats_320 = downsample_conv(output_tensor_2048) # size is now (_, 320, 64, 96) from (_, 2048, 64,96)


        # white out GT img using gt_layout, add predicted scene layout, process with convolution
        processed_bg_img = self._whiteout(images, gt_layouts, layouts)

        if bool_visual and bool_semantics and bool_img: #ALL together, new recon objective
            input_feat_672 = torch.cat([feature_layout, processed_bg_img, img_feats_320], dim=1) #input feats of size (_, 672, 64, 96)
            downsample_conv1 = nn.Conv2d(input_feat_672.shape[1], 352, kernel_size=1).to(device)
            input_feat_352 = downsample_conv1(input_feat_672) # size is now (_, 352, 64, 96) from (_, 672, 64, 96)
            print('condition 1')
            return input_feat_352
        
        elif bool_visual and bool_semantics and not(bool_img): #ORIGINAL RECON OBJECTIVE
            input_feat_352 = torch.cat([feature_layout, processed_bg_img], dim=1) #input feats of size (_, 672, 64, 96)
            print('condition 2')
            return input_feat_352
        
        elif bool_visual and not(bool_semantics) and bool_img: #ONLY visual features (graph and image), no semantics
            input_feat_672 = torch.cat([feature_layout, processed_bg_img, img_feats_320], dim=1) #input feats of size (_, 672, 64, 96)
            downsample_conv1 = nn.Conv2d(input_feat_672.shape[1], 352, kernel_size=1).to(device)
            input_feat_352 = downsample_conv1(input_feat_672) # size is now (_, 352, 64, 96) from (_, 672, 64, 96)
            print('condition 3')
            return input_feat_352
        
        elif not(bool_visual) and bool_semantics and bool_img: #ONLY graph semantic and image features, no graph visual features
            
            # if feature_layout.shape[0]!=16 or processed_bg_img.shape[0]!=16 or img_feats_320.shape[0]!=16:
            #     breakpoint()
            input_feat_672 = torch.cat([feature_layout, processed_bg_img, img_feats_320], dim=1) #input feats of size (_, 672, 64, 96)
            downsample_conv1 = nn.Conv2d(input_feat_672.shape[1], 352, kernel_size=1).to(device)
            input_feat_352 = downsample_conv1(input_feat_672) # size is now (_, 352, 64, 96) from (_, 672, 64, 96)
            print('condition 4')
            return input_feat_352
        
        else:
            return []

        # combine processed_bg_img, layout_with_features, semantics
        input_feat = torch.cat([feature_layout, processed_bg_img], dim=1)

        return []

    def _whiteout(self, images, gt_layouts, layouts):
        # resize GT
        whited_out_images = TF.resize(images, self.reconstruction_size.tolist()).nan_to_num()

        # extract layouts (if no gt, use predicted to whiteout img)
        if gt_layouts is not None and not self.use_pred_boxes_whiteout:
            gt_box_layout, _, _ = gt_layouts
        else:
            gt_box_layout, _, _ = layouts

        box_layout, layout, _ = layouts

        # use gt_box_layout to white out foreground
        whited_out_images *= (gt_box_layout.sum(1) <= 1).int().unsqueeze(1) # multiply by bg inds [1, 0, ..., 0]

        # finally concat with predicted layout
        if self.use_seg_recon:
            processed_bg_img = self.img_conv(torch.cat([whited_out_images,
                layout], dim=1))
        else:
            processed_bg_img = self.img_conv(torch.cat([whited_out_images,
                box_layout], dim=1))

        return processed_bg_img

    def _rescale_results(self, results: SampleList) -> SampleList:
        for r in results:
            # rescale boxes
            pred_actual_size = list(r.ori_shape)
            pred_scale_factor = (self.reconstruction_size / Tensor(pred_actual_size)).flip(0).tolist()
            r.pred_instances.bboxes = scale_boxes(r.pred_instances.bboxes, pred_scale_factor)

            gt_actual_size = [r.gt_instances.masks.height, r.gt_instances.masks.width]
            gt_scale_factor = (self.reconstruction_size / Tensor(gt_actual_size)).flip(0).tolist()
            r.gt_instances.bboxes = scale_boxes(r.gt_instances.bboxes, gt_scale_factor)

            if 'masks' in r.pred_instances:
                if r.pred_instances.masks.shape[0] == 0:
                    r.pred_instances.masks = torch.zeros(0, *self.reconstruction_size).to(r.pred_instances.masks.device)
                else:
                    r.pred_instances.masks = TF.resize(r.pred_instances.masks,
                            self.reconstruction_size.tolist(), InterpolationMode.NEAREST)

                r.gt_instances.masks = r.gt_instances.masks.resize(self.reconstruction_size.tolist())

        return results

@MODELS.register_module()
class DecoderNetwork(torch.nn.Module):
    """
    Decoder Network that generates a target image from a pair of masked source image and layout
    Implemented in two options: with a CRN block or a SPADE block
    """
    def __init__(self, dims, normalization='instance', activation='leakyrelu', spade_blocks=False, source_image_dims=32):
        super(DecoderNetwork, self).__init__()

        self.spade_block = spade_blocks
        self.source_image_dims = source_image_dims

        layout_dim = dims[0]
        self.decoder_modules = torch.nn.ModuleList()
        for i in range(1, len(dims)):
            input_dim = 1 if i == 1 else dims[i - 1]
            output_dim = dims[i]

            if self.spade_block:
                # Resnet SPADE block
                mod = SPADEResnetBlock(input_dim, output_dim, layout_dim-self.source_image_dims, self.source_image_dims)

            else:
                # CRN block
                mod = CRNBlock(layout_dim, input_dim, output_dim,
                                             normalization=normalization, activation=activation)

            self.decoder_modules.append(mod)

        output_conv_layers = [
            torch.nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
            get_activation(activation),
            torch.nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
        ]
        torch.nn.init.kaiming_normal_(output_conv_layers[0].weight)
        torch.nn.init.kaiming_normal_(output_conv_layers[2].weight)
        self.output_conv = torch.nn.Sequential(*output_conv_layers)

    def forward(self, layout):
        """
        Output will have same size as layout
        """
        # H, W = self.output_size
        N, _, H, W = layout.size()
        self.layout = layout

        # Figure out size of input
        input_H, input_W = H, W
        for _ in range(len(self.decoder_modules)):
            input_H //= 2
            input_W //= 2

        assert input_H != 0
        assert input_W != 0

        feats = torch.zeros(N, 1, input_H, input_W).to(layout)
        for mod in self.decoder_modules:
            feats = F.upsample(feats, scale_factor=2, mode='nearest')
            feats = mod(layout, feats)

        out = self.output_conv(feats)

        return out


