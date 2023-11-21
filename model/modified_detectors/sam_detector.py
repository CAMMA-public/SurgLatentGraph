from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList
from mmdet.structures.bbox import scale_boxes
from mmdet.models import BaseDetector
from mmengine.structures import InstanceData
import random
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import Tuple, Dict, List
from torchvision.ops.boxes import batched_nms, box_area
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import MaskData, generate_crop_boxes, batched_mask_to_box, \
        box_xyxy_to_xywh, is_box_near_crop_edge, remove_small_regions, calculate_stability_score, \
        build_all_layer_point_grids
from cuml import UMAP
from cuml.common.device_selection import using_device_type
import numpy as np
import pickle

@MODELS.register_module()
class SAMDetector(BaseDetector):
    def __init__(self, sam_type: str = 'vit_h', sam_weights_path: str = None,
            num_classes: int = 7, num_nodes: int = 16, cluster_info_path: str = None,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            **kwargs):

        super().__init__(**kwargs)

        build_sam = sam_model_registry[sam_type]
        self.sam_model = build_sam(checkpoint=sam_weights_path, img_size=400,
                pixel_mean=pixel_mean, pixel_std=pixel_std)

        if sam_weights_path is None:
            print("WARNING: NOT LOADING PRE-TRAINED WEIGHTS FOR SAM")

        # num classes, nodes
        self.num_classes = num_classes
        self.num_nodes = num_nodes

        if cluster_info_path is not None:
            with open(cluster_info_path, 'rb') as f:
                self.cluster_info = pickle.load(f)
                with using_device_type("GPU"):
                    self.umap_estimator = self.cluster_info.pop('umap_estimator')

        else:
            self.cluster_info = None

        # init point grids for auto-prompting
        self.point_grids = build_all_layer_point_grids(
            16, # points per side
            0, # crop_n_layers
            1, # crop_n_points_downscale_factor
        )
        self.box_nms_thresh = 0.7
        self.stability_score_thresh = 0.7
        self.stability_score_offset = 1.0

        # init transform
        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def extract_feat(self, batch_inputs: Tensor, batch_data_samples: SampleList,
            compute_instance_feats: bool = False, selected_inds: List[Tensor] = None,
            final_filter_key: bool = 'iou_preds') -> Tuple[Tensor]:
        # preprocess raw images for sam
        resized_inputs, sam_input_size = self.resize_to_sam(batch_inputs)
        sam_inputs = self.sam_model.preprocess(resized_inputs)

        if compute_instance_feats:
            # extract instance feats
            batch_outputs = self.generate_and_decode_prompts(sam_inputs, final_filter_key,
                    filter_preds=False, predict_mask=False)

            # filter based on selected inds of frozen detector
            assert selected_inds is not None

            # get img feats, remove from MaskData, and filter MaskData
            img_feats = []
            for b, s in zip(batch_outputs, selected_inds):
                img_feats.append(b['img_feats'])
                b.__delitem__('img_feats')
                b.filter(s)

            img_feats = torch.cat(img_feats)
            instance_feats = [b['feats'] for b in batch_outputs]

        else:
            # forward pass
            img_feats = self.sam_model.image_encoder(sam_inputs)

            instance_feats = None

        return (img_feats,), instance_feats

    def resize_to_sam(self, imgs: Tensor):
        sam_img_width = self.sam_model.image_encoder.img_size
        scale_factor = sam_img_width / imgs.shape[-1]
        sam_input_size = (int(scale_factor * imgs.shape[-2]), sam_img_width)
        sam_inputs = TF.resize(imgs, sam_input_size)

        return sam_inputs, sam_input_size

    def generate_and_decode_prompts(self, batch_inputs: Tensor, final_filter_key: str,
            return_logits: bool = True, multimask_output: bool = False,
            filter_preds: bool = True, predict_mask: bool = True) -> MaskData:
        # preprocess raw images for sam
        resized_inputs, sam_input_size = self.resize_to_sam(batch_inputs)
        sam_inputs = self.sam_model.preprocess(resized_inputs)

        # forward pass
        img_feats = self.sam_model.image_encoder(sam_inputs)

        # generate point prompts
        points_scale = np.array(sam_inputs.shape[-2:])[None, ::-1]
        init_points = self.point_grids[0] * points_scale
        transformed_points = self.transform.apply_coords(init_points,
                sam_inputs.shape[-2:])
        in_points = torch.as_tensor(transformed_points, device=img_feats.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        points = (in_points[:, None, :], in_labels[:, None])

        outputs = []
        for img, curr_embedding in zip(sam_inputs, img_feats):
            data = MaskData()
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )

            if predict_mask:
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                # TODO(adit98) refine masks if needed here and switch to dense embeddings

                masks = self.sam_model.postprocess_masks(
                    low_res_masks,
                    input_size=sam_input_size,
                    original_size=batch_inputs.shape[-2:],
                )

                thresholded_masks = masks > self.sam_model.mask_threshold
                boxes = batched_mask_to_box(thresholded_masks)

                data['boxes'] = boxes.flatten(0, 1)
                data['masks'] = thresholded_masks.flatten(0, 1)
                data['iou_preds'] = iou_predictions.flatten(0, 1)

                # calculate stability score
                data["stability_score"] = calculate_stability_score(masks.flatten(0, 1),
                        self.sam_model.mask_threshold, self.stability_score_offset).nan_to_num()

            # add preds to data
            if multimask_output: # 3 copies of each embedding
                data['feats'] = sparse_embeddings.unsqueeze(1).repeat(
                        1, 3, 1, 1).flatten(0, 1).sum(1)#[:, 0]#.sum(1)
            else:
                data['feats'] = sparse_embeddings.sum(1)#[:, 0]#.sum(1)

            # keep track of which mask instances we are keeping (since point grid is fixed can sync across SAM models with different weights)
            data['instance_ids'] = torch.arange(data['feats'].shape[0])

            if filter_preds:
                # filter by stability score
                if self.stability_score_thresh > 0.0:
                    keep_mask = data["stability_score"] >= self.stability_score_thresh
                    data.filter(keep_mask)

                # filter out small boxes (width or height < 10)
                widths = data['boxes'][:, 2] - data['boxes'][:, 0]
                heights = data['boxes'][:, 3] - data['boxes'][:, 1]
                boxes_to_keep = torch.logical_and((widths >= 10), (heights >= 10))
                data.filter(boxes_to_keep)

            # add img feats to data
            data['img_feats'] = curr_embedding.unsqueeze(0)

            outputs.append(data)

        return outputs

    def classify_instances(self, batch_outputs: MaskData) -> MaskData:
        if self.cluster_info is not None:
            # pass feats through umap
            for b in batch_outputs:
                if b['feats'].shape[0] == 0:
                    umap_feats = torch.zeros(b['feats'].shape[0], self.umap_estimator.n_components).to(b['feats'].device)
                else:
                    with using_device_type("GPU"):
                        umap_feats = torch.from_numpy(self.umap_estimator.transform(b['feats'].detach().cpu().numpy())).to(b['feats'].device)

                # compute distance to each cluster center
                arr_a = umap_feats.unsqueeze(1)
                arr_b = torch.from_numpy(self.cluster_info['centers']).to(umap_feats.device).unsqueeze(0)
                similarities = ((arr_a * arr_b).sum(-1) / torch.matmul(torch.linalg.norm(arr_a, dim=-1),
                    torch.linalg.norm(arr_b, dim=-1)) + 1) / 2

                # keep track of fg clusters
                fg_clusters = (torch.ones(similarities.shape[1]) * \
                        (torch.tensor(self.cluster_info['labels']) > 0).float()).to(similarities.device)

                # assign label as highest scoring fg cluster
                cluster_id = (similarities * fg_clusters).argmax(-1)
                b['labels'] = torch.tensor(self.cluster_info['labels']).to(cluster_id.device)[cluster_id] - 1

                # compute scores
                T = 0.05
                scores = torch.softmax(similarities / T, dim=-1).gather(-1,
                        cluster_id.unsqueeze(-1).long()).squeeze(-1)
                fg_scores = (cluster_id == similarities.argmax(-1)).float()

                # TODO(adit98) see if we need to incorporate fg_scores
                b['scores'] = scores.to(b['boxes'].device)
                b['labels'] = b['labels'].to(b['boxes'].device)

        else:
            for b in batch_outputs:
                b['labels'] = Tensor([random.randint(0, self.num_classes - 1) \
                        for i in range(len(b['boxes']))]).long().to(b['boxes'].device)

        return batch_outputs

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        raise NotImplementedError

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, final_filter_key: str = 'iou_preds'):
        # detect objects, get feats
        batch_outputs = self.generate_and_decode_prompts(batch_inputs, final_filter_key)

        # use feats to classify each detection
        batch_outputs = self.classify_instances(batch_outputs)

        # filter, resize, convert to mmdet format, add to results
        for b, r in zip(batch_outputs, batch_data_samples):
            # extract img feats
            img_feats = b['img_feats']
            b.__delitem__('img_feats')

            # figure out which key to filter by
            keys = [x[0] for x in b.items()]
            score_key = 'scores' if 'scores' in keys else final_filter_key

            # nms to filter preds
            keep_by_nms = batched_nms(
                b["boxes"].float(),
                b[score_key],
                torch.zeros_like(b["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            b.filter(keep_by_nms)

            # keep top preds based on score
            num_to_keep = min(self.num_nodes, b[score_key].shape[0])
            keep_by_score = torch.topk(b[score_key], num_to_keep).indices
            b.filter(keep_by_score)

            # get feats, boxes, masks, labels from batch_outputs
            feats = b['feats']
            boxes = b['boxes']
            masks = b['masks']
            labels = b['labels']
            instance_ids = b['instance_ids']
            scores = b[score_key]

            # resize boxes and masks
            scale_factor = tuple((Tensor(r.ori_shape) / Tensor(list(b['masks'].shape[-2:]))).flip(0).tolist())
            final_bboxes = scale_boxes(boxes.float(), scale_factor).long()
            if masks.shape[0] == 0:
                final_masks = torch.zeros(0, *r.ori_shape).to(masks)
            else:
                final_masks = TF.resize(masks, r.ori_shape, InterpolationMode.NEAREST)

            r.pred_instances = InstanceData(bboxes=final_bboxes, masks=final_masks,
                    labels=labels, feats=feats, scores=scores, instance_ids=instance_ids)
            r.img_feats = img_feats

        return batch_data_samples

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        raise NotImplementedError

    def to(self, *args, **kwargs) -> torch.nn.Module:
        # device
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        self.sam_model.to(device)

        return super().to(*args, **kwargs)
