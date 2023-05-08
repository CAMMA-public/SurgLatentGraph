from mmdet.registry import METRICS
from mmdet.evaluation.metrics import CocoMetric
from mmdet.structures.bbox import scale_boxes
from mmengine.logging import MMLogger
from typing import Dict, Sequence
from torchmetrics.functional import multiscale_structural_similarity_index_measure as ms_ssim, structural_similarity_index_measure as ssim
import torch
from torch import Tensor
import torchvision.transforms.functional as TF
from torchmetrics import AveragePrecision as AP, Precision, Recall, F1Score
import os
import cv2
import numpy as np

@METRICS.register_module()
class CocoMetricRGD(CocoMetric):
    def __init__(self, data_root, data_prefix, use_pred_boxes_recon, additional_metrics=[], **kwargs):
        super().__init__(**kwargs)
        self.ssim_roi = SSIM_RoI(data_range=1, size_average=True, channel=3)
        self.data_root = data_root
        self.data_prefix = data_prefix
        self.use_pred_boxes_recon = use_pred_boxes_recon
        self.additional_metrics = additional_metrics

    def process(self, data_batch: Dict, data_samples: Sequence[dict]) -> None:
        if len(self.metrics) > 0:
            super().process(data_batch, data_samples)
        else:
            for data_sample in data_samples:
                result = dict()
                result['img_id'] = data_sample['img_id']

                # parse gt
                gt = dict()
                gt['width'] = data_sample['ori_shape'][1]
                gt['height'] = data_sample['ori_shape'][0]
                gt['img_id'] = data_sample['img_id']
                if self._coco_api is None:
                    # TODO: Need to refactor to support LoadAnnotations
                    assert 'instances' in data_sample, \
                        'ground truth is required for evaluation when ' \
                        '`ann_file` is not provided'
                    gt['anns'] = data_sample['instances']
                # add converted result to the results list
                self.results.append((gt, result))

        gts, preds = list(map(list, zip(*self.results[-1 * len(data_samples):])))
        for p, g, data_sample in zip(preds, gts, data_samples):
            if 'reconstruction' in data_sample:
                p['reconstruction'] = data_sample['reconstruction']

            if 'pred_ds' in data_sample:
                p['ds'] = data_sample['pred_ds']
                g['ds'] = data_sample['ds']

        self.results[-1 * len(data_samples):] = zip(gts, preds)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if len(self.metrics) > 0:
            eval_results = super().compute_metrics(results)
        else:
            eval_results = {}

        # init logger
        logger: MMLogger = MMLogger.get_current_instance()
        logger_info = []

        # load data
        img_paths = self._coco_api.load_imgs(self.img_ids)
        gts, preds = list(map(list, zip(*results)))

        if self.outfile_prefix is not None:
            result_files = self.results2json(preds, self.outfile_prefix, gts=gts)

        # compute reconstruction metrics
        if 'reconstruction' in preds[0] and 'reconstruction' in self.additional_metrics:
            # read img, convert to rgb, 0-1 normalize
            gt_img_paths = [os.path.join(self.data_root, self.data_prefix, i['file_name']) for i in img_paths]

            # get boxes
            if self.use_pred_boxes_recon:
                recon_boxes = [Tensor(r[1]['bboxes']) for r in results]
            else:
                ann_ids_by_img = [self._coco_api.get_ann_ids(i) for i in self.img_ids]
                ann_info_by_img = [[self._coco_api.load_anns(a_id)[0] for a_id in i_id] for i_id in ann_ids_by_img]
                recon_boxes = [Tensor([g['bbox'] for g in a]) for a in ann_info_by_img]

            orig_img_shape = [gts[0]['height'], gts[0]['width']]

            ssim_vals = []
            for pred, gt_path, b in zip(preds, gt_img_paths, recon_boxes):
                # load
                g = torch.from_numpy(cv2.imread(gt_path)).permute(2, 0, 1).flip(0) / 255

                # resize pred img to gt_img size
                p = TF.resize(pred['reconstruction'].detach(), g.shape[-2:])

                # resize gt img to pred img size
                #g = TF.resize(g, p.shape[-2:])

                # compute img ssim
                ssim_vals.append(ssim(p.unsqueeze(0), g.unsqueeze(0)))

                # compute ssim roi

                # rescale boxes
                #sf = (Tensor(list(p.shape[-2:])) / Tensor(orig_img_shape)).flip(0)
                #if b.shape[0] != 0:
                #    scaled_b = scale_boxes(b, sf).round()
                #else:
                #    scaled_b = b
                self.ssim_roi.update(p.unsqueeze(0), g.unsqueeze(0), [b])

            ssim_val = torch.mean(torch.stack(ssim_vals)).detach().cpu().item()
            ssim_roi = self.ssim_roi.compute().cpu().item()
            self.ssim_roi.reset()
            logger_info += [f'ssim: {ssim_val:.4f}', f'ssim_roi: {ssim_roi:.4f}']
            eval_results['ssim'] = ssim_val
            eval_results['ssim_roi'] = ssim_roi

        # TODO compute graph metrics

        # compute DS metrics
        # TODO(adit98) add other metrics besides AP
        if 'ds' in preds[0]:
            if preds[0]['ds'].ndim > 1:
                ds_preds = torch.stack([p['ds'] for p in preds]).max(-1).indices
                ds_gt = torch.stack([Tensor(g['ds']) for g in gts])

                # compute precision, recall, f1
                torch_prec = Precision(task='multiclass', average='macro', num_classes=3)
                torch_rec = Recall(task='multiclass', average='macro', num_classes=3)
                torch_f1 = F1Score(task='multiclass', average='macro', num_classes=3)

                ds_prec = np.mean([torch_prec(ds_preds[:, i], ds_gt[:, i]) for i in range(ds_gt.shape[-1])])
                ds_rec = np.mean([torch_rec(ds_preds[:, i], ds_gt[:, i]) for i in range(ds_gt.shape[-1])])
                ds_f1 = np.mean([torch_f1(ds_preds[:, i], ds_gt[:, i]) for i in range(ds_gt.shape[-1])])

                # log
                logger_info.append(f'ds_precision: {ds_prec:.4f}')
                logger_info.append(f'ds_recall: {ds_rec:.4f}')
                logger_info.append(f'ds_f1: {ds_f1:.4f}')
                eval_results['ds_precision'] = ds_prec
                eval_results['ds_recall'] = ds_rec
                eval_results['ds_f1'] = ds_f1

            else:
                torch_ap = AP(task='multilabel', num_labels=3)
                ds_preds = torch.stack([p['ds'] for p in preds]).sigmoid()
                ds_gt = torch.stack([Tensor(g['ds']).round() for g in gts]).long()
                ds_ap = torch_ap(ds_preds, ds_gt)

                # log
                logger_info.append(f'ds_average_precision: {ds_ap:.4f}')
                eval_results['ds_average_precision'] = ds_ap

        logger.info(' '.join(logger_info))
        return eval_results

    def results2json(self, results: Sequence[dict], outfile_prefix: str, gts: Sequence[dict] = None) -> dict:
        if len(self.metrics) > 0:
            result_files = super().results2json(results, outfile_prefix)
        else:
            result_files = {}

        if 'reconstruction' in results[0]:
            recon_imgs = []
            img_ids = []
            for idx, result in enumerate(results):
                img_ids.append(result['img_id'])
                recon_imgs.append(result['reconstruction'])

            # save recon imgs
            if not os.path.exists(os.path.join(outfile_prefix, 'reconstructions')):
                os.makedirs(os.path.join(outfile_prefix, 'reconstructions'))

            for img_id, r in zip(img_ids, recon_imgs):
                # resize img
                r = TF.resize(r, (480, 854))
                cv2_img = (r.flip(0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8) # BGR, 0-255
                outname = os.path.join(outfile_prefix, 'reconstructions', str(img_id) + '.jpg')
                cv2.imwrite(outname, cv2_img)

            result_files['reconstruction'] = os.path.join(outfile_prefix, 'reconstructions')

        if 'ds' in results[0]:
            # save ds preds
            pred_ds = torch.stack([r['ds'] for r in results]).sigmoid()

            if not os.path.exists(outfile_prefix):
                os.makedirs(outfile_prefix)

            if pred_ds.ndim > 2:
                pred_outname = os.path.join(outfile_prefix, 'pred_ds.npy')
                np.save(pred_outname, pred_ds.detach().cpu().numpy())
            else:
                pred_outname = os.path.join(outfile_prefix, 'pred_ds.txt')
                np.savetxt(pred_outname, pred_ds.detach().cpu().numpy())

            if gts is not None:
                gt_ds = np.stack([g['ds'] for g in gts])
                gt_outname = os.path.join(outfile_prefix, 'gt_ds.txt')
                np.savetxt(gt_outname, gt_ds)

        return result_files

class SSIM_RoI:
    def __init__(self, data_range, size_average, channel):
        self.running_vals = []

    def __call__(self, pred_imgs, gt_imgs, gt_boxes):
        all_pred_patches, all_gt_patches, fixed_boxes = self.crop_boxes(pred_imgs, gt_imgs,
                gt_boxes)

        # compute ssim b/w each pred patch and gt patch
        ssim_vals = []
        for pred_patches, gt_patches in zip(all_pred_patches, all_gt_patches):
            for p, g in zip(pred_patches, gt_patches):
                p_mod = p.unsqueeze(0)
                g_mod = g.unsqueeze(0)
                ssim_val = ssim(p_mod, g_mod)
                ssim_vals.append(ssim_val)

        return torch.mean(Tensor(ssim_vals))

    def update(self, pred_imgs, gt_imgs, gt_boxes):
        v = self.__call__(pred_imgs, gt_imgs, gt_boxes)
        self.running_vals.append(v.detach())

    def compute(self):
        return torch.nanmean(torch.stack(self.running_vals))

    def reset(self):
        self.running_vals = []

    def crop_boxes(self, pred_imgs, gt_imgs, boxes, min_size=12):
        boxes_per_img = [len(b) for b in boxes]
        new_image_shape = pred_imgs.shape[-2:]

        # resize boxes based on reconstructed img size
        boxes = torch.cat(boxes).round()
        if boxes.shape[0] == 0:
            return [[]], [[]], boxes

        # pad small boxes
        small_box_inds_x = torch.nonzero(boxes[:, 2] < min_size).flatten()
        small_box_inds_y = torch.nonzero(boxes[:, 3] < min_size).flatten()
        if small_box_inds_x.shape[0] > 0:
            # pad to at least min_size x min_size box
            x1 = boxes[small_box_inds_x, 0]
            x2 = x1 + boxes[small_box_inds_x, 2]
            h_pad = min_size - (x2 - x1)
            new_x2 = torch.minimum(x2 + h_pad, torch.ones(1).to(boxes) * new_image_shape[1])
            extra_h_pad = min_size - (new_x2 - x1)
            new_x1 = torch.maximum(x1 - extra_h_pad, torch.zeros_like(x1))

            boxes[small_box_inds_x, 0] = new_x1
            boxes[small_box_inds_x, 2] = new_x2 - new_x1

        if small_box_inds_y.shape[0] > 0:
            # pad to at least min_sizexmin_size box
            y1 = boxes[small_box_inds_y, 1]
            y2 = y1 + boxes[small_box_inds_y, 3]
            v_pad = min_size - (y2 - y1)
            new_y2 = torch.minimum(y2 + v_pad, torch.ones(1).to(boxes) * new_image_shape[0])
            extra_v_pad = min_size - (new_y2 - y1)
            new_y1 = torch.maximum(y1 - extra_v_pad, torch.zeros_like(y1))

            boxes[small_box_inds_y, 1] = new_y1
            boxes[small_box_inds_y, 3] = new_y2 - new_y1

        # cast and split
        boxes = boxes.int().split(boxes_per_img)

        # get all gt and pred patches (List[List[Tensor]])
        all_pred_patches = [[img[:, b[1]:b[1] + b[3], b[0]:b[0] + b[2]] for b in boxes[ind]] \
                for ind, img in enumerate(pred_imgs)]
        all_gt_patches = [[img[:, b[1]:b[1] + b[3], b[0]:b[0] + b[2]] for b in boxes[ind]] \
                for ind, img in enumerate(gt_imgs)]

        return all_pred_patches, all_gt_patches, boxes
