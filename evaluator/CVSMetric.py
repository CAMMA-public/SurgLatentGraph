from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from typing import Dict, Optional, Sequence
import torch
import os
from torchmetrics import AveragePrecision as AP, MeanAbsoluteError as MAE, MeanSquaredError as MSE, F1Score
import numpy as np

@METRICS.register_module()
class CVSMetric(BaseMetric):
    """
    Metrics for CVS prediction (mAP, F1, brier score with inter-rater disagreement)
    """
    def __init__(self, num_classes: int = 3, agg: str = 'per_frame',
            outfile_prefix: str = 'results', ds_per_class: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agg = agg
        self.num_classes = num_classes
        self.outfile_prefix = outfile_prefix
        self.ds_per_class = ds_per_class
        self.cvs_preds = []
        self.cvs_gt = []
        self.rounded_cvs_gt = []

    def evaluate(self, size) -> dict:
        metrics = super().evaluate(size)
        self.results2json()

        return metrics

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            cvs_pred = data_sample['pred_ds'].sigmoid()
            cvs_gt = torch.from_numpy(data_sample['ds']).to(cvs_pred.device)
            self.cvs_preds.append(cvs_pred)
            self.cvs_gt.append(cvs_gt)
            self.rounded_cvs_gt.append(cvs_gt.round().long())

    def compute_metrics(self, results: list) -> dict:
        # init logger
        logger: MMLogger = MMLogger.get_current_instance()
        logger_info = []

        # init results dict
        eval_results = {}

        ds_preds = torch.stack(self.cvs_preds)
        ds_gt = torch.stack(self.cvs_gt)
        rounded_ds_gt = torch.stack(self.rounded_cvs_gt)

        # AP
        torch_ap = AP(task='multilabel', num_labels=self.num_classes, average='none').to(ds_preds.device)
        torch_f1 = F1Score(task='multilabel', num_labels=self.num_classes, average='none').to(ds_preds.device)
        torch_mse = MSE(num_outputs=self.num_classes).to(ds_preds.device)

        if 'per_video' in self.agg:
            # split ds_preds, gt by video
            _, vid_lengths = torch.unique_consecutive(vid_ids, return_counts=True)
            ds_preds_per_vid = ds_preds.split(vid_lengths.tolist())
            ds_gt_per_vid = ds_gt.split(vid_lengths.tolist())
            rounded_ds_gt_per_vid = rounded_ds_gt.split(vid_lengths.tolist())
            aps = []
            mses = []
            f1s = []
            for p, gt, r_gt in zip(ds_preds_per_vid, ds_gt_per_vid, rounded_ds_gt_per_vid):
                # use rounded gt for ap, unrounded for MAE and MSE
                aps.append(torch_ap(p, r_gt))
                mses.append(torch_mse(p, gt))
                f1s.append(torch_f1(p, r_gt))

            ds_vid_ap_per_class = torch.stack(aps).nanmean(0)
            ds_vid_mse_per_class = torch.stack(mses).nanmean(0)
            ds_vid_f1_per_class = torch.stack(f1s).nanmean(0)
            if self.ds_per_class:
                for ind, (ap_i, mse_i, f1_i) in enumerate(zip(ds_vid_ap_per_class, ds_vid_mse_per_class, ds_vid_mse_per_class)):
                    logger_info.append(f'ds_vid_ap_C{ind+1}: {ap_i:.4f}')
                    logger_info.append(f'ds_vid_mse_C{ind+1}: {mse_i:.4f}')
                    logger_info.append(f'ds_vid_f1_C{ind+1}: {f1_i:.4f}')

                    eval_results[f'ds_vid_ap_C{ind+1}'] = ap_i
                    eval_results[f'ds_vid_mse_C{ind+1}'] = mse_i
                    eval_results[f'ds_vid_f1_C{ind+1}'] = f1_i

            if 'per_class' in self.agg:
                ds_vid_ap = ds_vid_ap_per_class.nanmean()
                ds_vid_mse = ds_vid_mse_per_class.nanmean()
                ds_vid_f1 = ds_vid_f1_per_class.nanmean()

                ds_vid_ap_std = ds_vid_ap_per_class[~ds_vid_ap_per_class.isnan()].std()
                ds_vid_mse_std = ds_vid_mse_per_class[~ds_vid_mse_per_class.isnan()].std()
                ds_vid_f1_std = ds_vid_f1_per_class[~ds_vid_f1_per_class.isnan()].std()

            else:
                ds_per_vid_ap = torch.stack(aps).nanmean(1)
                ds_per_vid_mse = torch.stack(mses).nanmean(1)
                ds_per_vid_f1 = torch.stack(f1s).nanmean(1)

                ds_vid_ap = ds_per_vid_ap.nanmean()
                ds_vid_mse = ds_per_vid_mse.nanmean()
                ds_vid_f1 = ds_per_vid_f1.nanmean()

                ds_vid_ap_std = ds_per_vid_ap[~ds_per_vid_ap.isnan()].std()
                ds_vid_mse_std = ds_per_vid_mse[~ds_per_vid_mse.isnan()].std()
                ds_vid_f1_std = ds_per_vid_f1[~ds_per_vid_f1.isnan()].std()

            logger_info.append(f'ds_vid_ap: {ds_vid_ap} +- {ds_vid_ap_std}')
            logger_info.append(f'ds_vid_mse: {ds_vid_mse} +- {ds_vid_mse_std}')
            logger_info.append(f'ds_vid_f1: {ds_vid_f1} +- {ds_vid_f1_std}')

            eval_results['ds_video_average_precision'] = ds_vid_ap
            eval_results['ds_video_brier_score'] = ds_vid_mse
            eval_results['ds_video_f1'] = ds_vid_f1

            eval_results['ds_video_average_precision_std'] = ds_vid_ap_std
            eval_results['ds_video_brier_score_std'] = ds_vid_mse_std
            eval_results['ds_video_f1_std'] = ds_vid_f1_std

        else:
            ds_ap = torch_ap(ds_preds, rounded_ds_gt)
            ds_mse = torch_mse(ds_preds, ds_gt)
            ds_f1 = torch_f1(ds_preds, rounded_ds_gt)

            # log overall
            logger_info.append(f'ds_average_precision: {torch.nanmean(ds_ap):.4f}')
            logger_info.append(f'ds_brier_score: {torch.nanmean(ds_mse):.4f}')
            logger_info.append(f'ds_f1: {torch.nanmean(ds_f1):.4f}')

            eval_results['ds_average_precision'] = torch.nanmean(ds_ap)
            eval_results['ds_brier_score'] = torch.nanmean(ds_mse)
            eval_results['ds_f1'] = torch.nanmean(ds_f1)

            if self.ds_per_class:
                # log component-wise
                for ind, (ap_i, mse_i, f1_i) in enumerate(zip(ds_ap, ds_mse, ds_f1)):
                    logger_info.append(f'ds_average_precision_C{ind+1}: {ap_i:.4f}')
                    logger_info.append(f'ds_brier_score_C{ind+1}: {mse_i:.4f}')
                    logger_info.append(f'ds_f1_C{ind+1}: {f1_i:.4f}')

                    eval_results['ds_average_precision_C{}'.format(ind+1)] = ap_i
                    eval_results['ds_brier_score_C{}'.format(ind+1)] = mse_i
                    eval_results['ds_f1_C{}'.format(ind+1)] = f1_i

        logger.info(' '.join(logger_info))

        return eval_results

    def results2json(self):
        # make results dir if needed
        if not os.path.exists(self.outfile_prefix):
            os.makedirs(self.outfile_prefix)

        # save preds, gt, rounded gt
        pred_path = os.path.join(self.outfile_prefix, 'pred_ds.txt')
        gt_path = os.path.join(self.outfile_prefix, 'gt_ds.txt')
        rounded_gt_path = os.path.join(self.outfile_prefix, 'rounded_gt_ds.txt')

        np.savetxt(pred_path, torch.stack(self.cvs_preds).detach().cpu().numpy())
        np.savetxt(gt_path, torch.stack(self.cvs_gt).detach().cpu().numpy())
        np.savetxt(rounded_gt_path, torch.stack(self.rounded_cvs_gt).detach().cpu().numpy())
