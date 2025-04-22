#!/usr/bin/env python
# Evaluation script for corruption robustness of SurgLatentGraph on EndoScapes

import os
import glob
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from mmengine.registry import METRICS
from mmdet.evaluation import CocoMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate predictions for corruption robustness')
    parser.add_argument('--pred-dir', default='results/endoscapes_preds/', 
                      help='Directory containing prediction files')
    parser.add_argument('--out-dir', default='results/evaluation/',
                      help='Directory to save evaluation results')
    parser.add_argument('--gt-file', required=True,
                      help='Path to the ground truth annotation file')
    parser.add_argument('--metrics', nargs='+', default=['bbox'], 
                      choices=['bbox', 'segm', 'proposal'],
                      help='Evaluation metrics')
    return parser.parse_args()

def load_predictions(pred_path):
    """Load prediction results from file."""
    try:
        with open(pred_path, 'rb') as f:
            predictions = pickle.load(f)
        return predictions
    except (FileNotFoundError, pickle.PickleError) as e:
        print(f"Error loading predictions from {pred_path}: {e}")
        return None

def load_metadata(pred_dir):
    """Load metadata from prediction directory."""
    try:
        metadata_path = os.path.join(pred_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading metadata from {pred_dir}: {e}")
        return {}

def evaluate_predictions(predictions, gt_file, metrics):
    """Evaluate predictions using specified metrics."""
    # Initialize evaluation metric
    evaluator = CocoMetric(
        ann_file=gt_file,
        metric=metrics,
        classwise=True
    )
    
    # Reset evaluator
    evaluator.dataset_meta = None
    evaluator.reset()
    
    # Process predictions
    for pred in predictions:
        evaluator.process(None, pred)
    
    # Compute metrics
    eval_results = evaluator.compute_metrics(evaluator.results)
    
    return eval_results

def extract_per_video_metrics(eval_results, predictions):
    """Extract per-video metrics from evaluation results."""
    # This is a simplified implementation - you would need to adapt this 
    # to extract actual per-video results based on how your evaluator works
    video_metrics = {}
    
    # Example: Extract per-video metrics if available in predictions
    for i, pred in enumerate(predictions):
        if 'video_id' in pred:
            video_id = pred['video_id']
        else:
            # If video_id not present, use index as placeholder
            video_id = f"video_{i}"
            
        if video_id not in video_metrics:
            video_metrics[video_id] = {}
            
        # Add per-video metrics if available
        # This is a placeholder - real implementation would depend on your evaluator
        
    return video_metrics

def save_evaluation_results(eval_results, video_metrics, out_path):
    """Save evaluation results to file."""
    # Make sure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save overall metrics
    with open(out_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    # Return path to saved file
    return out_path

def extract_condition_info(pred_dir):
    """Extract training and evaluation condition information from directory name."""
    # Format: {train_condition}_{eval_condition}
    base_dir = os.path.basename(pred_dir)
    parts = base_dir.split('_')
    
    if len(parts) >= 2:
        train_condition = parts[0]
        eval_condition = '_'.join(parts[1:])
    else:
        train_condition = base_dir
        eval_condition = 'clean'
    
    return train_condition, eval_condition

def create_long_format_csv(evaluation_dirs, out_file):
    """Create a long format CSV with evaluation results for all conditions."""
    # Initialize dataframe data
    data = []
    
    for eval_dir in evaluation_dirs:
        # Load evaluation results
        try:
            with open(os.path.join(eval_dir, 'eval_results.json'), 'r') as f:
                eval_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Could not load evaluation results from {eval_dir}")
            continue
        
        # Extract condition information
        train_condition, eval_condition = extract_condition_info(eval_dir)
        
        # Extract video-specific results if available
        try:
            with open(os.path.join(eval_dir, 'video_metrics.json'), 'r') as f:
                video_metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            video_metrics = {}
        
        # Add overall metrics
        for metric_name, metric_value in eval_results.items():
            if isinstance(metric_value, (int, float)):
                data.append({
                    'video_id': 'overall',
                    'train_condition': train_condition,
                    'eval_condition': eval_condition,
                    'metric': metric_name,
                    'score': metric_value
                })
        
        # Add per-video metrics
        for video_id, metrics in video_metrics.items():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    data.append({
                        'video_id': video_id,
                        'train_condition': train_condition,
                        'eval_condition': eval_condition,
                        'metric': metric_name,
                        'score': metric_value
                    })
    
    # Create dataframe and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(out_file, index=False)
    
    print(f"Long format CSV saved to {out_file}")
    return df

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all prediction directories
    pred_dirs = glob.glob(os.path.join(args.pred_dir, '*'))
    
    evaluation_dirs = []
    
    for pred_dir in pred_dirs:
        if not os.path.isdir(pred_dir):
            continue
        
        print(f"Processing {pred_dir}...")
        
        # Load predictions
        pred_file = os.path.join(pred_dir, 'predictions.pkl')
        predictions = load_predictions(pred_file)
        if predictions is None:
            continue
        
        # Load metadata
        metadata = load_metadata(pred_dir)
        
        # Get train and eval conditions from directory name
        train_condition, eval_condition = extract_condition_info(pred_dir)
        
        # Create output directory for this evaluation
        eval_dir = os.path.join(args.out_dir, f"{train_condition}_{eval_condition}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Evaluate predictions
        eval_results = evaluate_predictions(predictions, args.gt_file, args.metrics)
        
        # Extract per-video metrics
        video_metrics = extract_per_video_metrics(eval_results, predictions)
        
        # Save evaluation results
        eval_file = os.path.join(eval_dir, 'eval_results.json')
        save_evaluation_results(eval_results, video_metrics, eval_file)
        
        # Save video metrics
        video_file = os.path.join(eval_dir, 'video_metrics.json')
        with open(video_file, 'w') as f:
            json.dump(video_metrics, f, indent=4)
        
        # Add to list of evaluation directories
        evaluation_dirs.append(eval_dir)
    
    # Create long format CSV
    csv_file = os.path.join(args.out_dir, 'all_evaluations.csv')
    create_long_format_csv(evaluation_dirs, csv_file)

if __name__ == '__main__':
    main()