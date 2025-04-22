#!/usr/bin/env python
# Statistical analysis script for corruption robustness

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description='Perform statistical analysis on evaluation results')
    parser.add_argument('--csv-file', default='results/evaluation/all_evaluations.csv',
                      help='Path to the long-format CSV file with all evaluations')
    parser.add_argument('--out-dir', default='results/statistics/',
                      help='Directory to save statistical results')
    parser.add_argument('--baseline', default='clean',
                      help='Baseline training condition to compare against')
    parser.add_argument('--alpha', type=float, default=0.05,
                      help='Significance level for statistical tests')
    parser.add_argument('--test', default='wilcoxon', choices=['wilcoxon', 'ttest'],
                      help='Statistical test to use')
    parser.add_argument('--metrics', nargs='+', 
                      default=['bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'segm_mAP'],
                      help='Metrics to analyze')
    
    return parser.parse_args()

def load_data(csv_file):
    """Load evaluation data from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error loading CSV file: {e}")
        return None

def check_normality(data):
    """Check if data follows a normal distribution using Shapiro-Wilk test."""
    _, p_value = stats.shapiro(data)
    return p_value > 0.05  # Return True if data appears normal (p > 0.05)

def perform_statistical_test(baseline_data, comparison_data, test_type='wilcoxon', alpha=0.05):
    """Perform statistical test between baseline and comparison data."""
    result = {
        'mean_baseline': np.mean(baseline_data),
        'mean_comparison': np.mean(comparison_data),
        'diff': np.mean(comparison_data) - np.mean(baseline_data),
        'rel_diff_percent': 100 * (np.mean(comparison_data) - np.mean(baseline_data)) / np.mean(baseline_data)
    }
    
    # Check if both datasets are normally distributed
    baseline_normal = check_normality(baseline_data)
    comparison_normal = check_normality(comparison_data)
    
    # Choose test based on normality and user preference
    if test_type == 'ttest' and baseline_normal and comparison_normal:
        # Paired t-test for normally distributed data
        t_stat, p_value = stats.ttest_rel(baseline_data, comparison_data)
        result['test'] = 'Paired t-test'
    else:
        # Wilcoxon signed-rank test (non-parametric alternative)
        t_stat, p_value = stats.wilcoxon(baseline_data, comparison_data)
        result['test'] = 'Wilcoxon signed-rank test'
    
    result['statistic'] = t_stat
    result['p_value'] = p_value
    result['significant'] = p_value < alpha
    
    return result

def analyze_metrics(df, baseline, metrics, test_type, alpha):
    """Analyze each metric for each corruption condition."""
    results = []
    
    # Filter to only include video-level metrics (exclude 'overall')
    df_videos = df[df['video_id'] != 'overall']
    
    # Get unique evaluation conditions
    eval_conditions = df['eval_condition'].unique()
    
    for metric in metrics:
        if metric not in df['metric'].unique():
            print(f"Warning: Metric '{metric}' not found in data. Available metrics: {df['metric'].unique()}")
            continue
            
        metric_df = df_videos[df_videos['metric'] == metric]
        
        # Get baseline data (clean training, various evaluation conditions)
        baseline_df = metric_df[metric_df['train_condition'] == baseline]
        
        for eval_condition in eval_conditions:
            # Skip if this is just the clean vs clean case
            if eval_condition == 'clean' and baseline == 'clean':
                continue
                
            # Get data for this evaluation condition
            baseline_eval_df = baseline_df[baseline_df['eval_condition'] == eval_condition]
            baseline_data = baseline_eval_df['score'].values
            
            # For each training condition compared to baseline
            for train_condition in df['train_condition'].unique():
                if train_condition == baseline:
                    continue
                    
                # Get comparison data
                comparison_df = metric_df[
                    (metric_df['train_condition'] == train_condition) & 
                    (metric_df['eval_condition'] == eval_condition)
                ]
                
                # Skip if no data
                if len(comparison_df) == 0 or len(baseline_eval_df) == 0:
                    continue
                    
                # Make sure we have matching video IDs
                common_videos = set(baseline_eval_df['video_id']).intersection(set(comparison_df['video_id']))
                if len(common_videos) == 0:
                    continue
                    
                # Filter to common videos
                baseline_data = baseline_eval_df[baseline_eval_df['video_id'].isin(common_videos)]['score'].values
                comparison_data = comparison_df[comparison_df['video_id'].isin(common_videos)]['score'].values
                
                # Perform statistical test
                result = perform_statistical_test(baseline_data, comparison_data, test_type, alpha)
                
                # Add context
                result['metric'] = metric
                result['train_condition'] = train_condition
                result['eval_condition'] = eval_condition
                result['baseline'] = baseline
                result['num_videos'] = len(common_videos)
                
                results.append(result)
    
    return pd.DataFrame(results)

def create_summary_table(results):
    """Create a summary table of statistical results."""
    if len(results) == 0:
        return "No results to display"
        
    # Format table
    table_data = []
    for _, row in results.iterrows():
        sig_marker = "(*)" if row['significant'] else ""
        table_data.append([
            row['metric'], 
            f"{row['train_condition']} vs {row['baseline']}", 
            row['eval_condition'],
            f"{row['mean_baseline']:.4f}",
            f"{row['mean_comparison']:.4f}",
            f"{row['rel_diff_percent']:.2f}% {sig_marker}",
            f"{row['p_value']:.4f}"
        ])
    
    headers = ["Metric", "Training", "Evaluation", "Baseline", "Comparison", "Change (%)", "p-value"]
    return tabulate(table_data, headers=headers, tablefmt="grid")

def create_visualizations(results, df, out_dir):
    """Create visualizations of the results."""
    if len(results) == 0:
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot 1: Performance difference across evaluation conditions
    plt.figure(figsize=(12, 8))
    subset = results[results['metric'] == results['metric'].iloc[0]]
    sns.barplot(data=subset, x='eval_condition', y='rel_diff_percent', hue='train_condition')
    plt.title(f'Performance Difference from {subset["baseline"].iloc[0]} ({subset["metric"].iloc[0]})')
    plt.xlabel('Evaluation Condition')
    plt.ylabel('Relative Difference (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'perf_diff_by_eval.png'))
    plt.close()
    
    # Plot 2: Heatmap of p-values
    metrics = results['metric'].unique()
    for metric in metrics:
        metric_results = results[results['metric'] == metric].copy()
        if len(metric_results) == 0:
            continue
            
        # Create a pivot table of p-values
        pivot = metric_results.pivot(index='train_condition', columns='eval_condition', values='p_value')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap='RdYlGn_r', vmin=0, vmax=0.1)
        plt.title(f'P-values for {metric} (vs {metric_results["baseline"].iloc[0]})')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'pvalue_heatmap_{metric}.png'))
        plt.close()
    
    # Plot 3: Performance across training and evaluation conditions
    for metric in df['metric'].unique():
        if metric not in metrics:
            continue
            
        metric_df = df[df['metric'] == metric]
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=metric_df, x='eval_condition', y='score', hue='train_condition')
        plt.title(f'Performance Across Conditions ({metric})')
        plt.xlabel('Evaluation Condition')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'performance_boxplot_{metric}.png'))
        plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    df = load_data(args.csv_file)
    if df is None:
        return
    
    # Analyze metrics
    results = analyze_metrics(df, args.baseline, args.metrics, args.test, args.alpha)
    
    # Save results
    results_file = os.path.join(args.out_dir, 'statistical_results.csv')
    results.to_csv(results_file, index=False)
    
    # Create summary table
    summary_table = create_summary_table(results)
    summary_file = os.path.join(args.out_dir, 'summary_table.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_table)
    
    # Print summary
    print("\nStatistical Analysis Summary:")
    print(summary_table)
    
    # Create visualizations
    create_visualizations(results, df, args.out_dir)
    
    print(f"\nResults saved to {args.out_dir}")
    print(f"Statistical results: {results_file}")
    print(f"Summary table: {summary_file}")

if __name__ == '__main__':
    main()