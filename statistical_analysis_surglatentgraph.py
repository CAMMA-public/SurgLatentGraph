#!/usr/bin/env python3
"""
statistical_analysis_surglatentgraph.py

Performs Wilcoxon signed-rank tests comparing clean-trained vs corruption-trained models
for each corruption type and metric in the SurgLatentGraph project.

For each corruption type, compares:
- Clean-trained model tested on corruption
- Corruption-trained model tested on same corruption

Usage:
  python3 statistical_analysis_surglatentgraph.py \
    --input metrics/consolidated_metrics.csv \
    --output results/statistical_analysis_results.txt \
    [--alpha 0.05]
"""

import argparse
import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical significance testing for SurgLatentGraph corruption study"
    )
    parser.add_argument('-i', '--input', 
                       default='metrics/consolidated_metrics.csv',
                       help='Path to consolidated_metrics.csv')
    parser.add_argument('-o', '--output', 
                       default='metrics/statistical_analysis_results.txt',
                       help='Output file for results')
    parser.add_argument('-a', '--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--csv-output', 
                       default='metrics/statistical_results.csv',
                       help='CSV output file for results')
    return parser.parse_args()

def load_and_validate_data(file_path):
    """Load and validate the consolidated metrics data"""
    try:
        df = pd.read_csv(file_path)
        required_cols = {'video_id', 'train_condition', 'test_condition', 'metric', 'score'}
        
        if not required_cols.issubset(df.columns):
            print(f"ERROR: Missing required columns. Found: {list(df.columns)}")
            print(f"Required: {required_cols}")
            sys.exit(1)
            
        print(f"✅ Loaded data: {df.shape[0]} rows, {df['video_id'].nunique()} videos")
        return df
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

def identify_comparisons(df):
    """Identify valid clean vs corruption comparisons"""
    comparisons = []
    
    # Get corruption types (exclude 'none' which is clean)
    corruption_types = [c for c in df['train_condition'].unique() if c != 'none']
    
    for corruption in corruption_types:
        # Check if we have both clean and corruption-trained data for this corruption
        test_subset = df[df['test_condition'] == corruption]
        
        clean_data = test_subset[test_subset['train_condition'] == 'none']
        corrupt_data = test_subset[test_subset['train_condition'] == corruption]
        
        if len(clean_data) > 0 and len(corrupt_data) > 0:
            # Check if we have the same videos in both groups
            clean_videos = set(clean_data['video_id'].unique())
            corrupt_videos = set(corrupt_data['video_id'].unique())
            common_videos = clean_videos.intersection(corrupt_videos)
            
            if len(common_videos) > 0:
                comparisons.append({
                    'corruption': corruption,
                    'clean_train_test_corrupt': len(clean_data),
                    'corrupt_train_test_corrupt': len(corrupt_data),
                    'common_videos': len(common_videos),
                    'total_videos': len(common_videos)
                })
                
    return comparisons

def calculate_effect_size(x, y):
    """Calculate effect size (Cohen's d) for paired samples"""
    diff = y - x
    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    if pooled_std == 0:
        return 0
    return np.mean(diff) / pooled_std

def perform_statistical_tests(df, comparisons, metrics, alpha):
    """Perform Wilcoxon signed-rank tests for all comparisons"""
    results = []
    
    for comp in comparisons:
        corruption = comp['corruption']
        
        print(f"\n🔬 Testing corruption: {corruption}")
        print(f"   Videos available: {comp['common_videos']}")
        
        for metric in metrics:
            # Get data for this corruption and metric
            test_subset = df[
                (df['test_condition'] == corruption) & 
                (df['metric'] == metric)
            ]
            
            # Pivot to get clean vs corruption-trained scores for each video
            pivot_data = test_subset.pivot_table(
                index='video_id', 
                columns='train_condition', 
                values='score'
            )
            
            # Check if we have both conditions
            if 'none' not in pivot_data.columns or corruption not in pivot_data.columns:
                print(f"   ⚠️  {metric}: Missing data for comparison")
                continue
                
            # Get paired data (remove NaN values)
            clean_scores = pivot_data['none'].dropna()
            corrupt_scores = pivot_data[corruption].dropna()
            
            # Find common videos
            common_videos = clean_scores.index.intersection(corrupt_scores.index)
            
            if len(common_videos) < 3:
                print(f"   ⚠️  {metric}: Insufficient paired data (n={len(common_videos)})")
                continue
                
            # Extract paired scores
            clean_paired = clean_scores.loc[common_videos].values
            corrupt_paired = corrupt_scores.loc[common_videos].values
            
            # Perform Wilcoxon signed-rank test
            try:
                stat, p_value = wilcoxon(clean_paired, corrupt_paired, alternative='two-sided')
                
                # Calculate descriptive statistics
                clean_median = np.median(clean_paired)
                corrupt_median = np.median(corrupt_paired)
                median_diff = corrupt_median - clean_median
                
                # Calculate effect size
                effect_size = calculate_effect_size(clean_paired, corrupt_paired)
                
                # Determine significance
                is_significant = p_value < alpha
                
                # Determine direction of effect
                if median_diff > 0:
                    direction = "Corruption training IMPROVES performance"
                elif median_diff < 0:
                    direction = "Corruption training DEGRADES performance"
                else:
                    direction = "No difference"
                
                result = {
                    'corruption': corruption,
                    'metric': metric,
                    'n_videos': len(common_videos),
                    'clean_median': clean_median,
                    'corrupt_median': corrupt_median,
                    'median_difference': median_diff,
                    'wilcoxon_statistic': stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': is_significant,
                    'direction': direction
                }
                
                results.append(result)
                
                # Print result
                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"   📊 {metric:12} | n={len(common_videos):2d} | "
                      f"Clean: {clean_median:.3f} | Corrupt: {corrupt_median:.3f} | "
                      f"Diff: {median_diff:+.3f} | p={p_value:.3f}{sig_marker}")
                
            except Exception as e:
                print(f"   ❌ {metric}: Error in statistical test: {e}")
                
    return results

def write_detailed_results(results, output_file, alpha):
    """Write detailed results to text file"""
    with open(output_file, 'w') as f:
        f.write("# SurgLatentGraph Corruption Training - Statistical Analysis Results\n")
        f.write("# Wilcoxon Signed-Rank Tests: Clean-trained vs Corruption-trained Models\n")
        f.write(f"# Significance level (α): {alpha}\n")
        f.write(f"# Total comparisons: {len(results)}\n\n")
        
        # Group results by corruption type
        corruption_types = sorted(set(r['corruption'] for r in results))
        
        for corruption in corruption_types:
            corruption_results = [r for r in results if r['corruption'] == corruption]
            
            f.write(f"{'='*80}\n")
            f.write(f"CORRUPTION TYPE: {corruption.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            # Create table
            f.write(f"{'Metric':<15} {'N':<4} {'Clean':<8} {'Corrupt':<8} {'Diff':<8} "
                   f"{'Stat':<8} {'P-value':<10} {'Effect':<8} {'Significant':<11} {'Direction'}\n")
            f.write("-" * 100 + "\n")
            
            for r in corruption_results:
                sig_str = "Yes" if r['significant'] else "No"
                f.write(f"{r['metric']:<15} {r['n_videos']:<4} "
                       f"{r['clean_median']:<8.3f} {r['corrupt_median']:<8.3f} "
                       f"{r['median_difference']:<+8.3f} "
                       f"{r['wilcoxon_statistic']:<8.1f} {r['p_value']:<10.3f} "
                       f"{r['effect_size']:<+8.3f} {sig_str:<11} {r['direction']}\n")
            
            # Summary for this corruption
            sig_count = sum(1 for r in corruption_results if r['significant'])
            total_count = len(corruption_results)
            
            f.write(f"\nSummary for {corruption}:\n")
            f.write(f"  Significant results: {sig_count}/{total_count} ({sig_count/total_count*100:.1f}%)\n")
            
            # Direction analysis
            improvements = [r for r in corruption_results if r['significant'] and r['median_difference'] > 0]
            degradations = [r for r in corruption_results if r['significant'] and r['median_difference'] < 0]
            
            f.write(f"  Significant improvements: {len(improvements)}\n")
            f.write(f"  Significant degradations: {len(degradations)}\n\n")

def write_csv_results(results, csv_file):
    """Write results to CSV file for further analysis"""
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_file, index=False)
    print(f"📊 CSV results saved to: {csv_file}")

def print_summary(results, alpha):
    """Print overall summary"""
    print(f"\n{'='*80}")
    print("📈 OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    significant_tests = sum(1 for r in results if r['significant'])
    
    print(f"Total statistical tests performed: {total_tests}")
    print(f"Significant results (α={alpha}): {significant_tests} ({significant_tests/total_tests*100:.1f}%)")
    
    # Direction analysis
    improvements = [r for r in results if r['significant'] and r['median_difference'] > 0]
    degradations = [r for r in results if r['significant'] and r['median_difference'] < 0]
    
    print(f"\nSignificant improvements (corruption training helps): {len(improvements)}")
    print(f"Significant degradations (corruption training hurts): {len(degradations)}")
    
    # Corruption-wise summary
    corruption_summary = {}
    for corruption in set(r['corruption'] for r in results):
        corruption_results = [r for r in results if r['corruption'] == corruption]
        sig_results = [r for r in corruption_results if r['significant']]
        corruption_summary[corruption] = {
            'total': len(corruption_results),
            'significant': len(sig_results)
        }
    
    print(f"\nPer-corruption significance rates:")
    for corruption, stats in corruption_summary.items():
        rate = stats['significant'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {corruption:<20}: {stats['significant']}/{stats['total']} ({rate:.1f}%)")

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_dir = Path(args.csv_output).parent
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("🔄 Loading and validating data...")
    df = load_and_validate_data(args.input)
    
    # Identify possible comparisons
    print("\n🔍 Identifying valid comparisons...")
    comparisons = identify_comparisons(df)
    
    if not comparisons:
        print("❌ No valid comparisons found!")
        sys.exit(1)
    
    print(f"✅ Found {len(comparisons)} valid corruption types for comparison:")
    for comp in comparisons:
        print(f"   {comp['corruption']}: {comp['common_videos']} videos")
    
    # Get metrics
    metrics = sorted(df['metric'].unique())
    print(f"\n📊 Metrics to analyze: {metrics}")
    
    # Perform statistical tests
    print(f"\n🧪 Performing Wilcoxon signed-rank tests (α={args.alpha})...")
    results = perform_statistical_tests(df, comparisons, metrics, args.alpha)
    
    if not results:
        print("❌ No statistical tests could be performed!")
        sys.exit(1)
    
    # Write detailed results
    print(f"\n📝 Writing detailed results to: {args.output}")
    write_detailed_results(results, args.output, args.alpha)
    
    # Write CSV results
    write_csv_results(results, args.csv_output)
    
    # Print summary
    print_summary(results, args.alpha)
    
    print(f"\n✅ Statistical analysis complete!")
    print(f"📄 Detailed results: {args.output}")
    print(f"📊 CSV results: {args.csv_output}")

if __name__ == '__main__':
    main()
