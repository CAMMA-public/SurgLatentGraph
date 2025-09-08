#!/usr/bin/env python3
"""
generate_significance_heatmaps.py

Creates heatmaps showing statistical significance levels for corruption training analysis.
Generates multiple visualizations:
1. P-value heatmap with significance color coding
2. Effect size heatmap with direction indicators
3. Combined overview heatmap
4. Performance difference heatmap

Usage:
  python3 generate_significance_heatmaps.py \
    --input metrics/statistical_results.csv \
    --output-dir metrics/heatmaps \
    [--alpha 0.05]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate significance heatmaps for SurgLatentGraph corruption study"
    )
    parser.add_argument('-i', '--input', 
                       default='metrics/statistical_results.csv',
                       help='Path to statistical_results.csv file')
    parser.add_argument('-o', '--output-dir', 
                       default='metrics/heatmaps_4',
                       help='Output directory for heatmap images')
    parser.add_argument('-a', '--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output images (default: 300)')
    return parser.parse_args()

def load_statistical_results(file_path):
    """Load statistical results from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded statistical results: {len(df)} comparisons")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("💡 Run statistical_analysis_surglatentgraph.py first to generate the CSV file")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def create_significance_matrix(df, value_col, fill_value=np.nan):
    """Create a matrix for heatmap visualization"""
    # Create pivot table
    matrix = df.pivot_table(
        index='metric',
        columns='corruption',
        values=value_col,
        fill_value=fill_value
    )
    
    # Reorder columns for better visualization
    corruption_order = ['gaussian_noise', 'motion_blur', 'defocus_blur', 
                       'smoke_effect', 'uneven_illumination', 'random_corruptions']
    
    # Only include columns that exist in the data
    available_corruptions = [c for c in corruption_order if c in matrix.columns]
    matrix = matrix[available_corruptions]
    
    return matrix

def create_direct_pvalue_heatmap(df, output_path, alpha=0.05, dpi=300):
    """Create p-value heatmap showing direct p-values with clear significance indicators"""
    
    # Create p-value matrix
    pvalue_matrix = create_significance_matrix(df, 'p_value', fill_value=1.0)
    
    # Create significance matrix for annotations
    sig_matrix = create_significance_matrix(df, 'significant', fill_value=False)
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Create heatmap with direct p-values
    # Custom colormap: Green for significant (p < 0.05), Red for not significant
    ax = sns.heatmap(
        pvalue_matrix,
        annot=False,  # We'll add custom annotations
        cmap='RdYlGn_r',  # Inverted so small p-values are green
        vmin=0,
        vmax=0.15,  # Extended range to show non-significant values
        cbar_kws={'label': 'P-value (Green = Significant, Red = Not Significant)'},
        linewidths=1,
        square=True
    )
    
    # Add custom annotations with p-values and significance indicators
    for i, metric in enumerate(pvalue_matrix.index):
        for j, corruption in enumerate(pvalue_matrix.columns):
            p_val = pvalue_matrix.loc[metric, corruption]
            is_sig = sig_matrix.loc[metric, corruption]
            
            # Format p-value
            if p_val < 0.001:
                p_text = 'p<0.001'
            else:
                p_text = f'p={p_val:.3f}'
            
            # Add significance indicator
            if is_sig:
                p_text += '\n***'
                text_color = 'white' if p_val < 0.03 else 'black'
            else:
                p_text += '\nn.s.'
                text_color = 'black'
            
            # Add text annotation
            ax.text(j + 0.5, i + 0.5, p_text, 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color=text_color)
    
    # Customize plot
    plt.title('Direct P-value Heatmap\n(*** = Significant, n.s. = Not Significant)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Corruption Type', fontsize=14, fontweight='bold')
    plt.ylabel('Metric', fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Add significance threshold line to colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.axhline(y=alpha, color='red', linestyle='--', linewidth=3)
    cbar.ax.text(0.5, alpha + 0.01, f'α={alpha}', transform=cbar.ax.transData, 
                ha='center', va='bottom', fontweight='bold', color='red', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Direct p-value heatmap saved: {output_path}")

def create_pvalue_heatmap(df, output_path, alpha=0.05, dpi=300):
    """Create p-value heatmap with significance color coding"""
    
    # Create p-value matrix
    pvalue_matrix = create_significance_matrix(df, 'p_value', fill_value=1.0)
    
    # Create significance matrix for annotations
    sig_matrix = create_significance_matrix(df, 'significant', fill_value=False)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with direct p-values
    # Use inverted colormap so smaller p-values (more significant) are greener
    ax = sns.heatmap(
        pvalue_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',  # Inverted so small p-values are green
        vmin=0,
        vmax=0.1,  # Focus on the significant range
        cbar_kws={'label': 'P-value\n(Green = More Significant)'},
        linewidths=0.5,
        square=True,
        annot_kws={'size': 10}
    )
    
    # Add significance markers
    for i, metric in enumerate(pvalue_matrix.index):
        for j, corruption in enumerate(pvalue_matrix.columns):
            if sig_matrix.loc[metric, corruption]:
                # Add asterisk for significant results
                ax.text(j + 0.7, i + 0.3, '*', fontsize=16, fontweight='bold', 
                       color='black', ha='center', va='center')
    
    # Customize plot
    plt.title('P-value Heatmap\n(* = Significant, Green = More Significant)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Corruption Type', fontsize=12, fontweight='bold')
    plt.ylabel('Metric', fontsize=12, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add significance threshold line to colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2)
    cbar.ax.text(0.5, alpha + 0.005, f'α={alpha}', transform=cbar.ax.transData, 
                ha='center', va='bottom', fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"📊 P-value heatmap saved: {output_path}")

def create_effect_size_heatmap(df, output_path, dpi=300):
    """Create effect size heatmap with direction indicators"""
    
    # Create effect size matrix
    effect_matrix = create_significance_matrix(df, 'effect_size', fill_value=0)
    
    # Create significance matrix for annotations
    sig_matrix = create_significance_matrix(df, 'significant', fill_value=False)
    
    # Create median difference matrix for direction
    diff_matrix = create_significance_matrix(df, 'median_difference', fill_value=0)
    
    plt.figure(figsize=(12, 8))
    
    # Use RdBu_r colormap: Red for negative (worse), Blue for positive (better)
    max_abs_effect = max(abs(effect_matrix.min().min()), abs(effect_matrix.max().max()))
    vmax = max(max_abs_effect, 0.1)  # Ensure we have some range
    
    ax = sns.heatmap(
        effect_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={'label': 'Effect Size\n(Blue = Improvement, Red = Degradation)'},
        linewidths=0.5,
        square=True,
        annot_kws={'size': 10}
    )
    
    # Add significance markers and direction arrows
    for i, metric in enumerate(effect_matrix.index):
        for j, corruption in enumerate(effect_matrix.columns):
            if sig_matrix.loc[metric, corruption]:
                # Add asterisk for significant results
                ax.text(j + 0.8, i + 0.2, '*', fontsize=16, fontweight='bold', 
                       color='white', ha='center', va='center')
                
                # Add direction arrow based on median difference
                diff_val = diff_matrix.loc[metric, corruption]
                if diff_val > 0:
                    arrow = '↑'  # Improvement
                    color = 'darkblue'
                else:
                    arrow = '↓'  # Degradation
                    color = 'darkred'
                
                ax.text(j + 0.2, i + 0.2, arrow, fontsize=14, fontweight='bold', 
                       color=color, ha='center', va='center')
    
    plt.title('Effect Size Heatmap\n(* = Significant, ↑ = Improvement, ↓ = Degradation)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Corruption Type', fontsize=12, fontweight='bold')
    plt.ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Effect size heatmap saved: {output_path}")

def create_performance_difference_heatmap(df, output_path, dpi=300):
    """Create performance difference heatmap (corrupt - clean)"""
    
    # Create median difference matrix
    diff_matrix = create_significance_matrix(df, 'median_difference', fill_value=0)
    
    # Create significance matrix for annotations
    sig_matrix = create_significance_matrix(df, 'significant', fill_value=False)
    
    plt.figure(figsize=(12, 8))
    
    # Use RdYlGn colormap: Red for negative (worse), Green for positive (better)
    max_abs_diff = max(abs(diff_matrix.min().min()), abs(diff_matrix.max().max()))
    vmax = max(max_abs_diff, 0.01)  # Ensure we have some range
    
    ax = sns.heatmap(
        diff_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={'label': 'Performance Difference\n(Corruption-trained - Clean-trained)'},
        linewidths=0.5,
        square=True,
        annot_kws={'size': 9}
    )
    
    # Add significance markers
    for i, metric in enumerate(diff_matrix.index):
        for j, corruption in enumerate(diff_matrix.columns):
            if sig_matrix.loc[metric, corruption]:
                ax.text(j + 0.8, i + 0.2, '★', fontsize=12, fontweight='bold', 
                       color='black', ha='center', va='center')
    
    plt.title('Performance Difference Heatmap\n(★ = Statistically Significant)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Corruption Type', fontsize=12, fontweight='bold')
    plt.ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Performance difference heatmap saved: {output_path}")

def create_combined_overview(df, output_dir, alpha=0.05, dpi=300):
    """Create a combined overview with multiple subplots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. P-value heatmap (direct p-values)
    pvalue_matrix = create_significance_matrix(df, 'p_value', fill_value=1.0)
    sig_matrix = create_significance_matrix(df, 'significant', fill_value=False)
    
    sns.heatmap(pvalue_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                vmin=0, vmax=0.1, ax=axes[0,0], cbar_kws={'label': 'P-value'},
                linewidths=0.5, square=True, annot_kws={'size': 8})
    
    # Add significance markers
    for i, metric in enumerate(pvalue_matrix.index):
        for j, corruption in enumerate(pvalue_matrix.columns):
            if sig_matrix.loc[metric, corruption]:
                axes[0,0].text(j + 0.7, i + 0.3, '*', fontsize=12, fontweight='bold', 
                              color='black', ha='center', va='center')
    
    axes[0,0].set_title('A) P-values\n(* = Significant, Green = More Significant)', fontweight='bold')
    axes[0,0].set_xlabel('')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Effect size heatmap
    effect_matrix = create_significance_matrix(df, 'effect_size', fill_value=0)
    max_abs_effect = max(abs(effect_matrix.min().min()), abs(effect_matrix.max().max()))
    vmax_effect = max(max_abs_effect, 0.1)
    
    sns.heatmap(effect_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-vmax_effect, vmax=vmax_effect, ax=axes[0,1], 
                cbar_kws={'label': 'Effect Size'}, linewidths=0.5, square=True,
                annot_kws={'size': 8})
    
    axes[0,1].set_title('B) Effect Size\n(Blue = Better, Red = Worse)', fontweight='bold')
    axes[0,1].set_xlabel('')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Performance difference heatmap
    diff_matrix = create_significance_matrix(df, 'median_difference', fill_value=0)
    max_abs_diff = max(abs(diff_matrix.min().min()), abs(diff_matrix.max().max()))
    vmax_diff = max(max_abs_diff, 0.01)
    
    sns.heatmap(diff_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                vmin=-vmax_diff, vmax=vmax_diff, ax=axes[1,0], 
                cbar_kws={'label': 'Performance Δ'}, linewidths=0.5, square=True,
                annot_kws={'size': 8})
    
    axes[1,0].set_title('C) Performance Difference\n(Green = Improvement)', fontweight='bold')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Summary significance count
    corruption_summary = df.groupby('corruption').agg({
        'significant': 'sum',
        'metric': 'count'
    }).rename(columns={'metric': 'total'})
    corruption_summary['significance_rate'] = corruption_summary['significant'] / corruption_summary['total']
    
    # Create a simple bar plot
    corruptions = corruption_summary.index
    rates = corruption_summary['significance_rate'].values
    
    bars = axes[1,1].bar(range(len(corruptions)), rates, 
                        color=['green' if r > 0.5 else 'orange' if r > 0.25 else 'red' for r in rates])
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_ylabel('Significance Rate')
    axes[1,1].set_title('D) Significance Rate by Corruption', fontweight='bold')
    axes[1,1].set_xticks(range(len(corruptions)))
    axes[1,1].set_xticklabels(corruptions, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        axes[1,1].text(i, rate + 0.02, f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('SurgLatentGraph Corruption Training - Statistical Analysis Overview', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    output_path = output_dir / 'combined_overview.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Combined overview saved: {output_path}")

def create_summary_statistics(df, output_dir):
    """Create and save summary statistics"""
    
    summary_stats = {
        'total_comparisons': len(df),
        'significant_results': df['significant'].sum(),
        'significance_rate': df['significant'].mean(),
        'improvements': len(df[(df['significant']) & (df['median_difference'] > 0)]),
        'degradations': len(df[(df['significant']) & (df['median_difference'] < 0)]),
    }
    
    # Per-corruption summary
    corruption_summary = df.groupby('corruption').agg({
        'significant': ['count', 'sum'],
        'median_difference': 'mean',
        'effect_size': 'mean'
    }).round(3)
    
    # Per-metric summary
    metric_summary = df.groupby('metric').agg({
        'significant': ['count', 'sum'],
        'median_difference': 'mean',
        'effect_size': 'mean'
    }).round(3)
    
    # Save summary to text file
    summary_path = output_dir / 'summary_statistics.txt'
    with open(summary_path, 'w') as f:
        f.write("# SurgLatentGraph Corruption Training - Summary Statistics\n\n")
        
        f.write("## Overall Summary\n")
        f.write(f"Total comparisons: {summary_stats['total_comparisons']}\n")
        f.write(f"Significant results: {summary_stats['significant_results']} "
               f"({summary_stats['significance_rate']:.1%})\n")
        f.write(f"Significant improvements: {summary_stats['improvements']}\n")
        f.write(f"Significant degradations: {summary_stats['degradations']}\n\n")
        
        f.write("## Per-Corruption Summary\n")
        f.write(corruption_summary.to_string())
        f.write("\n\n## Per-Metric Summary\n")
        f.write(metric_summary.to_string())
    
    print(f"📈 Summary statistics saved: {summary_path}")
    
    return summary_stats

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load statistical results
    print("🔄 Loading statistical results...")
    df = load_statistical_results(args.input)
    
    if df is None:
        return
    
    print(f"📊 Generating heatmaps for {len(df)} statistical comparisons...")
    
    # Create individual heatmaps
    print("\n🎨 Creating individual heatmaps...")
    
    # 1. Direct P-value heatmap (new)
    direct_pvalue_path = output_dir / 'direct_pvalue_heatmap.png'
    create_direct_pvalue_heatmap(df, direct_pvalue_path, args.alpha, args.dpi)
    
    # 2. P-value significance heatmap (original)
    pvalue_path = output_dir / 'pvalue_significance_heatmap.png'
    create_pvalue_heatmap(df, pvalue_path, args.alpha, args.dpi)
    
    # 3. Effect size heatmap
    effect_path = output_dir / 'effect_size_heatmap.png'
    create_effect_size_heatmap(df, effect_path, args.dpi)
    
    # 4. Performance difference heatmap
    diff_path = output_dir / 'performance_difference_heatmap.png'
    create_performance_difference_heatmap(df, diff_path, args.dpi)
    
    # 5. Combined overview
    print("\n🎨 Creating combined overview...")
    create_combined_overview(df, output_dir, args.alpha, args.dpi)
    
    # 6. Generate summary statistics
    print("\n📈 Generating summary statistics...")
    summary_stats = create_summary_statistics(df, output_dir)
    
    # Final summary
    print(f"\n✅ Heatmap generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Generated {len(list(output_dir.glob('*.png')))} heatmap images")
    print(f"📈 Overall significance rate: {summary_stats['significance_rate']:.1%}")
    
    print(f"\n🎯 Key findings:")
    print(f"   • {summary_stats['significant_results']}/{summary_stats['total_comparisons']} "
          f"comparisons are statistically significant")
    print(f"   • {summary_stats['improvements']} significant improvements")
    print(f"   • {summary_stats['degradations']} significant degradations")

if __name__ == '__main__':
    main()
