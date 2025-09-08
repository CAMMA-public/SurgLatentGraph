#!/usr/bin/env python3
"""
analyze_data_structure.py
Quick script to understand the structure of consolidated_metrics.csv
"""

import pandas as pd

def main():
    # Load the data
    df = pd.read_csv('/workspace/metrics/consolidated_metrics.csv')
    
    print("=== DATA STRUCTURE ANALYSIS ===")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nTrain conditions: {sorted(df['train_condition'].unique())}")
    print(f"Test conditions: {sorted(df['test_condition'].unique())}")
    print(f"Metrics: {sorted(df['metric'].unique())}")
    
    print(f"\nVideo ID range: {df['video_id'].min()} to {df['video_id'].max()}")
    print(f"Unique videos: {df['video_id'].nunique()}")
    
    print("\n=== COMBINATION MATRIX ===")
    combination_counts = df.groupby(['train_condition', 'test_condition']).size().unstack(fill_value=0)
    print(combination_counts)
    
    print("\n=== SAMPLE DATA ===")
    print(df.head(10))
    
    # Check for baseline (clean) comparisons
    clean_train = df[df['train_condition'] == 'none']
    corruption_train = df[df['train_condition'] != 'none']
    
    print(f"\nClean-trained data: {len(clean_train)} rows")
    print(f"Corruption-trained data: {len(corruption_train)} rows")
    
    if len(corruption_train) > 0:
        print(f"Corruption types in training: {sorted(corruption_train['train_condition'].unique())}")
    
    # Check what comparisons are possible
    print(f"\n=== STATISTICAL TESTING POSSIBILITIES ===")
    
    # Find videos that have both clean and corruption-trained results on same test condition
    for test_cond in df['test_condition'].unique():
        test_subset = df[df['test_condition'] == test_cond]
        video_train_combinations = test_subset.groupby('video_id')['train_condition'].nunique()
        paired_videos = video_train_combinations[video_train_combinations > 1]
        
        if len(paired_videos) > 0:
            print(f"Test condition '{test_cond}': {len(paired_videos)} videos have multiple training conditions")
            # Show which train conditions are available
            available_trains = test_subset['train_condition'].unique()
            print(f"  Available training conditions: {sorted(available_trains)}")

if __name__ == "__main__":
    main()
