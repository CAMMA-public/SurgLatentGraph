#!/usr/bin/env python3
"""
Script to merge all CSV files in the metrics folder into a single consolidated CSV file.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def merge_csv_files(metrics_folder="metrics", output_file="merged_metrics.csv"):
    """
    Merge all CSV files in the metrics folder into a single CSV file.
    
    Args:
        metrics_folder (str): Path to the folder containing CSV files
        output_file (str): Name of the output merged CSV file
    """
    
    # Get all CSV files in the metrics folder
    csv_pattern = os.path.join(metrics_folder, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {metrics_folder} folder")
        return
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for file in sorted(csv_files):
        print(f"  - {os.path.basename(file)}")
    
    merged_data = []
    file_sources = []
    
    for csv_file in sorted(csv_files):
        try:
            # Read each CSV file
            df = pd.read_csv(csv_file)
            
            # Add a column to track source file
            df['source_file'] = os.path.basename(csv_file)
            
            # Add to merged data
            merged_data.append(df)
            file_sources.append(os.path.basename(csv_file))
            
            print(f"  ✓ Processed {os.path.basename(csv_file)}: {len(df)} rows")
            
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(csv_file)}: {e}")
            continue
    
    if not merged_data:
        print("No valid CSV files could be processed")
        return
    
    # Concatenate all dataframes
    try:
        merged_df = pd.concat(merged_data, ignore_index=True)
        
        # Sort by source_file and video_id (if it exists)
        if 'video_id' in merged_df.columns:
            merged_df = merged_df.sort_values(['source_file', 'video_id', 'metric'])
        else:
            merged_df = merged_df.sort_values(['source_file'])
        
        # Save the merged file
        output_path = os.path.join(metrics_folder, output_file)
        merged_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Successfully merged {len(csv_files)} files into {output_path}")
        print(f"  Total rows: {len(merged_df)}")
        print(f"  Columns: {list(merged_df.columns)}")
        
        # Show summary by source file
        print(f"\nSummary by source file:")
        file_summary = merged_df['source_file'].value_counts().sort_index()
        for source_file, count in file_summary.items():
            print(f"  {source_file}: {count} rows")
            
        # Show sample of merged data
        print(f"\nFirst 5 rows of merged data:")
        print(merged_df.head().to_string(index=False))
        
    except Exception as e:
        print(f"Error merging data: {e}")

def main():
    """Main function to run the CSV merging process"""
    
    # Check if metrics folder exists
    metrics_folder = "metrics"
    if not os.path.exists(metrics_folder):
        print(f"Error: {metrics_folder} folder not found")
        return
    
    print("=== CSV Merger for Metrics Folder ===")
    print(f"Merging CSV files from: {os.path.abspath(metrics_folder)}")
    
    # Run the merge
    merge_csv_files(metrics_folder, "consolidated_metrics.csv")
    
    print("\n=== Merge Complete ===")

if __name__ == "__main__":
    main()
