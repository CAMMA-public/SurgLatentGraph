#!/bin/bash

# Script to extract and organize test metrics into a summary folder
# This should be run after testing has completed

# Find the latest results directory
LATEST_RESULTS_DIR=$(readlink -f "results/latest")

if [ ! -d "$LATEST_RESULTS_DIR" ]; then
    echo "Cannot find latest results directory. Make sure results/latest symlink exists."
    exit 1
fi

echo "Processing metrics from: $LATEST_RESULTS_DIR"

# Process each model subdirectory
for MODEL_DIR in "$LATEST_RESULTS_DIR"/lg_*_rcnn; do
    if [ -d "$MODEL_DIR" ]; then
        MODEL_NAME=$(basename "$MODEL_DIR")
        TEST_RESULTS_DIR="$MODEL_DIR/test_results"
        
        if [ -d "$TEST_RESULTS_DIR" ]; then
            echo "Processing test results for $MODEL_NAME..."
            
            # Create metrics summary directory
            METRICS_DIR="$TEST_RESULTS_DIR/metrics_summary"
            mkdir -p "$METRICS_DIR"
            
            # Find and copy key metric files
            if [ -f "$TEST_RESULTS_DIR/threshs.txt" ]; then
                cp "$TEST_RESULTS_DIR/threshs.txt" "$METRICS_DIR/"
            fi
            
            # Extract metrics from the log file and create a summary
            LOG_FILE="$MODEL_DIR/latest.log"
            if [ -f "$LOG_FILE" ]; then
                echo "Extracting metrics from log file..."
                
                # Extract mAP lines and save to summary file
                grep -E "bbox_mAP" "$LOG_FILE" | tail -n 1 > "$METRICS_DIR/map_summary.txt"
                
                # Create a more readable summary file
                {
                    echo "===== Metrics Summary for $MODEL_NAME ====="
                    echo "Timestamp: $(date)"
                    echo ""
                    echo "=== Detection Performance ==="
                    grep -E "bbox_mAP" "$LOG_FILE" | tail -n 1 | sed 's/.*INFO - //'
                    echo ""
                    echo "=== Model Configuration ==="
                    echo "Model: $MODEL_NAME"
                    
                    # Try to extract epoch information
                    EPOCH_INFO=$(grep -E "Epoch\([^)]+\)" "$LOG_FILE" | tail -n 1 | sed 's/.*Epoch/Epoch/')
                    if [ -n "$EPOCH_INFO" ]; then
                        echo "Training progress: $EPOCH_INFO"
                    fi
                    
                    echo ""
                    echo "Full results available in: $TEST_RESULTS_DIR"
                } > "$METRICS_DIR/readable_summary.txt"
                
                echo "Metrics saved to: $METRICS_DIR/readable_summary.txt"
            else
                echo "Warning: Log file not found at $LOG_FILE"
            fi
            
            # Check for COCO evaluation JSON files
            for COCO_JSON in "$TEST_RESULTS_DIR"/*.bbox.json; do
                if [ -f "$COCO_JSON" ]; then
                    echo "Found COCO evaluation file: $(basename "$COCO_JSON")"
                    cp "$COCO_JSON" "$METRICS_DIR/"
                fi
            done
            
            echo "Metrics summary created at: $METRICS_DIR"
        else
            echo "No test results found for $MODEL_NAME"
        fi
    fi
done

echo "Metrics extraction completed!"