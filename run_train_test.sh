#!/bin/bash

# Script to automate training and testing of lg models for CVS detection
# Usage: ./run_train_test.sh [options]
#
# Options:
#   --mode [train|test|both]      : Specify whether to run training, testing, or both (default: both)
#   --model [faster|ds_faster|all]: Specify which model to run (default: all)
#   --epochs N                    : Number of epochs for training (default: 20)
#   --help                        : Show this help message

# Default values
MODE="both"
MODEL="all"
EPOCHS=20
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="results/endoscapes_training_${TIMESTAMP}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --mode)
            MODE="$2"
            if [[ ! "$MODE" =~ ^(train|test|both)$ ]]; then
                echo "Error: Mode must be 'train', 'test', or 'both'"
                exit 1
            fi
            shift 2
            ;;
        --model)
            MODEL="$2"
            if [[ ! "$MODEL" =~ ^(faster|ds_faster|all)$ ]]; then
                echo "Error: Model must be 'faster', 'ds_faster', or 'all'"
                exit 1
            fi
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
                echo "Error: Epochs must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --help)
            echo "Usage: ./run_train_test.sh [options]"
            echo ""
            echo "Options:"
            echo "  --mode [train|test|both]       : Specify whether to run training, testing, or both (default: both)"
            echo "  --model [faster|ds_faster|all] : Specify which model to run (default: all)"
            echo "  --epochs N                     : Number of epochs for training (default: 20)"
            echo "  --help                         : Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create result directories
mkdir -p "${RESULTS_DIR}"
if [[ "$MODEL" == "faster" || "$MODEL" == "all" ]]; then
    mkdir -p "${RESULTS_DIR}/lg_faster_rcnn/checkpoints"
    mkdir -p "${RESULTS_DIR}/lg_faster_rcnn/test_results/metrics_summary"
fi
if [[ "$MODEL" == "ds_faster" || "$MODEL" == "all" ]]; then
    mkdir -p "${RESULTS_DIR}/lg_ds_faster_rcnn/checkpoints"
    mkdir -p "${RESULTS_DIR}/lg_ds_faster_rcnn/test_results/metrics_summary"
fi
mkdir -p "weights/endoscapes/"

# Create a symbolic link to the latest results directory
ln -sf "${RESULTS_DIR}" "results/latest"

# Function to run training
run_training() {
    local model_type=$1
    local config_path="configs/models/faster_rcnn/lg_${model_type}_rcnn.py"
    local model_dir="${RESULTS_DIR}/lg_${model_type}_rcnn"
    
    echo "===================== TRAINING: lg_${model_type}_rcnn ====================="
    echo "Starting training with $EPOCHS epochs..."
    
    mim train mmdet ${config_path} \
        --cfg-options train_cfg.max_epochs=$EPOCHS \
        work_dir="${model_dir}" \
        default_hooks.checkpoint.out_dir="${model_dir}/checkpoints" \
        test_evaluator.outfile_prefix="${model_dir}/test_results"
}

# Function to run testing
run_testing() {
    local model_type=$1
    local config_path="configs/models/faster_rcnn/lg_${model_type}_rcnn.py"
    local model_dir="${RESULTS_DIR}/lg_${model_type}_rcnn"
    local ckpt_dir="${model_dir}/checkpoints"
    local metrics_dir="${model_dir}/test_results/metrics_summary"
    
    echo "===================== TESTING: lg_${model_type}_rcnn ====================="
    
    # Find the best checkpoint
    local best_ckpt=$(find "${ckpt_dir}" -name "best_*.pth" | head -1)
    
    if [ -z "$best_ckpt" ]; then
        # If no best checkpoint, use the latest epoch
        best_ckpt=$(find "${ckpt_dir}" -name "epoch_*.pth" | sort -r | head -1)
    fi
    
    # If still no checkpoint and we're only testing, try the weights directory
    if [ -z "$best_ckpt" ] && [ "$MODE" == "test" ]; then
        if [ -f "weights/endoscapes/lg_${model_type}_rcnn.pth" ]; then
            best_ckpt="weights/endoscapes/lg_${model_type}_rcnn.pth"
        fi
    fi
    
    # Run testing with the best checkpoint
    if [ -n "$best_ckpt" ]; then
        echo "Found checkpoint: $best_ckpt"
        echo "Running test with checkpoint..."
        
        # Create a file to directly capture the test output
        TEST_OUTPUT_FILE="${metrics_dir}/test_terminal_output.txt"
        
        # Run the test command and tee the output to capture it
        mim test mmdet ${config_path} \
            --checkpoint "$best_ckpt" \
            --cfg-options test_evaluator.outfile_prefix="${model_dir}/test_results" \
            | tee "${TEST_OUTPUT_FILE}"
        
        # Copy the best checkpoint to the weights directory
        echo "Copying checkpoint to weights/endoscapes/"
        cp "$best_ckpt" "weights/endoscapes/lg_${model_type}_rcnn.pth"
        
        # Create metrics summary after testing
        extract_metrics "${model_dir}" "${TEST_OUTPUT_FILE}"
    else
        echo "No checkpoint found for testing lg_${model_type}_rcnn"
        if [ "$MODE" == "both" ]; then
            echo "This is unexpected as we just trained the model. Please check the logs."
        fi
        return 1
    fi
}

# Function to extract and organize test metrics
extract_metrics() {
    local model_dir=$1
    local terminal_output_file=$2
    local model_name=$(basename "${model_dir}")
    local test_results_dir="${model_dir}/test_results"
    local metrics_dir="${test_results_dir}/metrics_summary"
    
    echo "===================== EXTRACTING METRICS: ${model_name} ====================="
    
    if [ -d "${test_results_dir}" ]; then
        # Create metrics summary file (keeping the existing format)
        {
            echo "===== Metrics Summary for ${model_name} ====="
            echo "Timestamp: $(date)"
            echo ""
            echo "=== Test Performance Metrics ==="
            
            # Extract metrics from terminal output if available
            if [ -f "${terminal_output_file}" ]; then
                # Look for test results line with metrics
                if grep -q "Epoch(test)" "${terminal_output_file}"; then
                    # Get the last line with test metrics
                    grep "Epoch(test)" "${terminal_output_file}" | tail -n 1 | sed 's/.*INFO - //' >> "${metrics_dir}/raw_metrics.txt"
                    cat "${metrics_dir}/raw_metrics.txt"
                    
                    # Extract DS average precision metrics if they exist
                    if grep -q "ds_average_precision" "${terminal_output_file}"; then
                        echo ""
                        echo "=== DS Average Precision Metrics ==="
                        grep -o "endoscapes/ds_average_precision[^[:space:]]*: [0-9.]*" "${terminal_output_file}" > "${metrics_dir}/ds_metrics.txt"
                        cat "${metrics_dir}/ds_metrics.txt"
                    fi
                    
                    # Extract bbox mAP metrics if they exist
                    if grep -q "bbox_mAP" "${terminal_output_file}"; then
                        echo ""
                        echo "=== Detection Performance ==="
                        grep -o "endoscapes/bbox_mAP[^[:space:]]*: [0-9.]*" "${terminal_output_file}" > "${metrics_dir}/bbox_metrics.txt"
                        cat "${metrics_dir}/bbox_metrics.txt"
                    fi
                else
                    echo "No test metrics found in terminal output."
                fi
            else
                echo "No terminal output file found."
            fi
            
            echo ""
            echo "=== Model Configuration ==="
            echo "Model: ${model_name}"
            echo "Epochs trained: ${EPOCHS}"
            echo "Checkpoint: $(basename "$(readlink -f "weights/endoscapes/lg_${model_name}.pth" 2>/dev/null || echo "not found")")"
            
            echo ""
            echo "Full results available in: ${test_results_dir}"
        } > "${metrics_dir}/complete_metrics.txt"
        
        # Create a tabular format metrics file
        {
            echo "===== Performance Metrics for ${model_name} ====="
            echo "Timestamp: $(date +"%Y-%m-%d %H:%M:%S")"
            echo ""
            echo "┌───────────────────────────────────┬─────────┐"
            echo "│ Metric                            │   Value │"
            echo "├───────────────────────────────────┼─────────┤"
            
            # Extract metrics for tabular format
            if [ -f "${terminal_output_file}" ]; then
                # Look for test results line with metrics
                if grep -q "Epoch(test)" "${terminal_output_file}"; then
                    # Get the last line with test metrics
                    METRICS_LINE=$(grep "Epoch(test)" "${terminal_output_file}" | tail -n 1)
                    
                    # DS average precision metrics
                    if echo "$METRICS_LINE" | grep -q "ds_average_precision"; then
                        # Extract overall DS AP
                        DS_AP=$(echo "$METRICS_LINE" | grep -o "endoscapes/ds_average_precision: [0-9.]*" | grep -o "[0-9.]*")
                        printf "│ %-35s │ %7.4f │\n" "DS Average Precision" "$DS_AP"
                        echo "├───────────────────────────────────┼─────────┤"
                        
                        # Extract class-specific DS AP metrics
                        DS_AP_C1=$(echo "$METRICS_LINE" | grep -o "endoscapes/ds_average_precision_C1: [0-9.]*" | grep -o "[0-9.]*")
                        DS_AP_C2=$(echo "$METRICS_LINE" | grep -o "endoscapes/ds_average_precision_C2: [0-9.]*" | grep -o "[0-9.]*")
                        DS_AP_C3=$(echo "$METRICS_LINE" | grep -o "endoscapes/ds_average_precision_C3: [0-9.]*" | grep -o "[0-9.]*")
                        
                        [ -n "$DS_AP_C1" ] && printf "│ %-35s │ %7.4f │\n" "DS AP - Class 1" "$DS_AP_C1"
                        [ -n "$DS_AP_C2" ] && printf "│ %-35s │ %7.4f │\n" "DS AP - Class 2" "$DS_AP_C2"
                        [ -n "$DS_AP_C3" ] && printf "│ %-35s │ %7.4f │\n" "DS AP - Class 3" "$DS_AP_C3"
                    fi
                    
                    # BBOX mAP metrics
                    if echo "$METRICS_LINE" | grep -q "bbox_mAP"; then
                        # Header separator if DS metrics were already added
                        if ! echo "$METRICS_LINE" | grep -q "ds_average_precision"; then
                            echo "├───────────────────────────────────┼─────────┤"
                        fi
                        
                        # Extract overall mAP
                        BBOX_MAP=$(echo "$METRICS_LINE" | grep -o "endoscapes/bbox_mAP: [0-9.]*" | grep -o "[0-9.]*")
                        printf "│ %-35s │ %7.4f │\n" "Detection mAP" "$BBOX_MAP"
                        
                        # Extract IoU-specific mAP metrics
                        BBOX_MAP_50=$(echo "$METRICS_LINE" | grep -o "endoscapes/bbox_mAP_50: [0-9.]*" | grep -o "[0-9.]*")
                        BBOX_MAP_75=$(echo "$METRICS_LINE" | grep -o "endoscapes/bbox_mAP_75: [0-9.]*" | grep -o "[0-9.]*")
                        
                        [ -n "$BBOX_MAP_50" ] && printf "│ %-35s │ %7.4f │\n" "Detection mAP@0.5" "$BBOX_MAP_50"
                        [ -n "$BBOX_MAP_75" ] && printf "│ %-35s │ %7.4f │\n" "Detection mAP@0.75" "$BBOX_MAP_75"
                    fi
                    
                    # Add timing information
                    DATA_TIME=$(echo "$METRICS_LINE" | grep -o "data_time: [0-9.]*" | grep -o "[0-9.]*")
                    PROC_TIME=$(echo "$METRICS_LINE" | grep -o "time: [0-9.]*" | grep -o "[0-9.]*")
                    
                    echo "├───────────────────────────────────┼─────────┤"
                    [ -n "$DATA_TIME" ] && printf "│ %-35s │ %7.4f │\n" "Data loading time (s)" "$DATA_TIME"
                    [ -n "$PROC_TIME" ] && printf "│ %-35s │ %7.4f │\n" "Processing time (s)" "$PROC_TIME"
                else
                    printf "│ %-35s │ %7s │\n" "No test metrics found" "N/A"
                fi
            else
                printf "│ %-35s │ %7s │\n" "No terminal output found" "N/A"
            fi
            
            echo "└───────────────────────────────────┴─────────┘"
            echo ""
            echo "Model: ${model_name}"
            echo "Epochs: ${EPOCHS}"
            echo "Date: $(date +"%Y-%m-%d")"
        } > "${metrics_dir}/tabular_metrics.txt"
        
        # Copy tabular metrics directly to the test_results folder for easier access
        cp "${metrics_dir}/tabular_metrics.txt" "${test_results_dir}/"
        
        echo "Metrics summaries created:"
        echo "- ${metrics_dir}/complete_metrics.txt (detailed metrics)"
        echo "- ${test_results_dir}/tabular_metrics.txt (tabular format)"
    else
        echo "No test results found for ${model_name}"
    fi
}

# Run the faster_rcnn model if requested
if [[ "$MODEL" == "faster" || "$MODEL" == "all" ]]; then
    if [[ "$MODE" == "train" || "$MODE" == "both" ]]; then
        run_training "faster"
    fi
    
    if [[ "$MODE" == "test" || "$MODE" == "both" ]]; then
        run_testing "faster"
    fi
fi

# Run the ds_faster_rcnn model if requested
if [[ "$MODEL" == "ds_faster" || "$MODEL" == "all" ]]; then
    if [[ "$MODE" == "train" || "$MODE" == "both" ]]; then
        run_training "ds_faster"
    fi
    
    if [[ "$MODE" == "test" || "$MODE" == "both" ]]; then
        run_testing "ds_faster"
    fi
fi

echo "====================================================================="
echo "Operations completed successfully."
echo "Results are saved in: ${RESULTS_DIR}"
echo "Best checkpoints are copied to weights/endoscapes/"
echo "Metrics summaries are available in: ${RESULTS_DIR}/*/test_results/"
echo "====================================================================="