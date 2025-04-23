#!/bin/bash

# Script to automate training and testing of lg models for CVS detection
# Usage: ./run_train_test.sh [options]
#
# Options:
#   --mode [train|test|both]      : Specify whether to run training, testing, or both (default: both)
#   --model [faster|ds_faster|all]: Specify which model to run (default: all)
#   --epochs, --epoch N           : Number of epochs for training (default: 20)
#   --train-corruption TYPE       : Apply corruption during training (default: none)
#   --test-corruption TYPE        : Apply corruption during testing (default: none)
#   --help                        : Show this help message
#
# Corruption options: none, gaussian_noise, motion_blur, defocus_blur, uneven_illumination, smoke_effect, random_corruptions

# Default values
MODE="both"
MODEL="all"
EPOCHS=20
TRAIN_CORRUPTION="none"
TEST_CORRUPTION="none"
CPU_COUNT=$(nproc --all 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

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
        --epochs|--epoch)
            EPOCHS="$2"
            if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
                echo "Error: Epochs must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --train-corruption)
            TRAIN_CORRUPTION="$2"
            if [[ ! "$TRAIN_CORRUPTION" =~ ^(none|gaussian_noise|motion_blur|defocus_blur|uneven_illumination|smoke_effect|random_corruptions)$ ]]; then
                echo "Error: Train corruption must be one of: none, gaussian_noise, motion_blur, defocus_blur, uneven_illumination, smoke_effect, random_corruptions"
                exit 1
            fi
            shift 2
            ;;
        --test-corruption)
            TEST_CORRUPTION="$2"
            if [[ ! "$TEST_CORRUPTION" =~ ^(none|gaussian_noise|motion_blur|defocus_blur|uneven_illumination|smoke_effect|random_corruptions)$ ]]; then
                echo "Error: Test corruption must be one of: none, gaussian_noise, motion_blur, defocus_blur, uneven_illumination, smoke_effect, random_corruptions"
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
            echo "  --epochs, --epoch N            : Number of epochs for training (default: 20)"
            echo "  --train-corruption TYPE        : Apply corruption during training (default: none)"
            echo "  --test-corruption TYPE         : Apply corruption during testing (default: none)"
            echo ""
            echo "Corruption types:"
            echo "  none               : No corruption (clean data)"
            echo "  gaussian_noise     : Add Gaussian noise to images"
            echo "  motion_blur        : Apply motion blur effect"
            echo "  defocus_blur       : Apply defocus blur (Gaussian blur)"
            echo "  uneven_illumination: Apply uneven illumination effect"
            echo "  smoke_effect       : Apply smoke effect"
            echo "  random_corruptions : Randomly apply one of the above corruptions"
            echo ""
            echo "Examples:"
            echo "  ./run_train_test.sh --mode train --model faster --train-corruption none"
            echo "  ./run_train_test.sh --mode test --model all --test-corruption gaussian_noise"
            echo "  ./run_train_test.sh --mode both --epoch 10 --train-corruption none --test-corruption motion_blur"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create descriptive folder names based on training settings
if [[ "$TRAIN_CORRUPTION" == "none" ]]; then
    TRAIN_DESC="clean"
else
    TRAIN_DESC="${TRAIN_CORRUPTION}"
fi

# Create main results directory based on training settings
# Format: results/{epochs}_epoch_cpu{cpu_count}_{model}_{train_desc}
# This resembles the Cholec80 folder format like "11_epoch_w28"
if [[ "$MODEL" == "all" ]]; then
    MODEL_DESC="all_models"
else
    MODEL_DESC="${MODEL}"
fi

MAIN_RESULTS_DIR="results/${EPOCHS}_epoch_cpu${CPU_COUNT}_${MODEL_DESC}_${TRAIN_DESC}"
mkdir -p "${MAIN_RESULTS_DIR}"

# Create a symbolic link to the latest training results directory
ln -sf "${MAIN_RESULTS_DIR}" "results/latest_training"

# Test results will be in subdirectories named after the test corruption
if [[ "$TEST_CORRUPTION" == "none" ]]; then
    TEST_DIR_NAME="clean"
else
    TEST_DIR_NAME="${TEST_CORRUPTION}"
fi

# Create test results directory
TEST_RESULTS_DIR="${MAIN_RESULTS_DIR}/${TEST_DIR_NAME}_predicts"
mkdir -p "${TEST_RESULTS_DIR}"

# Create model-specific directories
if [[ "$MODEL" == "faster" || "$MODEL" == "all" ]]; then
    mkdir -p "${MAIN_RESULTS_DIR}/lg_faster_rcnn/checkpoints"
    mkdir -p "${TEST_RESULTS_DIR}/lg_faster_rcnn/metrics_summary"
fi
if [[ "$MODEL" == "ds_faster" || "$MODEL" == "all" ]]; then
    mkdir -p "${MAIN_RESULTS_DIR}/lg_ds_faster_rcnn/checkpoints"
    mkdir -p "${TEST_RESULTS_DIR}/lg_ds_faster_rcnn/metrics_summary"
fi

# Create ground truth directory for test data
mkdir -p "${TEST_RESULTS_DIR}/gt"

# Create weights directory if it doesn't exist
mkdir -p "weights/endoscapes/"

# Function to run training
run_training() {
    local model_type=$1
    local config_path="configs/models/faster_rcnn/lg_${model_type}_rcnn.py"
    local model_dir="${MAIN_RESULTS_DIR}/lg_${model_type}_rcnn"
    
    # For DS models, check if base model weights exist first
    if [[ "$model_type" == "ds_faster" ]]; then
        local base_weights_path="weights/endoscapes/lg_faster_rcnn.pth"
        if [[ ! -f "$base_weights_path" ]]; then
            echo "ERROR: Base model weights not found at $base_weights_path"
            echo "The DS-enhanced model requires weights from the base Faster R-CNN model."
            echo "Please train the base Faster R-CNN model first or obtain pre-trained weights."
            
            # Check if Faster R-CNN was already trained in this session
            local faster_ckpt_dir="${MAIN_RESULTS_DIR}/lg_faster_rcnn/checkpoints"
            local best_faster_ckpt=$(find "${faster_ckpt_dir}" -name "best_*.pth" 2>/dev/null | head -1)
            
            if [[ -n "$best_faster_ckpt" ]]; then
                echo "Found Faster R-CNN checkpoint from this session: $best_faster_ckpt"
                echo "Copying to weights directory for DS model to use..."
                cp "$best_faster_ckpt" "$base_weights_path"
                echo "Successfully copied weights. Proceeding with DS model training."
            else
                echo "No Faster R-CNN weights found. Cannot train DS model."
                return 1
            fi
        else
            echo "Found base model weights at $base_weights_path"
            echo "These weights will be used to initialize the DS model."
        fi
    fi
    
    echo "===================== TRAINING: lg_${model_type}_rcnn ====================="
    echo "Starting training with $EPOCHS epochs and corruption: $TRAIN_CORRUPTION"
    
    # Add corruption argument for training
    mim train mmdet ${config_path} \
        --cfg-options train_cfg.max_epochs=$EPOCHS \
        work_dir="${model_dir}" \
        default_hooks.checkpoint.out_dir="${model_dir}/checkpoints" \
        test_evaluator.outfile_prefix="${model_dir}/test_results" \
        corruption="${TRAIN_CORRUPTION}"
    
    local training_success=$?
    
    # If training was successful, copy the best checkpoint to the weights directory
    if [[ $training_success -eq 0 ]]; then
        local ckpt_dir="${model_dir}/checkpoints"
        local best_ckpt=$(find "${ckpt_dir}" -name "best_*.pth" | head -1)
        
        if [[ -n "$best_ckpt" ]]; then
            echo "Training successful. Copying best checkpoint to weights directory..."
            # Standard weights path without corruption suffix
            local weights_path="weights/endoscapes/lg_${model_type}_rcnn.pth"
            # Additional path with corruption info
            local corruption_weights_path="weights/endoscapes/lg_${model_type}_rcnn_train-${TRAIN_DESC}.pth"
            
            cp "$best_ckpt" "$weights_path"
            cp "$best_ckpt" "$corruption_weights_path"
            echo "Weights saved to:"
            echo "  - $weights_path (standard path for other models to use)"
            echo "  - $corruption_weights_path (with corruption information)"
        fi
    fi
    
    return $training_success
}

# Function to run testing
run_testing() {
    local model_type=$1
    local config_path="configs/models/faster_rcnn/lg_${model_type}_rcnn.py"
    local model_dir="${MAIN_RESULTS_DIR}/lg_${model_type}_rcnn"
    local test_model_dir="${TEST_RESULTS_DIR}/lg_${model_type}_rcnn"
    local ckpt_dir="${model_dir}/checkpoints"
    local metrics_dir="${test_model_dir}/metrics_summary"
    
    echo "===================== TESTING: lg_${model_type}_rcnn ====================="
    echo "Testing with corruption: $TEST_CORRUPTION"
    
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
        
        # Create test output directories
        mkdir -p "${metrics_dir}"
        
        # Create a file to directly capture the test output
        TEST_OUTPUT_FILE="${metrics_dir}/test_terminal_output.txt"
        
        # Add corruption argument for testing and save the raw output
        mim test mmdet ${config_path} \
            --checkpoint "$best_ckpt" \
            --cfg-options test_evaluator.outfile_prefix="${test_model_dir}" \
            corruption="${TEST_CORRUPTION}" \
            &> "${TEST_OUTPUT_FILE}" 
            
        # Also display the output in the terminal
        cat "${TEST_OUTPUT_FILE}"
        
        # Save a copy of raw output for debugging
        cp "${TEST_OUTPUT_FILE}" "${metrics_dir}/raw_test_output.txt"
        
        # Copy the best checkpoint to the weights directory with corruption info
        echo "Copying checkpoint to weights/endoscapes/"
        
        # Create a unique name for this model based on corruption settings
        if [[ "$TRAIN_CORRUPTION" == "none" ]]; then
            WEIGHT_SUFFIX="clean"
        else
            WEIGHT_SUFFIX="${TRAIN_CORRUPTION}"
        fi
        
        cp "$best_ckpt" "weights/endoscapes/lg_${model_type}_rcnn_train-${WEIGHT_SUFFIX}.pth"
        
        # Create metrics summary after testing
        extract_metrics "${test_model_dir}" "${TEST_OUTPUT_FILE}" "${TEST_CORRUPTION}"
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
    local test_corruption=$3
    local model_name=$(basename "${model_dir}")
    local metrics_dir="${model_dir}/metrics_summary"
    
    echo "===================== EXTRACTING METRICS: ${model_name} with ${test_corruption} ====================="
    
    if [ -d "${model_dir}" ]; then
        # Create directories if they don't exist
        mkdir -p "${metrics_dir}"
        
        # Create metrics summary file
        {
            echo "===== Metrics Summary for ${model_name} ====="
            echo "Timestamp: $(date)"
            echo ""
            echo "=== Test Configuration ==="
            echo "Train Corruption: ${TRAIN_CORRUPTION}"
            echo "Test Corruption: ${test_corruption}"
            echo ""
            echo "=== Test Performance Metrics ==="
            
            # Extract metrics from terminal output if available
            if [ -f "${terminal_output_file}" ]; then
                # Look for metrics in different formats
                echo "Analyzing test output for metrics..."
                
                # Try different patterns for metrics - this handles both mmdet output formats
                # Pattern 1: Standard mmdet format with Epoch(test)
                if grep -q "Epoch(test)" "${terminal_output_file}"; then
                    echo "Found metrics in standard mmdet format (Epoch(test))"
                    
                    # Extract the line with metrics
                    grep "Epoch(test)" "${terminal_output_file}" | tail -n 1 > "${metrics_dir}/raw_metrics.txt"
                    cat "${metrics_dir}/raw_metrics.txt"
                
                # Pattern 2: Alternative format with bbox_mAP
                elif grep -q "bbox_mAP" "${terminal_output_file}"; then
                    echo "Found metrics in alternative format (bbox_mAP)"
                    
                    # Extract lines with metrics
                    grep -A5 "bbox_mAP" "${terminal_output_file}" > "${metrics_dir}/raw_metrics.txt"
                    cat "${metrics_dir}/raw_metrics.txt"
                
                # Pattern 3: Look for any lines with metric values
                elif grep -q -E "[0-9]+\.[0-9]+" "${terminal_output_file}"; then
                    echo "Found numeric values that might be metrics"
                    
                    # Extract lines with numeric values (potential metrics)
                    grep -E "[0-9]+\.[0-9]+" "${terminal_output_file}" | head -20 > "${metrics_dir}/raw_metrics.txt"
                    cat "${metrics_dir}/raw_metrics.txt"
                
                else
                    echo "No recognizable metrics found in test output."
                    echo "See ${terminal_output_file} for raw test output."
                    # Save the first 50 lines as a sample
                    head -50 "${terminal_output_file}" > "${metrics_dir}/output_sample.txt"
                fi
                
                # Extract DS average precision metrics if they exist
                if grep -q "ds_average_precision" "${terminal_output_file}"; then
                    echo ""
                    echo "=== DS Average Precision Metrics ==="
                    grep -a -o -E "ds_average_precision[^[:space:]]*: [0-9.]+" "${terminal_output_file}" > "${metrics_dir}/ds_metrics.txt"
                    cat "${metrics_dir}/ds_metrics.txt"
                fi
                
                # Extract bbox mAP metrics if they exist
                if grep -q "bbox_mAP" "${terminal_output_file}"; then
                    echo ""
                    echo "=== Detection Performance ==="
                    grep -a -o -E "bbox_mAP[^[:space:]]*: [0-9.]+" "${terminal_output_file}" > "${metrics_dir}/bbox_metrics.txt"
                    cat "${metrics_dir}/bbox_metrics.txt"
                fi
            else
                echo "No terminal output file found at ${terminal_output_file}"
            fi
            
            echo ""
            echo "=== Model Configuration ==="
            echo "Model: ${model_name}"
            echo "Epochs trained: ${EPOCHS}"
            echo "Checkpoint: $(basename "$(readlink -f "weights/endoscapes/lg_${model_name}_train-${TRAIN_DESC}.pth" 2>/dev/null || echo "not found")")"
            
            echo ""
            echo "Full results available in: ${model_dir}"
        } > "${metrics_dir}/complete_metrics.txt"
        
        # Create a minimal results file (similar to your Cholec80 example)
        {
            echo "===== Evaluation Results for ${model_name} ====="
            echo "Train Setting: ${EPOCHS}_epoch_cpu${CPU_COUNT}_${TRAIN_DESC}"
            echo "Test Setting: ${test_corruption}"
            echo ""
            
            # Extract metrics for CSV - more flexible regex patterns
            if [ -f "${terminal_output_file}" ]; then
                # Look for bbox_mAP with more flexible pattern matching
                BBOX_MAP=$(grep -ao -E "bbox_mAP.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                BBOX_MAP_50=$(grep -ao -E "bbox_mAP_50.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                BBOX_MAP_75=$(grep -ao -E "bbox_mAP_75.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                
                # Look for ds_average_precision with more flexible pattern matching
                DS_AP=$(grep -ao -E "ds_average_precision.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                DS_AP_C1=$(grep -ao -E "ds_average_precision_C1.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                DS_AP_C2=$(grep -ao -E "ds_average_precision_C2.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                DS_AP_C3=$(grep -ao -E "ds_average_precision_C3.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                
                # Print metrics if found
                if [ -n "$DS_AP" ]; then
                    echo "Overall DS Average Precision: $DS_AP"
                    [ -n "$DS_AP_C1" ] && echo "Class 1 DS AP: $DS_AP_C1"
                    [ -n "$DS_AP_C2" ] && echo "Class 2 DS AP: $DS_AP_C2"
                    [ -n "$DS_AP_C3" ] && echo "Class 3 DS AP: $DS_AP_C3"
                fi
                
                if [ -n "$BBOX_MAP" ]; then
                    echo "Detection mAP: $BBOX_MAP"
                    [ -n "$BBOX_MAP_50" ] && echo "Detection mAP@0.5: $BBOX_MAP_50"
                    [ -n "$BBOX_MAP_75" ] && echo "Detection mAP@0.75: $BBOX_MAP_75"
                fi
                
                # If no metrics were found
                if [ -z "$DS_AP" ] && [ -z "$BBOX_MAP" ]; then
                    echo "No metrics available. Check raw test output for details."
                fi
            else
                echo "No test output file found."
            fi
        } > "${model_dir}/eval_results.txt"
        
        # Create a simplified metrics CSV for this test
        {
            echo "Metric,Value"
            
            # More flexible extraction of metrics for CSV
            if [ -f "${terminal_output_file}" ]; then
                # Get bbox metrics with more flexible pattern matching
                BBOX_MAP=$(grep -ao -E "bbox_mAP.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                BBOX_MAP_50=$(grep -ao -E "bbox_mAP_50.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                BBOX_MAP_75=$(grep -ao -E "bbox_mAP_75.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                
                # Get ds metrics with more flexible pattern matching
                DS_AP=$(grep -ao -E "ds_average_precision.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                DS_AP_C1=$(grep -ao -E "ds_average_precision_C1.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                DS_AP_C2=$(grep -ao -E "ds_average_precision_C2.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                DS_AP_C3=$(grep -ao -E "ds_average_precision_C3.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1)
                
                # Write all available metrics to CSV
                [ -n "$DS_AP" ] && echo "DS_Average_Precision,$DS_AP"
                [ -n "$DS_AP_C1" ] && echo "DS_AP_Class1,$DS_AP_C1"
                [ -n "$DS_AP_C2" ] && echo "DS_AP_Class2,$DS_AP_C2" 
                [ -n "$DS_AP_C3" ] && echo "DS_AP_Class3,$DS_AP_C3"
                [ -n "$BBOX_MAP" ] && echo "Detection_mAP,$BBOX_MAP"
                [ -n "$BBOX_MAP_50" ] && echo "Detection_mAP_50,$BBOX_MAP_50" 
                [ -n "$BBOX_MAP_75" ] && echo "Detection_mAP_75,$BBOX_MAP_75"
                
                # Add run configuration 
                echo "Epochs,$EPOCHS"
                echo "Train_Corruption,$TRAIN_CORRUPTION"
                echo "Test_Corruption,$test_corruption"
                echo "CPU_Count,$CPU_COUNT"
                echo "Timestamp,$(date +%Y%m%d-%H%M%S)"
            else
                echo "Error,No test output file found"
            fi
        } > "${model_dir}/metrics.csv"
        
        # Update the global comparison CSV file
        {
            # CSV header (only if file doesn't exist)
            if [ ! -f "results/corruption_comparison.csv" ]; then
                echo "model,train_corruption,test_corruption,epochs,cpu_count,timestamp,ds_ap,ds_ap_c1,ds_ap_c2,ds_ap_c3,bbox_map,bbox_map50,bbox_map75" > "results/corruption_comparison.csv"
            fi
            
            # Extract metrics with more flexible regex
            if [ -f "${terminal_output_file}" ]; then
                # Get all metrics with more flexible pattern matching
                DS_AP=$(grep -ao -E "ds_average_precision.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1 || echo "NA")
                DS_AP_C1=$(grep -ao -E "ds_average_precision_C1.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1 || echo "NA")
                DS_AP_C2=$(grep -ao -E "ds_average_precision_C2.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1 || echo "NA")
                DS_AP_C3=$(grep -ao -E "ds_average_precision_C3.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1 || echo "NA")
                
                BBOX_MAP=$(grep -ao -E "bbox_mAP.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1 || echo "NA")
                BBOX_MAP_50=$(grep -ao -E "bbox_mAP_50.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1 || echo "NA")
                BBOX_MAP_75=$(grep -ao -E "bbox_mAP_75.*?[0-9]+\.[0-9]+" "${terminal_output_file}" | grep -o -E "[0-9]+\.[0-9]+" | head -1 || echo "NA")
                
                # Add entry to CSV
                echo "${model_name},${TRAIN_CORRUPTION},${test_corruption},${EPOCHS},${CPU_COUNT},$(date +"%Y%m%d-%H%M%S"),${DS_AP},${DS_AP_C1},${DS_AP_C2},${DS_AP_C3},${BBOX_MAP},${BBOX_MAP_50},${BBOX_MAP_75}" >> "results/corruption_comparison.csv"
            fi
        }
        
        echo "Metrics files created:"
        echo "- ${model_dir}/eval_results.txt (summary report)"
        echo "- ${model_dir}/metrics.csv (CSV format)"
        echo "- ${metrics_dir}/complete_metrics.txt (detailed report)"
        echo "- Updated results/corruption_comparison.csv with new entry"
    else
        echo "No model directory found for ${model_name}"
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
echo ""
echo "Training Results Directory: ${MAIN_RESULTS_DIR}"
echo "  - Train corruption: ${TRAIN_CORRUPTION}"
echo "  - Epochs: ${EPOCHS}"
echo ""
echo "Testing Results Directory: ${TEST_RESULTS_DIR}"
echo "  - Test corruption: ${TEST_CORRUPTION}"
echo ""
echo "Results Summary:"
echo "  - Checkpoints saved to: ${MAIN_RESULTS_DIR}/*/checkpoints/"
echo "  - Metrics available in: ${TEST_RESULTS_DIR}/*/eval_results.txt"
echo "  - Consolidated metrics: results/corruption_comparison.csv"
echo "====================================================================="