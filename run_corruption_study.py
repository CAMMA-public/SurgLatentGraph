#!/usr/bin/env python
# Runner script for the complete CVS detection corruption robustness pipeline

import os
import argparse
import subprocess
import datetime
import shutil
import json
import glob
from pathlib import Path

def is_docker():
    """Check if we're running inside a Docker container."""
    return os.path.exists('/.dockerenv')

def get_base_directories():
    """Get base directories for host and container environments."""
    host_base = '/home/santhi/Documents/SurgLatentGraph'
    container_base = '/workspace'
    return host_base, container_base

def normalize_path(path):
    """Normalize path regardless of whether it's relative or absolute."""
    if os.path.isabs(path):
        return path
    
    # Get script directory to construct absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, path))

def convert_path_between_environments(path):
    """Convert paths between host and container environments."""
    host_base, container_base = get_base_directories()
    
    # Normalize path
    norm_path = normalize_path(path)
    
    if is_docker():
        # Running in Docker container, convert host path to container path
        if host_base in norm_path:
            return norm_path.replace(host_base, container_base)
    else:
        # Running on host, convert container path to host path
        if container_base in norm_path:
            return norm_path.replace(container_base, host_base)
    
    # Path doesn't need conversion or can't be converted
    return norm_path

def find_file_with_fallbacks(file_path, file_type="file"):
    """
    Attempt to find a file, checking multiple path variations.
    Returns the valid path if found, otherwise None.
    """
    # Step 1: Check if the direct path exists
    if os.path.exists(file_path):
        print(f"Found {file_type} at original path: {file_path}")
        return file_path
    
    # Step 2: Try normalizing the path
    norm_path = normalize_path(file_path)
    if norm_path != file_path and os.path.exists(norm_path):
        print(f"Found {file_type} at normalized path: {norm_path}")
        return norm_path
    
    # Step 3: Try converted path (between host and container)
    converted_path = convert_path_between_environments(file_path)
    if converted_path != file_path and os.path.exists(converted_path):
        print(f"Found {file_type} at converted path: {converted_path}")
        return converted_path
    
    # Step 4: If the path contains a filename that might be somewhere else, search for it
    filename = os.path.basename(file_path)
    if len(filename) > 0 and '.' in filename:  # Only search for files with extensions
        host_base, container_base = get_base_directories()
        search_bases = [os.getcwd(), host_base, container_base] 
        
        for base in search_bases:
            if not os.path.exists(base):
                continue
                
            # Try to find the file in this base directory and its subdirectories
            for root, _, files in os.walk(base):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    print(f"Found {file_type} by searching: {found_path}")
                    return found_path
        
        # For checkpoint files, try a more specific approach using glob patterns
        if file_type == "checkpoint" and filename.endswith('.pth'):
            # Check weights directory specifically
            weights_dirs = [
                os.path.join(base, 'weights') for base in search_bases
                if os.path.exists(os.path.join(base, 'weights'))
            ]
            
            # Extract key part of filename for partial matching
            name_parts = filename.split('_')
            if len(name_parts) > 1:
                model_type = name_parts[0]  # e.g., "faster" from "faster_rcnn_..."
                
                for weights_dir in weights_dirs:
                    # Try pattern matching with glob
                    pattern = os.path.join(weights_dir, f"{model_type}*.pth")
                    matches = glob.glob(pattern)
                    
                    if matches:
                        print(f"Found similar {file_type} by pattern matching: {matches[0]}")
                        return matches[0]
    
    # File not found after all attempts
    print(f"ERROR: Could not find {file_type} at: {file_path}")
    print(f"Tried paths: {file_path}, {norm_path}, {converted_path}")
    return None

def parse_args():
    parser = argparse.ArgumentParser(description='Run the complete corruption robustness evaluation pipeline')
    
    # Main parameters
    parser.add_argument('--config', default='configs/models/faster_rcnn/lg_faster_rcnn.py', 
                      help='Path to the model config file')
    parser.add_argument('--results-dir', default='/workspace/results',
                      help='Directory to store all results')
    parser.add_argument('--experiment-name', default=None,
                      help='Name for this experiment (default: auto-generated timestamp)')
    
    # Dataset selection
    parser.add_argument('--dataset', default='endoscapes',
                      choices=['endoscapes', 'cholecT50', 'c80_phase'],
                      help='Dataset to use')
    
    # Corruption options with separate args for train/predict
    parser.add_argument('--train-corruptions', nargs='+',
                      default=['none'],
                      help='List of corruptions to train with (default: none). Options include: none, gaussian_noise, motion_blur, defocus_blur, uneven_illumination, smoke_effect, random_corruptions')
    parser.add_argument('--predict-corruptions', nargs='+',
                      default=['none'],
                      help='List of corruptions to evaluate/predict on (default: none)')
    parser.add_argument('--corruption-severity', type=float, default=0.5,
                      help='Severity of corruptions (0.0-1.0)')
    
    # Task selection
    parser.add_argument('--task', default='all',
                      choices=['train', 'predict', 'evaluate', 'analyze', 'all'],
                      help='Task(s) to run')
    
    # Device options
    parser.add_argument('--device', default='cuda:0',
                      help='Device for training/inference')
    
    # Options for specific stages
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs for training')
    parser.add_argument('--metrics', nargs='+', default=['bbox', 'segm'],
                      choices=['bbox', 'segm', 'proposal'],
                      help='Metrics to evaluate')
    
    # Checkpoint selection (direct checkpoint path override)
    parser.add_argument('--checkpoint', default=None,
                      help='Direct path to a checkpoint file to use for prediction (bypasses automatic checkpoint finding)')
    parser.add_argument('--checkpoint-corruption', default=None,
                      help='Name of corruption used to train the checkpoint (for labeling purposes)')
    
    return parser.parse_args()

def setup_experiment_directory(args):
    """Set up experiment directory structure."""
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.experiment_name = f"{args.dataset}_{Path(args.config).stem}_{timestamp}"
    
    # Create main experiment directory with adjusted path if needed
    if is_docker() and not args.results_dir.startswith('/workspace'):
        args.results_dir = '/workspace/results'
        print(f"Adjusting results directory for Docker: {args.results_dir}")
    
    experiment_dir = os.path.join(args.results_dir, args.experiment_name)
    print(f"Creating experiment directory at: {experiment_dir}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories for each stage with explicit permissions
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    predictions_dir = os.path.join(experiment_dir, "predictions")
    evaluation_dir = os.path.join(experiment_dir, "evaluation")
    statistics_dir = os.path.join(experiment_dir, "statistics")
    outputs_dir = os.path.join(experiment_dir, "outputs")  # New directory for general outputs
    
    # Create directories with appropriate permissions
    for directory in [checkpoints_dir, predictions_dir, evaluation_dir, statistics_dir, outputs_dir]:
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)
        # Set permissions to 777 to ensure visibility outside the container
        os.chmod(directory, 0o777)
    
    # Save experiment configuration
    config = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config_file": args.config,
        "dataset": args.dataset,
        "train_corruptions": args.train_corruptions,
        "predict_corruptions": args.predict_corruptions,
        "corruption_severity": args.corruption_severity,
        "command_line_args": vars(args)
    }
    
    config_file = os.path.join(experiment_dir, "experiment_config.json")
    print(f"Saving experiment configuration to: {config_file}")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set permissions on the config file
    os.chmod(config_file, 0o666)
    
    # Also set permissions on experiment_dir itself
    os.chmod(experiment_dir, 0o777)
    
    return {
        "root": experiment_dir,
        "checkpoints": checkpoints_dir,
        "predictions": predictions_dir,
        "evaluation": evaluation_dir,
        "statistics": statistics_dir,
        "outputs": outputs_dir
    }

def run_dataset_selection(args):
    """Run the dataset selection script."""
    print("\n=== SELECTING DATASET ===")
    
    # Determine script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create all possible locations for the models directory
    possible_model_dirs = [
        os.path.join(base_dir, "configs/models"),
        "/workspace/configs/models",
        "/home/santhi/Documents/SurgLatentGraph/configs/models"
    ]
    
    # Find first existing models directory
    models_dir = None
    for dir_path in possible_model_dirs:
        if os.path.exists(dir_path):
            models_dir = dir_path
            print(f"Found models directory at: {models_dir}")
            break
    
    if not models_dir:
        print(f"ERROR: Could not find models directory in any of the expected locations")
        for dir_path in possible_model_dirs:
            print(f"- {dir_path}")
        return False
    
    # Look for select_dataset.sh
    select_script = os.path.join(models_dir, "select_dataset.sh")
    if not os.path.exists(select_script):
        print(f"ERROR: select_dataset.sh not found at: {select_script}")
        return False
    
    # Make sure the script is executable
    os.chmod(select_script, 0o775)
    
    # Run the script from its directory to avoid path issues
    current_dir = os.getcwd()
    try:
        os.chdir(models_dir)
        cmd = f"./select_dataset.sh {args.dataset}"
        print(f"Running: {cmd} in {models_dir}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Print output for debugging
        print(f"Command output:\n{result.stdout}")
        if result.stderr:
            print(f"Command error:\n{result.stderr}")
        
        # Check if command succeeded
        if result.returncode != 0:
            print(f"WARNING: Dataset selection returned exit code {result.returncode}")
            print(f"This may not be critical, continuing with the pipeline...")
            # Return True anyway since this isn't necessarily fatal
            return True
        return True
    except Exception as e:
        print(f"Exception during dataset selection: {e}")
        print(f"This may not be critical, continuing with the pipeline...")
        return True  # Continue despite errors
    finally:
        # Restore original directory
        os.chdir(current_dir)

def run_training(args, directories):
    """
    Run training for baseline and corrupted models.
    - For clean (none) training, creates a directory structure: /checkpoints/clean/
    - For corrupted training, creates: /checkpoints/{corruption_name}/
    - Ensures checkpoint paths reflect experiment name and corruption type
    """
    print("\n=== TRAINING MODELS ===")
    print(f"Training with corruptions: {args.train_corruptions}")
    
    # Find training script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = find_file_with_fallbacks(os.path.join(base_dir, "train_endoscapes.py"), "train script")
    
    if not train_script:
        print("ERROR: Could not find train_endoscapes.py script")
        return False
        
    # Find config file
    config_path = find_file_with_fallbacks(args.config, "config file")
    
    if not config_path:
        print(f"ERROR: Could not find config file: {args.config}")
        return False
    
    # Train models with each specified corruption (including 'none')
    for corruption in args.train_corruptions:
        # Create a checkpoint subdirectory that reflects both experiment name and corruption type
        if corruption == 'none':
            # For 'none' corruption, use the 'clean' directory
            checkpoint_subdir = "clean"
        else:
            # For actual corruptions, use directories named by the corruption type
            checkpoint_subdir = corruption
        
        work_dir = os.path.join(directories["checkpoints"], checkpoint_subdir)
        os.makedirs(work_dir, exist_ok=True)
        
        # Build the training command
        python_path = "python" if not is_docker() else "/opt/conda/envs/latentgraph/bin/python"
        
        train_cmd = f"{python_path} {train_script} {config_path} " \
                    f"--work-dir {work_dir} " \
                    f"--corruption {corruption} " \
                    f"--corruption-severity {args.corruption_severity} " \
                    f"--device {args.device} " \
                    f"--seed 42 " \
                    f"--max-epochs {args.epochs}"
        
        print(f"Training model with {corruption} corruption: {train_cmd}")
        print(f"Checkpoints will be saved to: {work_dir}/corruption_{corruption}")
        
        # Run training command
        result = subprocess.run(train_cmd, shell=True, capture_output=True, text=True)
        
        # Print output for debugging
        print(f"\nTraining command output for {corruption} corruption:")
        print(f"Return code: {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        # Save output to file for later analysis
        output_file = os.path.join(work_dir, f"training_output_{corruption}.txt")
        with open(output_file, 'w') as f:
            f.write(f"Training command: {train_cmd}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
        
        # Set permissions on output file
        os.chmod(output_file, 0o666)
    
    return True

def run_prediction(args, directories):
    """
    Run prediction for all trained models on all corruptions.
    Handles direct checkpoint specification or finds best checkpoints automatically.
    """
    print("\n=== GENERATING PREDICTIONS ===")
    
    # Check if a specific checkpoint was provided
    if args.checkpoint:
        print(f"Using specified checkpoint: {args.checkpoint}")
        
        # Find checkpoint with robust path handling
        checkpoint_path = find_file_with_fallbacks(args.checkpoint, "checkpoint")
        
        if not checkpoint_path:
            print(f"ERROR: Cannot find checkpoint: {args.checkpoint}")
            
            # Check if the name is in the weights directory
            weights_dir = "/workspace/weights" if is_docker() else "/home/santhi/Documents/SurgLatentGraph/weights"
            
            if os.path.exists(weights_dir):
                print(f"Listing available checkpoints in weights directory:")
                for file in os.listdir(weights_dir):
                    if file.endswith('.pth'):
                        print(f"  - {file}")
            
            # Create error file with detailed information
            error_file = os.path.join(directories['predictions'], "CHECKPOINT_NOT_FOUND.txt")
            with open(error_file, 'w') as f:
                f.write(f"Checkpoint not found at {datetime.datetime.now()}\n")
                f.write(f"Searched for: {args.checkpoint}\n")
                
                # Provide suggestions
                if args.checkpoint.startswith("weights/"):
                    host_path = os.path.join("/home/santhi/Documents/SurgLatentGraph", args.checkpoint)
                    container_path = os.path.join("/workspace", args.checkpoint)
                    f.write(f"Try using full paths instead:\n")
                    f.write(f"Host path: {host_path}\n")
                    f.write(f"Container path: {container_path}\n")
            
            return False
        
        if checkpoint_path != args.checkpoint:
            print(f"Checkpoint path adjusted from '{args.checkpoint}' to '{checkpoint_path}'")
        
        # Use the specified checkpoint with the provided corruption label (or 'manual_checkpoint' if none provided)
        args.checkpoint = checkpoint_path  # Update with resolved path
        train_condition = args.checkpoint_corruption or 'manual_checkpoint'
        checkpoints = {train_condition: checkpoint_path}
    else:
        # Find best checkpoints based on trained corruption models
        print("Finding best checkpoints for each trained model...")
        checkpoints = {}
        checkpoints_dir = directories["checkpoints"]
        
        # Check for clean model (none corruption) checkpoint
        if 'none' in args.train_corruptions:
            clean_dir = os.path.join(checkpoints_dir, "clean")
            try:
                clean_checkpoint = find_best_checkpoint(clean_dir)
                checkpoints["clean"] = clean_checkpoint
                print(f"Found clean model checkpoint: {clean_checkpoint}")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        
        # Find checkpoints for corrupted models
        for corruption in args.train_corruptions:
            if corruption == 'none':
                continue
                
            corrupt_dir = os.path.join(checkpoints_dir, corruption)
            try:
                corrupt_checkpoint = find_best_checkpoint(corrupt_dir)
                checkpoints[corruption] = corrupt_checkpoint
                print(f"Found {corruption} model checkpoint: {corrupt_checkpoint}")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
    
    if not checkpoints:
        print("Error: No checkpoints found. Cannot proceed with prediction.")
        error_file = os.path.join(directories['predictions'], "NO_CHECKPOINTS_FOUND.txt")
        with open(error_file, 'w') as f:
            f.write(f"No checkpoints found at {datetime.datetime.now()}\n")
            f.write(f"Train corruptions: {args.train_corruptions}\n")
            f.write(f"Checkpoints directory: {directories['checkpoints']}\n")
        return False
    
    # Find prediction script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = find_file_with_fallbacks(os.path.join(base_dir, "predict_endoscapes.py"), "prediction script")
    
    if not predict_script:
        print("ERROR: Could not find predict_endoscapes.py script")
        error_file = os.path.join(directories['predictions'], "PREDICTION_SCRIPT_NOT_FOUND.txt")
        with open(error_file, 'w') as f:
            f.write(f"Prediction script not found at {datetime.datetime.now()}\n")
        return False
    
    # Find config file
    config_path = find_file_with_fallbacks(args.config, "config file")
    
    if not config_path:
        print(f"ERROR: Could not find config file: {args.config}")
        error_file = os.path.join(directories['predictions'], "CONFIG_NOT_FOUND.txt")
        with open(error_file, 'w') as f:
            f.write(f"Config file not found at {datetime.datetime.now()}\n")
            f.write(f"Searched for: {args.config}\n")
        return False
    
    print(f"Using prediction script: {predict_script}")
    print(f"Using config file: {config_path}")
    
    # Generate predictions for each model x corruption combination
    prediction_results = []
    for train_condition, checkpoint in checkpoints.items():
        for eval_corruption in args.predict_corruptions:
            # Verify the checkpoint exists before attempting prediction
            if not os.path.exists(checkpoint):
                print(f"ERROR: Checkpoint does not exist: {checkpoint}")
                continue
                
            # Construct full python path to ensure we're using the right interpreter
            python_path = "python" if not is_docker() else "/opt/conda/envs/latentgraph/bin/python"
            
            # Create a unique output directory for this prediction
            output_subdir = os.path.join(directories['predictions'], f"{train_condition}_{eval_corruption}")
            os.makedirs(output_subdir, exist_ok=True)
            
            predict_cmd = f"{python_path} {predict_script} {config_path} " \
                         f"{checkpoint} " \
                         f"--out-dir {output_subdir} " \
                         f"--corruption {eval_corruption} " \
                         f"--train-condition {train_condition} " \
                         f"--device {args.device}"
            
            print(f"Predicting {train_condition} model on {eval_corruption}: {predict_cmd}")
            result = subprocess.run(predict_cmd, shell=True, capture_output=True, text=True)
            
            # Store prediction results
            success = result.returncode == 0
            prediction_results.append({
                'train_condition': train_condition,
                'eval_corruption': eval_corruption,
                'success': success,
                'checkpoint': checkpoint,
                'output_dir': output_subdir
            })
            
            # Print command output for debugging
            print(f"\nPrediction command output for {train_condition} on {eval_corruption}:")
            print(f"Return code: {result.returncode}")
            print("STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            # Create a complete output directory
            detail_dir = os.path.join(output_subdir, "details")
            os.makedirs(detail_dir, exist_ok=True)
            
            # Save command output to file for debugging
            with open(os.path.join(detail_dir, "predict_command_output.txt"), 'w') as f:
                f.write(f"Command: {predict_cmd}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
                
                # Add checkpoint and config information
                f.write(f"\nCheckpoint: {checkpoint}\n")
                f.write(f"Config: {config_path}\n")
                f.write(f"Train condition: {train_condition}\n")
                f.write(f"Eval corruption: {eval_corruption}\n")
                
                # Check if any predictions were generated
                prediction_files = glob.glob(os.path.join(output_subdir, "*.pkl")) + \
                                  glob.glob(os.path.join(output_subdir, "*.json"))
                
                f.write(f"\nPrediction files generated: {len(prediction_files)}\n")
                for pred_file in prediction_files:
                    f.write(f"  - {os.path.basename(pred_file)}\n")
    
    # Write summary of prediction attempts
    summary_file = os.path.join(directories['predictions'], "prediction_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Prediction summary at {datetime.datetime.now()}\n")
        f.write(f"Total prediction attempts: {len(prediction_results)}\n")
        
        successful = [r for r in prediction_results if r['success']]
        f.write(f"Successful predictions: {len(successful)}/{len(prediction_results)}\n\n")
        
        f.write("Prediction details:\n")
        for idx, result in enumerate(prediction_results):
            f.write(f"{idx+1}. {result['train_condition']} model on {result['eval_corruption']} corruption: ")
            f.write("SUCCESS" if result['success'] else "FAILED")
            f.write(f" (Checkpoint: {result['checkpoint']})\n")
            
            # Check for prediction outputs
            output_dir = result['output_dir']
            prediction_files = glob.glob(os.path.join(output_dir, "*.pkl")) + \
                               glob.glob(os.path.join(output_dir, "*.json"))
            
            f.write(f"   Output directory: {output_dir}\n")
            f.write(f"   Prediction files: {len(prediction_files)}\n")
            for pred_file in prediction_files[:5]:  # Show first 5 files only
                f.write(f"   - {os.path.basename(pred_file)}\n")
            
            if len(prediction_files) > 5:
                f.write(f"   ... and {len(prediction_files) - 5} more files\n")
    
    # Return True if any predictions were successful
    return any(r['success'] for r in prediction_results)

def find_best_checkpoint(checkpoint_dir):
    """
    Find the best checkpoint in the given directory.
    - Handles different corruption subfolder structure
    - Searches for best, latest, or highest epoch checkpoint
    - Returns clear error messages if checkpoints are not found
    """
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    
    # First check if there's a corruption subdirectory (corruption_none, corruption_gaussian_noise, etc.)
    corruption_subdirs = list(Path(checkpoint_dir).glob("corruption_*"))
    if corruption_subdirs:
        # For clean models, look for corruption_none subdirectory
        if os.path.basename(checkpoint_dir) == "clean":
            corruption_none = os.path.join(checkpoint_dir, "corruption_none")
            if os.path.exists(corruption_none) and os.path.isdir(corruption_none):
                checkpoint_dir = corruption_none
                print(f"Found clean model corruption directory: {checkpoint_dir}")
            else:
                # If there's no corruption_none subdirectory, use the first corruption subdirectory
                checkpoint_dir = str(corruption_subdirs[0])
                print(f"No corruption_none directory found, using: {checkpoint_dir}")
        else:
            # For specific corruption models, look for matching corruption subdirectory
            corruption_name = os.path.basename(checkpoint_dir)
            corruption_subdir = os.path.join(checkpoint_dir, f"corruption_{corruption_name}")
            if os.path.exists(corruption_subdir) and os.path.isdir(corruption_subdir):
                checkpoint_dir = corruption_subdir
                print(f"Found corruption directory for {corruption_name}: {checkpoint_dir}")
            else:
                # If there's no matching corruption subdirectory, use the first corruption subdirectory
                if corruption_subdirs:
                    checkpoint_dir = str(corruption_subdirs[0])
                    print(f"No corruption_{corruption_name} directory found, using: {checkpoint_dir}")
                else:
                    print(f"Warning: No corruption subdirectories found in {checkpoint_dir}")
    else:
        print(f"No corruption subdirectories found, looking for checkpoints directly in: {checkpoint_dir}")
    
    # Look for best_*.pth files first (preferred)
    best_checkpoints = list(Path(checkpoint_dir).glob("best_*.pth"))
    if best_checkpoints:
        best_checkpoint = str(best_checkpoints[0])
        print(f"Found best checkpoint: {best_checkpoint}")
        return best_checkpoint
    
    # If no best checkpoint, look for latest checkpoint
    latest_file = os.path.join(checkpoint_dir, "latest.pth")
    if os.path.exists(latest_file):
        print(f"Found latest checkpoint: {latest_file}")
        return latest_file
    
    # If no latest, use the highest epoch
    epoch_files = list(Path(checkpoint_dir).glob("epoch_*.pth"))
    if epoch_files:
        highest_epoch = str(max(epoch_files, key=lambda x: int(x.stem.split('_')[-1])))
        print(f"Found highest epoch checkpoint: {highest_epoch}")
        return highest_epoch
    
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}. Make sure training has completed successfully.")

def run_evaluation(args, directories):
    """Run evaluation on all predictions."""
    print("\n=== EVALUATING PREDICTIONS ===")
    
    # Find evaluation script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = find_file_with_fallbacks(os.path.join(base_dir, "evaluate_endoscapes.py"), "evaluation script")
    
    if not eval_script:
        print("ERROR: Could not find evaluate_endoscapes.py script")
        error_file = os.path.join(directories['evaluation'], "EVALUATION_SCRIPT_NOT_FOUND.txt")
        with open(error_file, 'w') as f:
            f.write(f"Evaluation script not found at {datetime.datetime.now()}\n")
        return False
    
    # Determine ground truth file based on dataset
    if args.dataset == 'endoscapes':
        gt_file_relative = "data/mmdet_datasets/endoscapes/annotations/test.json"
    elif args.dataset == 'cholecT50':
        gt_file_relative = "data/mmdet_datasets/cholecT50/annotations/test.json"
    elif args.dataset == 'c80_phase':
        gt_file_relative = "data/mmdet_datasets/c80_phase/annotations/test.json"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Find ground truth file with robust path handling
    gt_file = find_file_with_fallbacks(gt_file_relative, "ground truth file")
    
    if not gt_file:
        print(f"ERROR: Could not find ground truth file: {gt_file_relative}")
        error_file = os.path.join(directories['evaluation'], "GT_FILE_NOT_FOUND.txt")
        with open(error_file, 'w') as f:
            f.write(f"Ground truth file not found at {datetime.datetime.now()}\n")
            f.write(f"Searched for: {gt_file_relative}\n")
        return False
    
    # Construct full python path to ensure we're using the right interpreter
    python_path = "python" if not is_docker() else "/opt/conda/envs/latentgraph/bin/python"
    
    eval_cmd = f"{python_path} {eval_script} " \
              f"--pred-dir {directories['predictions']} " \
              f"--out-dir {directories['evaluation']} " \
              f"--gt-file {gt_file} " \
              f"--metrics {' '.join(args.metrics)}"
    
    print(f"Evaluating predictions: {eval_cmd}")
    result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
    
    # Print output for debugging
    print(f"\nEvaluation command output:")
    print(f"Return code: {result.returncode}")
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    # Save command output to file for debugging
    output_file = os.path.join(directories['evaluation'], "evaluation_output.txt")
    with open(output_file, 'w') as f:
        f.write(f"Command: {eval_cmd}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"STDOUT:\n{result.stdout}\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    return result.returncode == 0

def run_analysis(args, directories):
    """Run statistical analysis on evaluation results."""
    print("\n=== RUNNING STATISTICAL ANALYSIS ===")
    
    # Check if CSV file exists
    csv_file = os.path.join(directories['evaluation'], 'all_evaluations.csv')
    if not os.path.exists(csv_file):
        print(f"ERROR: Evaluation CSV file not found: {csv_file}")
        error_file = os.path.join(directories['statistics'], "CSV_FILE_NOT_FOUND.txt")
        with open(error_file, 'w') as f:
            f.write(f"CSV file not found at {datetime.datetime.now()}\n")
            f.write(f"Looked for: {csv_file}\n")
        return False
    
    # Find analysis script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_script = find_file_with_fallbacks(os.path.join(base_dir, "statistical_analysis.py"), "analysis script")
    
    if not analysis_script:
        print("ERROR: Could not find statistical_analysis.py script")
        error_file = os.path.join(directories['statistics'], "ANALYSIS_SCRIPT_NOT_FOUND.txt")
        with open(error_file, 'w') as f:
            f.write(f"Analysis script not found at {datetime.datetime.now()}\n")
        return False
    
    # Construct full python path to ensure we're using the right interpreter
    python_path = "python" if not is_docker() else "/opt/conda/envs/latentgraph/bin/python"
    
    analysis_cmd = f"{python_path} {analysis_script} " \
                  f"--csv-file {csv_file} " \
                  f"--out-dir {directories['statistics']} " \
                  f"--baseline clean " \
                  f"--test wilcoxon " \
                  f"--metrics {' '.join([m + '_mAP' for m in args.metrics])}"
    
    print(f"Analyzing results: {analysis_cmd}")
    result = subprocess.run(analysis_cmd, shell=True, capture_output=True, text=True)
    
    # Print output for debugging
    print(f"\nAnalysis command output:")
    print(f"Return code: {result.returncode}")
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    # Save command output to file for debugging
    output_file = os.path.join(directories['statistics'], "analysis_output.txt")
    with open(output_file, 'w') as f:
        f.write(f"Command: {analysis_cmd}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"STDOUT:\n{result.stdout}\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    return result.returncode == 0

def main():
    args = parse_args()
    
    print(f"Starting corruption robustness evaluation pipeline")
    print(f"Running in Docker container: {is_docker()}")
    print(f"Dataset: {args.dataset}")
    print(f"Model config: {args.config}")
    print(f"Training corruptions: {args.train_corruptions}")
    print(f"Prediction corruptions: {args.predict_corruptions}")
    
    # Setup experiment directory
    directories = setup_experiment_directory(args)
    print(f"Experiment directory: {directories['root']}")
    
    # Save command line information
    script_path = os.path.abspath(__file__)
    command = f"python {script_path} " + " ".join([f"--{k}={v}" if not isinstance(v, list) else f"--{k} {' '.join(v)}" for k, v in vars(args).items() if v is not None])
    
    with open(os.path.join(directories['root'], "command.txt"), 'w') as f:
        f.write(f"Command: {command}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Docker container: {is_docker()}\n")
    
    # Select dataset - continue even if this fails since it might not be critical
    dataset_selected = run_dataset_selection(args)
    if not dataset_selected:
        print("WARNING: Dataset selection failed, but continuing with the pipeline...")
    
    # Track successful stages
    results = {}
    
    # Run selected tasks
    if args.task in ['train', 'all']:
        results['train'] = run_training(args, directories)
    
    if args.task in ['predict', 'all']:
        results['predict'] = run_prediction(args, directories)
    
    if args.task in ['evaluate', 'all'] and results.get('predict', True):  # Only evaluate if predictions exist
        results['evaluate'] = run_evaluation(args, directories)
    
    if args.task in ['analyze', 'all'] and results.get('evaluate', True):  # Only analyze if evaluation exists
        results['analyze'] = run_analysis(args, directories)
    
    # Create a symlink to the latest experiment
    latest_link = os.path.join(args.results_dir, 'latest')
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.remove(latest_link)
        else:
            shutil.rmtree(latest_link)
    
    os.symlink(directories['root'], latest_link, target_is_directory=True)
    
    # Write status summary
    status_file = os.path.join(directories['root'], "status.txt")
    with open(status_file, 'w') as f:
        f.write(f"Pipeline completed at: {datetime.datetime.now()}\n")
        f.write(f"Results directory: {directories['root']}\n\n")
        
        f.write("Task Status:\n")
        if args.task in ['train', 'all']:
            f.write(f"Training: {'SUCCESS' if results.get('train', False) else 'FAILED'}\n")
        
        if args.task in ['predict', 'all']:
            f.write(f"Prediction: {'SUCCESS' if results.get('predict', False) else 'FAILED'}\n")
        
        if args.task in ['evaluate', 'all']:
            f.write(f"Evaluation: {'SUCCESS' if results.get('evaluate', False) else 'FAILED'}\n")
        
        if args.task in ['analyze', 'all']:
            f.write(f"Analysis: {'SUCCESS' if results.get('analyze', False) else 'FAILED'}\n")
    
    print(f"\nExperiment completed! Results stored in: {directories['root']}")
    print(f"Quick access link: {latest_link}")
    
    # Print task status
    print("\nTask Status:")
    if args.task in ['train', 'all']:
        print(f"Training: {'SUCCESS' if results.get('train', False) else 'FAILED'}")
    
    if args.task in ['predict', 'all']:
        print(f"Prediction: {'SUCCESS' if results.get('predict', False) else 'FAILED'}")
    
    if args.task in ['evaluate', 'all']:
        print(f"Evaluation: {'SUCCESS' if results.get('evaluate', False) else 'FAILED'}")
    
    if args.task in ['analyze', 'all']:
        print(f"Analysis: {'SUCCESS' if results.get('analyze', False) else 'FAILED'}")

if __name__ == "__main__":
    main()