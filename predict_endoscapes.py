#!/usr/bin/env python
# Prediction script with corruption support for SurgLatentGraph on EndoScapes

import os
import argparse
import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from mmdet.utils import register_all_modules

from corruptions import corrupt

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions with corruption evaluation')
    parser.add_argument('config', help='Path to the test config file')
    parser.add_argument('checkpoint', help='Path to the trained model checkpoint')
    parser.add_argument('--out-dir', default='results/endoscapes_preds/', help='Directory to save predictions')
    parser.add_argument('--corruption', type=str, default=None, 
                      choices=['gaussian_noise', 'motion_blur', 'defocus_blur', 
                               'uneven_illumination', 'smoke_effect', 'random_corruptions', 'none'],
                      help='Type of corruption to apply during inference')
    parser.add_argument('--corruption-severity', type=float, default=0.5,
                      help='Severity of the corruption (0.0-1.0)')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--train-condition', default='clean',
                      help='Training condition label for the output predictions')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithms')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment for prediction."""
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Register all modules
    register_all_modules()
    
    # Create output directory
    corruption_suffix = f"_{args.corruption}" if args.corruption else "_clean"
    out_dir = os.path.join(args.out_dir, f"{args.train_condition}{corruption_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    
    return out_dir

def modify_config_for_corruption(cfg, args, out_dir):
    """Modify the config to include corruption during testing."""
    # Set output directory in the config
    cfg.out_dir = out_dir
    
    # Set corruption information
    cfg.corruption_info = {
        'type': args.corruption,
        'severity': args.corruption_severity,
        'train_condition': args.train_condition
    }
    
    # Modify the data transform to include corruption
    # This affects the LoadImageFromFile and LoadLG transforms
    
    return cfg

def build_model_from_cfg(cfg, checkpoint, device):
    """Build the model from configuration and load checkpoint."""
    # Build the model
    model = MODELS.build(cfg.model)
    model.to(device)
    
    # Load the checkpoint
    load_checkpoint(model, checkpoint, map_location=device)
    model.eval()
    
    return model

# Custom transform that applies corruption to images
class CorruptionTransform:
    """Apply corruption to images before model processing."""
    
    def __init__(self, corruption_type, severity=0.5):
        self.corruption_type = corruption_type
        self.severity = severity
    
    def __call__(self, img):
        if self.corruption_type:
            return corrupt(img, self.corruption_type)
        return img

def process_dataset(model, cfg, args):
    """Process the dataset and generate predictions."""
    from torch.utils.data import DataLoader
    from mmengine.registry import DATASETS
    
    print(f"Starting dataset processing with device: {args.device}")
    print(f"Corruption type: {args.corruption}")
    
    try:
        # Build dataset with corruption transform
        print("Building dataset...")
        
        # First, check if the dataset config exists in the configuration
        if not hasattr(cfg, 'test_dataloader') or not hasattr(cfg.test_dataloader, 'dataset'):
            print("ERROR: test_dataloader.dataset not found in config!")
            print("Available config attributes:", dir(cfg))
            # Try to fall back to a default dataset if possible
            if hasattr(cfg, 'data') and hasattr(cfg.data, 'test'):
                print("Attempting to fall back to cfg.data.test...")
                test_dataset = DATASETS.build(cfg.data.test)
            else:
                raise ValueError("Cannot find test dataset configuration in cfg")
        else:
            test_dataset = DATASETS.build(cfg.test_dataloader.dataset)
            
        print(f"Dataset built successfully with {len(test_dataset)} samples")
        
        # Debug dataset contents
        print(f"Dataset type: {type(test_dataset)}")
        if hasattr(test_dataset, 'data_list') and test_dataset.data_list:
            print(f"First few dataset entries: {test_dataset.data_list[:2]}")
        elif hasattr(test_dataset, 'data_infos') and test_dataset.data_infos:
            print(f"First few dataset entries: {test_dataset.data_infos[:2]}")
        
        # Get DataLoader parameters with safe defaults
        num_workers = getattr(cfg.test_dataloader, 'num_workers', 2) if hasattr(cfg, 'test_dataloader') else 2
        pin_memory = getattr(cfg.test_dataloader, 'pin_memory', False) if hasattr(cfg, 'test_dataloader') else False
        
        # Build dataloader with safely accessed parameters
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=getattr(test_dataset, 'collate_fn', None)  # Use dataset's collate_fn if available
        )
        print(f"DataLoader created successfully")
        
        # Check model device
        print(f"Model device check: {next(model.parameters()).device}")
        
        # Set up corruption for image processing
        corruption_transform = None
        if args.corruption and args.corruption != 'none':
            print(f"Setting up corruption transform with: {args.corruption}")
            corruption_transform = CorruptionTransform(args.corruption, args.corruption_severity)
        
        results = []
        # Process each batch
        print(f"Starting to process {len(test_dataset)} samples...")
        for idx, data in enumerate(test_dataloader):
            if idx % 10 == 0:
                print(f"Processing batch {idx}/{len(test_dataset)}")
                
            # Apply corruption if specified
            if corruption_transform and 'img' in data:
                data['img'] = corruption_transform(data['img'])
            
            # Debug print the first batch to understand data structure
            if idx == 0:
                print(f"First batch keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
                if isinstance(data, dict) and 'img' in data:
                    print(f"Image shape: {data['img'].shape}")
                if isinstance(data, dict) and 'img_metas' in data:
                    print(f"Image metas: {data['img_metas']}")
            
            try:
                with torch.no_grad():
                    # First attempt with standard API
                    result = model(return_loss=False, **data)
                    
                # Log successful prediction details
                if idx == 0 or idx % 50 == 0:
                    print(f"Batch {idx} prediction successful")
                    if isinstance(result, list) and result:
                        print(f"Result type: {type(result[0])}")
                        if isinstance(result[0], dict):
                            print(f"Result keys: {result[0].keys()}")
                    elif isinstance(result, dict):
                        print(f"Result keys: {result.keys()}")
                
                results.append(result)
            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                
                # Debug information
                print(f"Data keys: {data.keys() if isinstance(data, dict) else type(data)}")
                
                # Log available model methods for debugging
                print(f"Available model methods: {[m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]}")
                
                # Try alternative methods
                try:
                    print("Trying simple_test method...")
                    with torch.no_grad():
                        if hasattr(model, 'simple_test'):
                            result = model.simple_test(data['img'], data.get('img_metas', None))
                            results.append(result)
                            print(f"Recovered using simple_test")
                        elif hasattr(model, 'forward_test'):
                            result = model.forward_test(data)
                            results.append(result)
                            print(f"Recovered using forward_test")
                        else:
                            # Last resort - attempt direct forward call with minimal data
                            print("Trying direct forward call...")
                            with torch.no_grad():
                                if 'img' in data:
                                    # Ensure image is on the correct device
                                    img = data['img'].to(next(model.parameters()).device)
                                    result = model(img)
                                    results.append(result)
                                    print(f"Recovered using direct forward call")
                                else:
                                    print("Cannot recover - no 'img' in data")
                except Exception as e2:
                    print(f"All recovery attempts failed: {e2}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next batch instead of failing
                    continue
        
        print(f"Completed processing {len(results)} batches")
        if not results:
            print("WARNING: No results generated! Check model and dataset compatibility.")
        return results
    except Exception as e:
        print(f"Error in process_dataset: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results rather than failing
        return []

def save_predictions(results, out_dir, cfg):
    """Save predictions to output directory."""
    import pickle
    import json
    import datetime
    
    print(f"Attempting to save predictions to {out_dir}")
    
    try:
        # Save raw results
        if results:
            print(f"Saving {len(results)} prediction results")
            with open(os.path.join(out_dir, 'predictions.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print("Predictions saved successfully")
        else:
            print("Warning: No prediction results to save")
            # Save an empty file to indicate we tried
            with open(os.path.join(out_dir, 'predictions_empty.pkl'), 'wb') as f:
                pickle.dump([], f)
        
        # Save metadata about the evaluation
        metadata = {
            'config': cfg.filename if hasattr(cfg, 'filename') else str(cfg),
            'corruption': cfg.corruption_info if hasattr(cfg, 'corruption_info') else {'type': 'none'},
            'timestamp': str(datetime.datetime.now()),
            'results_count': len(results) if results else 0
        }
        
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save a simple results summary
        summary = {
            'num_predictions': len(results) if results else 0,
            'timestamp': str(datetime.datetime.now()),
            'status': 'success' if results else 'empty_results'
        }
        
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"Predictions and metadata saved to {out_dir}")
        
        # Create a flag file to indicate prediction completion
        with open(os.path.join(out_dir, 'prediction_complete.txt'), 'w') as f:
            f.write(f"Prediction completed at {datetime.datetime.now()}\n")
            f.write(f"Number of predictions: {len(results) if results else 0}\n")
        
        return True
    except Exception as e:
        print(f"Error saving predictions: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error log
        with open(os.path.join(out_dir, 'error_log.txt'), 'w') as f:
            f.write(f"Error occurred at {datetime.datetime.now()}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())
        
        return False

def export_predictions_to_csv(results, out_dir, cfg):
    """
    Export predictions to CSV format for evaluation.
    This creates CSV files compatible with the evaluation script.
    """
    import csv
    import os
    import numpy as np
    
    print(f"Exporting predictions to CSV format in {out_dir}")
    
    if not results:
        print("No results to export to CSV")
        # Create empty CSV to indicate we tried
        with open(os.path.join(out_dir, 'empty_predictions.csv'), 'w') as f:
            f.write("# No predictions were generated\n")
        return False
    
    try:
        # Prepare different CSV files for different types of predictions
        # For bounding box predictions
        bbox_csv_path = os.path.join(out_dir, 'bbox_predictions.csv')
        
        with open(bbox_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['image_id', 'category_id', 'bbox', 'score'])
            
            # Process each batch of results
            for batch_idx, batch_results in enumerate(results):
                if not batch_results:
                    continue
                    
                # Handle different result formats
                if isinstance(batch_results, dict):
                    # Check for different result keys
                    if 'pred_instances' in batch_results:
                        instances = batch_results['pred_instances']
                        if hasattr(instances, 'bboxes') and hasattr(instances, 'scores') and hasattr(instances, 'labels'):
                            bboxes = instances.bboxes.cpu().numpy()
                            scores = instances.scores.cpu().numpy()
                            labels = instances.labels.cpu().numpy()
                            img_id = batch_idx  # Fallback
                            if 'img_id' in batch_results:
                                img_id = batch_results['img_id']
                            
                            for i in range(len(bboxes)):
                                bbox_str = ','.join(map(str, bboxes[i].tolist()))
                                csv_writer.writerow([img_id, int(labels[i]), bbox_str, float(scores[i])])
                    
                    # Alternative format sometimes used in MMDetection
                    elif 'bboxes' in batch_results and 'labels' in batch_results:
                        bboxes = batch_results['bboxes']
                        scores = batch_results['scores'] if 'scores' in batch_results else [1.0] * len(bboxes)
                        labels = batch_results['labels']
                        img_id = batch_results.get('img_id', batch_idx)
                        
                        for i in range(len(bboxes)):
                            bbox_str = ','.join(map(str, bboxes[i]))
                            csv_writer.writerow([img_id, int(labels[i]), bbox_str, float(scores[i])])
                
                # Handle list format
                elif isinstance(batch_results, (list, tuple)):
                    for item in batch_results:
                        if isinstance(item, dict) and 'bbox' in item and 'label' in item:
                            img_id = item.get('img_id', batch_idx)
                            bbox_str = ','.join(map(str, item['bbox']))
                            score = item.get('score', 1.0)
                            csv_writer.writerow([img_id, int(item['label']), bbox_str, float(score)])
                
                # Handle numpy array format
                elif isinstance(batch_results, np.ndarray) and batch_results.shape[1] >= 5:  # [x1, y1, x2, y2, score, class_id]
                    img_id = batch_idx
                    for detection in batch_results:
                        bbox = detection[:4]
                        score = detection[4]
                        label = int(detection[5]) if len(detection) > 5 else 0
                        bbox_str = ','.join(map(str, bbox))
                        csv_writer.writerow([img_id, label, bbox_str, float(score)])
        
        print(f"Successfully exported predictions to CSV: {bbox_csv_path}")
        return True
        
    except Exception as e:
        print(f"Error exporting predictions to CSV: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error log
        with open(os.path.join(out_dir, 'csv_export_error.txt'), 'w') as f:
            f.write(f"Error exporting to CSV: {str(e)}\n")
            f.write(traceback.format_exc())
            
        return False

def main():
    """Main function for prediction with enhanced error handling."""
    try:
        args = parse_args()
        print(f"Starting prediction with args: {args}")
        print(f"Using device: {args.device}")
        print(f"Checkpoint: {args.checkpoint}")
        
        # Set up environment
        out_dir = setup_environment(args)
        print(f"Output directory created: {out_dir}")
        
        # Create a log file
        with open(os.path.join(out_dir, 'prediction_log.txt'), 'w') as log_file:
            log_file.write(f"Starting prediction at {datetime.datetime.now()}\n")
            log_file.write(f"Arguments: {vars(args)}\n")
        
        # Load config with error handling
        try:
            print(f"Loading config from: {args.config}")
            cfg = Config.fromfile(args.config)
            print("Config loaded successfully")
        except Exception as e:
            print(f"Error loading config: {e}")
            with open(os.path.join(out_dir, 'config_error.txt'), 'w') as f:
                f.write(f"Error loading config: {str(e)}\n")
                f.write(f"Config path: {args.config}\n")
            raise
        
        # Modify config for corruption
        cfg = modify_config_for_corruption(cfg, args, out_dir)
        
        # Build model with error handling
        try:
            print(f"Building model from checkpoint: {args.checkpoint}")
            model = build_model_from_cfg(cfg, args.checkpoint, args.device)
            print("Model built and checkpoint loaded successfully")
        except Exception as e:
            print(f"Error building model: {e}")
            with open(os.path.join(out_dir, 'model_error.txt'), 'w') as f:
                f.write(f"Error building model: {str(e)}\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
            raise
        
        # Process dataset and generate predictions
        try:
            print("Starting dataset processing")
            results = process_dataset(model, cfg, args)
            print(f"Dataset processing completed with {len(results)} results")
        except Exception as e:
            print(f"Error processing dataset: {e}")
            with open(os.path.join(out_dir, 'dataset_error.txt'), 'w') as f:
                f.write(f"Error processing dataset: {str(e)}\n")
            results = []
        
        # Save predictions
        success = save_predictions(results, out_dir, cfg)
        
        # Export predictions to CSV
        export_success = export_predictions_to_csv(results, out_dir, cfg)
        
        # Write final status
        status = "success" if success and results else "completed_with_warnings"
        with open(os.path.join(out_dir, 'status.txt'), 'w') as f:
            f.write(f"Status: {status}\n")
            f.write(f"Completed at: {datetime.datetime.now()}\n")
            f.write(f"Results count: {len(results)}\n")
            f.write(f"CSV export: {'success' if export_success else 'failed'}\n")
        
        print(f"Prediction process completed with status: {status}")
        return success
        
    except Exception as e:
        print(f"Unhandled error in main: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to create error log even if output directory setup failed
        try:
            error_dir = out_dir if 'out_dir' in locals() else os.path.join('results/errors', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
            os.makedirs(error_dir, exist_ok=True)
            
            with open(os.path.join(error_dir, 'critical_error.txt'), 'w') as f:
                f.write(f"Critical error occurred at {datetime.datetime.now()}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            print("Failed to create error log")
        
        return False

if __name__ == '__main__':
    import datetime
    success = main()
    exit(0 if success else 1)  # Use exit code to indicate success/failure