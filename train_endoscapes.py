#!/usr/bin/env python
# Training script with corruption support for SurgLatentGraph on EndoScapes

import os
import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import RUNNERS

from corruptions import corrupt
# Import custom hooks to register them with mmengine
from hooks.custom_hooks import CorruptionLoggerHook

def parse_args():
    parser = argparse.ArgumentParser(description='Train SurgLatentGraph with corruption robustness')
    parser.add_argument('config', help='Path to the training config file')
    parser.add_argument('--work-dir', help='Directory to save logs and models')
    parser.add_argument('--train_corruption', type=str, default=None, 
                      choices=['gaussian_noise', 'motion_blur', 'defocus_blur', 
                               'uneven_illumination', 'smoke_effect', 'random_corruptions', 'none'],
                      help='Type of corruption to apply during training')
    parser.add_argument('--corruption-severity', type=float, default=0.5,
                      help='Severity of the corruption (0.0-1.0)')
    parser.add_argument('--device', default='cuda:0', help='Device to use for training')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithms')
    parser.add_argument('--max-epochs', type=int, default=None, 
                      help='Maximum number of training epochs (overrides config)')
    
    return parser.parse_args()

def apply_corruption_to_pipeline(cfg, corruption_type, severity=0.5):
    """
    Modifies the data pipeline in the config to include corruption.
    """
    if corruption_type is None or corruption_type == 'none':
        return cfg
    
    # Store corruption info in the config for later reference
    if not hasattr(cfg, 'corruption_info'):
        cfg.corruption_info = {}
    cfg.corruption_info['type'] = corruption_type
    cfg.corruption_info['severity'] = severity
    
    # Add a custom transform to the pipeline
    corruption_transform = dict(
        type='LoadLG',  # We'll modify the LoadLG class to include corruption
        corruption_type=corruption_type,
        corruption_severity=severity
    )
    
    # Inject our corruption transform into all relevant pipelines
    for dataloader_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        if hasattr(cfg, dataloader_key):
            dataloader_cfg = getattr(cfg, dataloader_key)
            if 'dataset' in dataloader_cfg and 'pipeline' in dataloader_cfg['dataset']:
                # Keep pipeline as is - we'll handle corruption in LoadLG
                pass
                
    return cfg

def main():
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load the configuration
    cfg = Config.fromfile(args.config)
    cfg.train_corruption = args.train_corruption

    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Use config filename as default work directory
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
        
    if args.corruption:
        # If corruption specified, update the work directory to include corruption type
        cfg.work_dir = os.path.join(cfg.work_dir, f'corruption_{args.corruption}')
        
    # Apply corruption to the data pipeline
    cfg = apply_corruption_to_pipeline(cfg, args.corruption, args.corruption_severity)
    
    # Configure the runner
    if args.resume:
        cfg.resume = True
        
    # Override max_epochs if specified
    if args.max_epochs is not None:
        if hasattr(cfg, 'train_cfg') and hasattr(cfg.train_cfg, 'max_epochs'):
            cfg.train_cfg.max_epochs = args.max_epochs
            print(f"Setting max epochs to {args.max_epochs}")
        else:
            print(f"Warning: Could not set max_epochs={args.max_epochs}, train_cfg not found in config")
        
    # Add a custom hook to log corruption metrics
    if args.corruption:
        if 'custom_hooks' not in cfg:
            cfg.custom_hooks = []
        cfg.custom_hooks.append(
            dict(type='CorruptionLoggerHook', corruption_type=args.corruption)
        )
    
    # Create and run the training runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    
    runner.train()

if __name__ == '__main__':
    main()