import torch
import numpy as np
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import MODELS
from mmengine.structures import PixelData

# Import our corruption functions
from corruptions import corrupt

@MODELS.register_module()
class CorruptionDataPreprocessor(BaseDataPreprocessor):
    """
    A data preprocessor that applies corruptions to images.
    
    This preprocessor wraps an existing data preprocessor and applies
    corruption transforms to the input images.
    """
    
    def __init__(self, 
                 base_preprocessor,
                 corruption_type='none',
                 apply_during_training=True,
                 apply_during_testing=True,
                 **kwargs):
        super().__init__()
        # Convert the configuration dict to a preprocessor instance if needed
        if isinstance(base_preprocessor, dict):
            self.base_preprocessor = MODELS.build(base_preprocessor)
        else:
            self.base_preprocessor = base_preprocessor
        self.corruption_type = corruption_type
        self.apply_during_training = apply_during_training
        self.apply_during_testing = apply_during_testing
        print(f"Initialized CorruptionDataPreprocessor with corruption: {corruption_type}")
        print(f"  - Apply during training: {apply_during_training}")
        print(f"  - Apply during testing: {apply_during_testing}")
    
    def forward(self, data, training=False):
        """Process the data with corruptions."""
        # First, let the base preprocessor do its work
        data = self.base_preprocessor(data, training)
        
        # Skip if no corruption is specified
        if self.corruption_type is None or self.corruption_type == 'none':
            return data
        
        # Check if we should apply corruption based on training/testing phase
        should_apply = False
        if training and self.apply_during_training:
            should_apply = True
            print(f"🔧 Applying {self.corruption_type} corruption during TRAINING")
        elif not training and self.apply_during_testing:
            should_apply = True
            print(f"🔧 Applying {self.corruption_type} corruption during TESTING")
        
        if not should_apply:
            return data
        
        # Apply corruption to inputs
        if isinstance(data, dict):
            # Extract the batch inputs tensor
            if 'inputs' in data and isinstance(data['inputs'], torch.Tensor):
                batch_size = data['inputs'].shape[0]
                
                # Apply corruption to each image individually
                corrupted_images = []
                for i in range(batch_size):
                    # Extract single image from batch
                    single_image = data['inputs'][i:i+1]  # Keep batch dimension
                    
                    # Apply corruption to individual image
                    if self.corruption_type == 'random_corruptions':
                        # For random corruptions, each image gets a different corruption
                        corrupted_image = corrupt(single_image, self.corruption_type)
                    else:
                        # For specific corruptions, apply the same corruption to each image
                        corrupted_image = corrupt(single_image, self.corruption_type)
                    
                    corrupted_images.append(corrupted_image)
                
                # Concatenate all corrupted images back into a batch
                data['inputs'] = torch.cat(corrupted_images, dim=0)
                
                if batch_size > 1:
                    print(f"   📦 Applied corruption to {batch_size} individual images in batch")
        
        return data