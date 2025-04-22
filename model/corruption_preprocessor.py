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
                 **kwargs):
        super().__init__()
        # Convert the configuration dict to a preprocessor instance if needed
        if isinstance(base_preprocessor, dict):
            self.base_preprocessor = MODELS.build(base_preprocessor)
        else:
            self.base_preprocessor = base_preprocessor
        self.corruption_type = corruption_type
        print(f"Initialized CorruptionDataPreprocessor with corruption: {corruption_type}")
    
    def forward(self, data, training=False):
        """Process the data with corruptions."""
        # First, let the base preprocessor do its work
        data = self.base_preprocessor(data, training)
        
        # Skip if no corruption is specified
        if self.corruption_type is None or self.corruption_type == 'none':
            return data
        
        # Apply corruption to inputs
        if isinstance(data, dict):
            # Extract the batch inputs tensor
            if 'inputs' in data and isinstance(data['inputs'], torch.Tensor):
                # Apply corruption to the preprocessed inputs
                data['inputs'] = corrupt(data['inputs'], self.corruption_type)
        
        return data