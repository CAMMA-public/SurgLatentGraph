import numpy as np
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from corruptions import corrupt

@TRANSFORMS.register_module()
class CorruptionTransform(BaseTransform):
    """Apply corruption to images during training."""
    
    def __init__(self, corruption_type='none'):
        self.corruption_type = corruption_type
        print(f"🔧 Initialized CorruptionTransform with: {corruption_type}")
    
    def transform(self, results):
        """Apply corruption to the loaded image."""
        if self.corruption_type == 'none':
            return results
        
        # Get image from results
        img = results['img']
        
        # Apply corruption
        print(f"🔧 Applying {self.corruption_type} corruption to image shape: {img.shape}")
        corrupted_img = corrupt(img, self.corruption_type)
        
        # Update results with corrupted image
        results['img'] = corrupted_img
        
        return results