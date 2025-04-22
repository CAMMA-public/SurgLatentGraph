import torch
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import os
import random
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_gaussian_noise(image, mean=0, std=0.5):
    """
    Adds Gaussian noise to a PyTorch image tensor.
    
    Args:
        image (torch.Tensor): Input image tensor
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise
        
    Returns:
        torch.Tensor: Noisy image tensor
    """
    # Generate Gaussian noise with same shape
    noise = torch.randn_like(image, dtype=torch.float32) * std + mean
    noisy_image = image.float() + noise
    
    # Clip values if needed
    if image.dtype == torch.uint8:
        noisy_image = torch.clamp(noisy_image, 0, 255).to(torch.uint8)
    
    return noisy_image

def apply_motion_blur(image, kernel_size=15):
    """
    Applies motion blur to an image tensor.
    
    Args:
        image (torch.Tensor): Input image tensor
        kernel_size (int): Size of the motion blur kernel
        
    Returns:
        torch.Tensor: Motion-blurred image tensor
    """
    # Check if the input is batched (5D: B, T, C, H, W)
    is_batched = len(image.shape) == 5
    if is_batched:
        batch_size, timesteps, channels, height, width = image.shape
        image = image.view(-1, channels, height, width)  # Reshape to (B*T, C, H, W)

    # Convert to (H, W, C) format for OpenCV processing
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # (B*T, H, W, C)

    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size

    # Apply motion blur to each image in the batch
    blurred_np = np.array([cv2.filter2D(img, -1, kernel) for img in image_np])

    # Convert back to PyTorch tensor
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device).float()

    # Reshape back if input was 5D
    if is_batched:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)

    return blurred_tensor

def apply_defocus_blur(image, kernel_size=15):
    """
    Applies defocus blur (Gaussian blur) to an image tensor.
    
    Args:
        image (torch.Tensor): Input image tensor
        kernel_size (int): Size of the Gaussian blur kernel
        
    Returns:
        torch.Tensor: Defocus-blurred image tensor
    """
    is_batched = len(image.shape) == 5
    if is_batched:
        batch_size, timesteps, channels, height, width = image.shape
        image = image.view(-1, channels, height, width)  # Flatten batch & time

    # Convert to (H, W, C) format for OpenCV
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()

    # Apply Gaussian blur to each frame
    blurred_np = np.array([cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in image_np])

    # Convert back to PyTorch tensor
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device).float()

    # Reshape back if input was 5D
    if is_batched:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)

    return blurred_tensor

def uneven_illumination(image, strength=0.5):
    """
    Applies uneven illumination to an image tensor.
    
    Args:
        image (torch.Tensor): Input image tensor
        strength (float): Strength of the illumination effect
        
    Returns:
        torch.Tensor: Image tensor with uneven illumination
    """
    # Check if input is 5D (batch + time)
    is_batched = (image.dim() == 5)
    if is_batched:
        b, t, c, h, w = image.shape
        # Flatten (B, T) into one dimension => (B*T, C, H, W)
        image = image.view(-1, c, h, w)
    else:
        # Single image: (C, H, W)
        c, h, w = image.shape
        b, t = 1, 1  # For convenient unflattening logic at the end

    # Convert to (N, H, W, C) for OpenCV-like operations
    image_np = image.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    N = image_np.shape[0]  # number of images/frames

    # Apply uneven illumination to each frame
    result_list = []
    for i in range(N):
        # Get one frame: shape (H, W, C)
        frame = image_np[i]
        h_i, w_i, c_i = frame.shape

        # Create horizontal gradient from (1 - strength) to 1
        gradient = np.linspace(1 - strength, 1, w_i, dtype=np.float32)
        # Tile vertically to match frame height, shape => (H, W, 1)
        gradient = np.tile(gradient, (h_i, 1)).reshape(h_i, w_i, 1)

        # Multiply frame by gradient, then clip to [0, 1]
        illuminated = frame * gradient
        illuminated = np.clip(illuminated, 0, 1)
        
        result_list.append(illuminated)

    # Stack all processed frames back into shape (N, H, W, C)
    result_np = np.stack(result_list, axis=0)

    # Convert to torch => shape (N, C, H, W)
    result_tensor = torch.from_numpy(result_np).permute(0, 3, 1, 2).to(image.device).float()

    # Unflatten if originally batched
    if is_batched:
        result_tensor = result_tensor.view(b, t, c, h, w)

    return result_tensor

def generate_perlin_noise(height, width, scale=10, intensity=0.5):
    """
    Generates Perlin noise for smoke effect.
    
    Args:
        height (int): Height of the output noise
        width (int): Width of the output noise
        scale (float): Scale of the noise
        intensity (float): Intensity of the noise
        
    Returns:
        numpy.ndarray: Perlin noise array
    """
    noise_pattern = np.zeros((height, width), dtype=np.float32)
    
    # Using a simpler approach since pnoise2 requires additional dependencies
    for y in range(height):
        for x in range(width):
            noise_pattern[y, x] = np.sin(x/scale) * np.sin(y/scale)
            noise_pattern[y, x] += 0.5 * np.sin(x/(scale*0.5)) * np.sin(y/(scale*0.5))
            noise_pattern[y, x] += 0.25 * np.sin(x/(scale*0.25)) * np.sin(y/(scale*0.25))

    # Normalize to [0,1] and apply intensity
    noise_pattern = (noise_pattern - noise_pattern.min()) / (noise_pattern.max() - noise_pattern.min())
    noise_pattern = noise_pattern * intensity
    
    return noise_pattern

def add_smoke_effect(image, intensity=0.7):
    """
    Apply a realistic smoke effect to an image tensor.
    
    Args:
        image (torch.Tensor): Input image tensor
        intensity (float): Intensity of the smoke effect
        
    Returns:
        torch.Tensor: Image tensor with smoke effect
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor")

    image = image.to(device).clone()
    is_sequence = False
    if image.dim() == 5:  
        batch_size, seq_len, channels, height, width = image.shape
        image = image.view(batch_size * seq_len, channels, height, width)
        is_sequence = True

    if image.dim() == 4:
        batch_size, channels, height, width = image.shape
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()
    else:
        image_np = image.permute(1, 2, 0).cpu().numpy()

    h, w = image_np.shape[-3:-1]
    noise_pattern = generate_perlin_noise(h, w, scale=50, intensity=intensity)
    noise_3ch = np.stack([noise_pattern] * 3, axis=-1)
    noise_3ch = gaussian_filter(noise_3ch, sigma=5)

    if image_np.ndim == 4:
        noise_3ch = np.expand_dims(noise_3ch, axis=0)
        noise_3ch = np.repeat(noise_3ch, batch_size, axis=0)

    # Add weighted smoke effect
    corrupted = cv2.addWeighted(image_np, 1.0 - intensity, noise_3ch, intensity, 0)
    corrupted = np.clip(corrupted, 0, 1)

    corrupted_tensor = torch.from_numpy(corrupted).permute(0, 3, 1, 2).to(device).float()

    if corrupted_tensor.shape[1] != 3:
        corrupted_tensor = corrupted_tensor[:, :3, :, :]

    if is_sequence:
        corrupted_tensor = corrupted_tensor.view(batch_size // seq_len, seq_len, 3, height, width)

    return corrupted_tensor

def random_corrupt(image):
    """
    Applies a single random corruption to an image with a 50% probability.
    
    Args:
        image (torch.Tensor): Input image tensor
        
    Returns:
        torch.Tensor: Possibly corrupted image tensor
    """
    corruption_methods = [
        add_gaussian_noise,
        apply_motion_blur,
        apply_defocus_blur,
        uneven_illumination,
        add_smoke_effect
    ]

    if random.random() < 0.5:  # 50% chance to apply corruption
        corruption_method = random.choice(corruption_methods)
        return corruption_method(image)
    else:
        return image

def corrupt(image, corruption_type):
    """
    Main function to apply a specific corruption to an image.
    
    Args:
        image (torch.Tensor): Input image tensor
        corruption_type (str): Type of corruption to apply
        
    Returns:
        torch.Tensor: Corrupted image tensor
    """
    if corruption_type == 'gaussian_noise':
        return add_gaussian_noise(image)
    elif corruption_type == 'motion_blur':
        return apply_motion_blur(image)
    elif corruption_type == 'defocus_blur':
        return apply_defocus_blur(image)
    elif corruption_type == 'uneven_illumination':
        return uneven_illumination(image)
    elif corruption_type == 'smoke_effect':
        return add_smoke_effect(image)
    elif corruption_type == 'random_corruptions':
        return random_corrupt(image)
    elif corruption_type == 'none' or corruption_type is None:
        return image
    else:
        raise ValueError(f"Invalid corruption type '{corruption_type}'. Choose from: 'gaussian_noise', 'motion_blur', 'defocus_blur', 'uneven_illumination', 'smoke_effect', 'random_corruptions', 'none'.")