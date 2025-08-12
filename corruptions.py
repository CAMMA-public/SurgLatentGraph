_gaussian_noise_save_counter = 0
import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
import subprocess
import sys
from ipdb import set_trace
import torch
import matplotlib
import uuid



# Try to import noise module, but provide a fallback if it's not available
try:
    import noise
    NOISE_MODULE_AVAILABLE = True
    print("Noise module available for Perlin noise corruptions.")
except ImportError:
    NOISE_MODULE_AVAILABLE = False
    print("Warning: 'noise' module not found, Perlin noise corruptions won't be available.")
    print("To enable full functionality, install with: pip install noise")
    
    # Auto-install noise module if it's not available
    try:
        print("Attempting to install noise module...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "noise"])
        import noise
        NOISE_MODULE_AVAILABLE = True
        print("Successfully installed noise module.")
    except Exception as e:
        print(f"Failed to install noise module: {e}")
        print("Continuing with limited functionality.")

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_gaussian_noise(image, mean=5, std=0.5):
    """
    Adds Gaussian noise to a PyTorch image tensor without changing its shape or type.
    """
    print(f"ADDING GAUSSIAN NOISE WITH MEAN={mean}, STD={std} TO IMAGE OF SHAPE {image.shape}")
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    import uuid
    global _gaussian_noise_save_counter
    if _gaussian_noise_save_counter < 1:
        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in [1,3]:
            img_np = img_np.transpose(1,2,0)

    # Increase std to make noise visible
    visible_std = 25.0 if image.dtype == torch.uint8 else 0.2
    # to change the intensity of noise change the std
    noise = torch.randn_like(image, dtype=torch.float32) * visible_std + 0
    noisy_image = image.float() + noise

    if image.dtype == torch.uint8:
        noisy_image = torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    if _gaussian_noise_save_counter < 1:
        noisy_np = noisy_image.detach().cpu().numpy()
        if noisy_np.ndim == 3 and noisy_np.shape[0] in [1,3]:
            noisy_np = noisy_np.transpose(1,2,0)
        import uuid
        os.makedirs('debug_images', exist_ok=True)
        unique_id = str(uuid.uuid4())
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Input Image')
        plt.imshow(img_np.astype('uint8') if img_np.dtype != 'uint8' else img_np)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title('Noisy Image')
        plt.imshow(noisy_np.astype('uint8') if noisy_np.dtype != 'uint8' else noisy_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'debug_images/input_and_noisy_m{mean}_std{visible_std}_{unique_id}.png')
        plt.close()
        _gaussian_noise_save_counter += 1

    noise = torch.randn_like(image, dtype=torch.float32) * std + mean
    noisy_image = image.float() + noise

    if image.dtype == torch.uint8:
        noisy_image = torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    if _gaussian_noise_save_counter < 1:
        # Display and save output image
        noisy_np = noisy_image.detach().cpu().numpy()
        if noisy_np.ndim == 3 and noisy_np.shape[0] in [1,3]:
            noisy_np = noisy_np.transpose(1,2,0)
        plt.figure()
        plt.title('Noisy Image')
        plt.imshow(noisy_np.astype('uint8') if noisy_np.dtype != 'uint8' else noisy_np)
        plt.axis('off')
        plt.savefig(f'debug_images/noisy_m{mean}_std{visible_std}_{unique_id}.png')
        plt.close()
        _gaussian_noise_save_counter += 1

    return noisy_image

def apply_motion_blur(image, kernel_size=45):
    # Handle 5D (B, T, C, H, W), 4D (B, C, H, W), and 3D (C, H, W) tensors
    orig_shape = image.shape
    is_5d = len(orig_shape) == 5
    is_4d = len(orig_shape) == 4
    is_3d = len(orig_shape) == 3

    if is_5d:
        batch_size, timesteps, channels, height, width = orig_shape
        image = image.view(-1, channels, height, width)  # (B*T, C, H, W)
    elif is_3d:
        # Add batch dimension
        image = image.unsqueeze(0)  # (1, C, H, W)

    # Save original for visualization
    original_tensor = image.clone()

    # Now image is (N, C, H, W)
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)

    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size

    # Apply motion blur to each image in the batch
    blurred_np = np.array([cv2.filter2D(img, -1, kernel) for img in image_np])

    # Convert back to PyTorch tensor
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device).float()  # (N, C, H, W)

    # Reshape back if input was 5D
    if is_5d:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)
    elif is_3d:
        blurred_tensor = blurred_tensor.squeeze(0)  # Remove batch dimension

    # If input was uint8, convert output to uint8
    if image.dtype == torch.uint8:
        blurred_tensor = torch.clamp(blurred_tensor, 0, 255).to(torch.uint8)

    return blurred_tensor

def apply_defocus_blur(image, kernel_size=15):
    print(f"Applying defocus blur with kernel size {kernel_size} to image of shape {image.shape}")
    is_batched = len(image.shape) == 5
    is_3d = len(image.shape) == 3
    if is_batched:
        batch_size, timesteps, channels, height, width = image.shape
        image = image.view(-1, channels, height, width)  # Flatten batch & time
    elif is_3d:
        image = image.unsqueeze(0)  # Add batch dimension

    original_tensor = image.clone()

    # Convert to (N, H, W, C) format for OpenCV
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (N, H, W, C)

    # Apply Gaussian blur to each frame
    blurred_np = np.array([cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in image_np])

    # Convert back to PyTorch tensor
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device).float()  # Shape: (N, C, H, W)

    # Reshape back if input was 5D or 3D
    if is_batched:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)
    elif is_3d:
        blurred_tensor = blurred_tensor.squeeze(0)  # Remove batch dimension

    # Ensure output is uint8 if input was uint8, or scale float to uint8
    if image.dtype == torch.uint8:
        blurred_tensor = torch.clamp(blurred_tensor, 0, 255).to(torch.uint8)
    elif blurred_tensor.max() <= 1.0:
        blurred_tensor = (blurred_tensor * 255.0).clamp(0, 255).to(torch.uint8)

    return blurred_tensor

def uneven_illumination(image, strength=0.5):
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

    original_tensor = image.clone()

    # Convert to (N, H, W, C) for OpenCV-like operations
    # N = B*T for batched input, or 1 for single input
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

# Function to generate Perlin noise for corruption
def generate_perlin_noise(height, width, scale=10, intensity=0.5):
    if intensity is None:
        raise ValueError("Error: 'intensity' cannot be None. Please provide a valid float value.")
    
    if not NOISE_MODULE_AVAILABLE:
        # Fallback to simple random noise if noise module is not available
        print("Warning: Using random noise instead of Perlin noise (noise module not available)")
        random_noise = np.random.rand(height, width).astype(np.float32)
        random_noise = random_noise * intensity
        return random_noise

    perlin_noise = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            perlin_noise[i, j] = noise.pnoise2(i / scale, j / scale, octaves=6)

    # Normalize to [0,1] and apply intensity
    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())
    perlin_noise = perlin_noise * intensity
    return perlin_noise

# Function to add realistic corruption (smoke effect)
def add_smoke_effect(image, intensity=0.7):
    """
    Apply a realistic smoke effect to an image tensor while handling different tensor shapes.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor")
    if intensity is None:
        raise ValueError("Error: 'intensity' must be a valid float value.")

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

    assert noise_3ch.shape == image_np.shape, f"Shape mismatch: noise {noise_3ch.shape} vs image {image_np.shape}"
    corrupted = cv2.addWeighted(image_np, 1.0 - intensity, noise_3ch, intensity, 0)
    corrupted = np.clip(corrupted, 0, 1)

    corrupted_tensor = torch.from_numpy(corrupted).permute(0, 3, 1, 2).to(device).float()

    if corrupted_tensor.shape[1] != 3:
        corrupted_tensor = corrupted_tensor[:, :3, :, :]

    if is_sequence:
        corrupted_tensor = corrupted_tensor.view(batch_size // seq_len, seq_len, 3, height, width)

    return corrupted_tensor

# Global counter for verification
corruption_tracker = {"total": 0, "corrupted": 0, "uncorrupted": 0}

def random_corrupt(image):
    """
    Applies a single random corruption to an image with a 50% probability.
    Keeps track of how many images are corrupted.
    """
    corruption_methods = [
        add_gaussian_noise,
        apply_motion_blur,
        apply_defocus_blur,
        uneven_illumination,
        add_smoke_effect
    ]

    corruption_tracker["total"] += 1  # Track total processed images
    if random.random() < 0.5:  # 50% chance to apply corruption
        corruption_method = random.choice(corruption_methods)
        image = corruption_method(image)
        corruption_tracker["corrupted"] += 1  # Track corrupted images
    else:
        corruption_tracker["uncorrupted"] += 1  # Track uncorrupted images
    return image  # Return the (possibly corrupted) image

# Main function to apply corruptions - renamed to 'corrupt' to match what the preprocessor expects
def corrupt(image, corruption_type):
    """
    Apply the specified corruption type to the image.
    
    Args:
        image: PyTorch tensor of shape (B, C, H, W) or (B, T, C, H, W)
        corruption_type: String specifying which corruption to apply
        
    Returns:
        Corrupted image with the same shape as input
    """
    # Ensure input is a torch.Tensor; convert if needed
    img_was_numpy = False
    if isinstance(image, np.ndarray):
        img_was_numpy = True
        image = torch.from_numpy(image)
    # Dispatch to the correct corruption function
    if corruption_type == 'gaussian_noise':
        out = add_gaussian_noise(image)
    elif corruption_type == 'motion_blur':
        out = apply_motion_blur(image)
    elif corruption_type == 'defocus_blur':
        out = apply_defocus_blur(image)
    elif corruption_type == 'uneven_illumination':
        out = uneven_illumination(image)
    elif corruption_type == 'smoke_effect':
        out = add_smoke_effect(image, intensity=0.7)
    elif corruption_type == 'random' or corruption_type == 'random_corruptions':
        out = random_corrupt(image)
    elif corruption_type == 'none' or corruption_type is None:
        out = image
    else:
        raise ValueError(f"Invalid corruption type '{corruption_type}'. Choose from: 'gaussian_noise', 'motion_blur', 'defocus_blur', 'uneven_illumination', 'smoke_effect', 'random_corruptions', 'none'.")
    # Convert back to numpy if original input was numpy
    if img_was_numpy:
        out = out.detach().cpu().numpy()
    return out

# Keep the original 'corruption' function for backwards compatibility
def corruption(image, corruption_type):
    """Alias for the corrupt function to maintain backward compatibility"""
    return corrupt(image, corruption_type)