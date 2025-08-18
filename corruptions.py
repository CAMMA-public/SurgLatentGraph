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

test_corruption = os.environ.get('TEST_CORRUPTION', 'none')
print(f"Using test corruption type: {test_corruption}")

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_gaussian_noise(image, mean=5, std=0.5):
    # Stop immediately if image is empty
    if any(dim == 0 for dim in image.shape):
        raise ValueError(f"add_gaussian_noise: Received empty image with shape {image.shape} and dtype {image.dtype}. Stopping.")
    """
    Adds Gaussian noise to a PyTorch image tensor without changing its shape or type.
    """
    print(f"ADDING GAUSSIAN NOISE WITH MEAN={mean}, STD={std} TO IMAGE OF SHAPE {image.shape}")
    print(f'Using g_n {test_corruption}')

    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    import uuid
    global _gaussian_noise_save_counter
    if _gaussian_noise_save_counter < 1:
        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in [1,3]:
            img_np = img_np.transpose(1,2,0)

    # Increase std to make noise visible
    visible_std = 50.0 if image.dtype == torch.uint8 else 0.2
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
        plt.savefig(f'debug_images/input_and_noisy_test_m{mean}_std{visible_std}_{unique_id}.png')
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
    # Stop immediately if image is empty
    if any(dim == 0 for dim in image.shape):
        raise ValueError(f"apply_motion_blur: Received empty image with shape {image.shape} and dtype {image.dtype}. Stopping.")
    # Stop immediately if image is empty
    if any(dim == 0 for dim in image.shape):
        raise ValueError(f"apply_defocus_blur: Received empty image with shape {image.shape} and dtype {image.dtype}. Stopping.")
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
    # Save original for visualization
    original_tensor = image.clone()

    # Now image is (N, C, H, W)
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)
    orig_dtype = image.dtype
    mx = np.max(image_np) if image_np.size else 1.0
    # Convert to uint8 for visible effect
    if image_np.dtype != np.uint8:
        if mx <= 1.0:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size

    # Apply motion blur to each image in the batch, per channel
    blurred_np = []
    for img in image_np:
        if img.shape[-1] == 1:
            # Grayscale
            blurred = cv2.filter2D(img.squeeze(-1), -1, kernel)
            blurred = np.expand_dims(blurred, axis=-1)
        else:
            # Color: apply kernel per channel
            channels = [cv2.filter2D(img[..., c], -1, kernel) for c in range(img.shape[-1])]
            blurred = np.stack(channels, axis=-1)
        blurred_np.append(blurred)
    blurred_np = np.stack(blurred_np, axis=0)

    # Convert back to PyTorch tensor
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device)
    # Convert back to original dtype/range
    if orig_dtype == torch.uint8:
        blurred_tensor = blurred_tensor.to(torch.uint8)
    else:
        blurred_tensor = blurred_tensor.float() / 255.0

    # Reshape back if input was 5D
    if is_5d:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)
    elif is_3d:
        blurred_tensor = blurred_tensor.squeeze(0)  # Remove batch dimension

    # --- Plot and save debug image (only once per session) ---
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    global _motion_blur_save_counter
    if '_motion_blur_save_counter' not in globals():
        _motion_blur_save_counter = 0
    if _motion_blur_save_counter < 1:
        def to_disp_np(t):
            arr = t.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return arr

        inp_np = to_disp_np(original_tensor[0])
        if blurred_tensor.ndim == 4:
            blur_np = to_disp_np(blurred_tensor[0])
        else:
            blur_np = to_disp_np(blurred_tensor)

        os.makedirs('debug_images', exist_ok=True)
        unique_id = str(uuid.uuid4())
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(inp_np)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Motion Blurred Image')
        plt.imshow(blur_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'debug_images/input_and_motion_blur_ks_{kernel_size}_{unique_id}.png')
        plt.close()
        _motion_blur_save_counter += 1
    # --------------------------------------------------------

    # return blurred_tensor
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.title('Input Image')
    #     plt.imshow(inp_np)
    #     plt.axis('off')
    #     plt.subplot(1, 2, 2)
    #     plt.title('Motion Blurred Image')
    #     plt.imshow(blur_np)
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'debug_images/input_and_motion_blur_ks_3_{kernel_size}_{unique_id}.png')
    #     plt.close()
    #     _motion_blur_save_counter += 1
    # --------------------------------------------------------

    return blurred_tensor

def apply_defocus_blur(image, kernel_size=21): # change kernel size with odd numbers to change the intensity of the effect.
    print(f"Applying defocus blur with kernel size {kernel_size} to image of shape {image.shape}")
    is_batched = len(image.shape) == 5
    is_3d = len(image.shape) == 3
    if is_batched:
        batch_size, timesteps, channels, height, width = image.shape
        image = image.view(-1, channels, height, width)  # Flatten batch & time
    elif is_3d:
        image = image.unsqueeze(0)  # Add batch dimension

    original_tensor = image.clone()



    # Convert to (N, H, W, C) format for OpenCV, ensure uint8
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)
    if image_np.dtype != np.uint8:
        mx = np.max(image_np) if image_np.size else 1.0
        if mx <= 1.0:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # If input is single channel, convert to 3 channel, else keep as is
    if image_np.shape[-1] == 1:
        image_np_color = np.repeat(image_np, 3, axis=-1)
    else:
        image_np_color = image_np

    # Convert RGB to BGR for OpenCV (ensure positive strides)
    image_np_bgr = image_np_color[..., ::-1].copy()

    # Apply Gaussian blur to each frame (preserve color)
    blurred_np_bgr = []
    for img in image_np_bgr:
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        # If result is 2D (grayscale), convert to 3 channel
        if blurred.ndim == 2:
            blurred = np.stack([blurred]*3, axis=-1)
        blurred_np_bgr.append(blurred)
    blurred_np_bgr = np.stack(blurred_np_bgr, axis=0)

    # Convert BGR back to RGB (ensure positive strides)
    blurred_np = blurred_np_bgr[..., ::-1].copy()

    # Convert back to PyTorch tensor, keep as uint8
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device)

    # Reshape back if input was 5D or 3D
    if is_batched:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)
    elif is_3d:
        blurred_tensor = blurred_tensor.squeeze(0)  # Remove batch dimension

    # --- Plot and save debug image (only once per session) ---
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    import matplotlib.pyplot as plt
    import uuid
    global _defocus_blur_save_counter
    if '_defocus_blur_save_counter' not in globals():
        _defocus_blur_save_counter = 0
    if _defocus_blur_save_counter < 1:
        def to_disp_np(t):
            arr = t.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)
            if arr.dtype != np.uint8:
                mx = np.max(arr) if arr.size else 1.0
                if mx <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

        inp_np = to_disp_np(original_tensor[0])
        if blurred_tensor.ndim == 4:
            blur_np = to_disp_np(blurred_tensor[0])
        else:
            blur_np = to_disp_np(blurred_tensor)

        os.makedirs('debug_images', exist_ok=True)
        unique_id = str(uuid.uuid4())
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(inp_np)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Defocus Blurred Image')
        plt.imshow(blur_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'debug_images/input_and_defocus_blur11_ks{kernel_size}_{unique_id}.png')
        plt.close()
        _defocus_blur_save_counter += 1
    # --------------------------------------------------------

    return blurred_tensor

def uneven_illumination(image, strength=0.8):
    # Stop immediately if image is empty
    if any(dim == 0 for dim in image.shape):
        raise ValueError(f"uneven_illumination: Received empty image with shape {image.shape} and dtype {image.dtype}. Stopping.")
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    import matplotlib.pyplot as plt
    import uuid
    global _uneven_illumination_save_counter
    if '_uneven_illumination_save_counter' not in globals():
        _uneven_illumination_save_counter = 0
    print("Uneven_illumination is added.")
    is_batched = (image.dim() == 5)
    added_batch = False
    orig_dtype = image.dtype
    # Always work in float32 [0,1] for processing
    if orig_dtype == torch.uint8:
        image = image.float() / 255.0
    if is_batched:
        b, t, c, h, w = image.shape
        image = image.view(-1, c, h, w)
    elif image.dim() == 3:
        image = image.unsqueeze(0)
        added_batch = True
        c, h, w = image.shape[1:]
        b, t = 1, 1
    else:
        c, h, w = image.shape[1:]
        b, t = image.shape[0], 1

    original_tensor = image.clone()
    image_np = image.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    N = image_np.shape[0]
    result_list = []
    min_grad = max(1 - strength, 0.7)
    for i in range(N):
        frame = image_np[i]
        h_i, w_i, c_i = frame.shape
        gradient = np.linspace(min_grad, 1, w_i, dtype=np.float32)
        gradient = np.tile(gradient, (h_i, 1)).reshape(h_i, w_i, 1)
        illuminated = frame * gradient
        illuminated = np.clip(illuminated, 0, 1)
        result_list.append(illuminated)
    result_np = np.stack(result_list, axis=0)
    result_tensor = torch.from_numpy(result_np).permute(0, 3, 1, 2).to(image.device)
    # Convert back to original dtype if needed
    if orig_dtype == torch.uint8:
        result_tensor = torch.clamp(result_tensor * 255.0, 0, 255).to(torch.uint8)
    else:
        result_tensor = torch.clamp(result_tensor, 0, 1)
    if is_batched:
        result_tensor = result_tensor.view(b, t, c, h, w)
    elif added_batch:
        result_tensor = result_tensor.squeeze(0)
    # --- Plot and save debug image (only once per session) ---
    if _uneven_illumination_save_counter < 1:
        def to_disp_np(t):
            arr = t.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(-1)
            # Always scale to [0,255] for display
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return arr
        inp_np = to_disp_np(original_tensor[0])
        if result_tensor.ndim == 4:
            out_np = to_disp_np(result_tensor[0])
        else:
            out_np = to_disp_np(result_tensor)
        os.makedirs('debug_images', exist_ok=True)
        unique_id = str(uuid.uuid4())
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(inp_np)  # No channel swap, display as imported
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Uneven Illuminated Image')
        plt.imshow(out_np)  # No channel swap, display as imported
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'debug_images/input_and_uneven_illumination_6_{unique_id}.png')
        plt.close()
        _uneven_illumination_save_counter += 1
    # --------------------------------------------------------
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
    if any(dim == 0 for dim in image.shape):
        print(f"add_smoke_effect: Received empty image with shape {image.shape} and dtype {image.dtype}. Stopping and returning None.")
        return None
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor")
    if intensity is None:
        raise ValueError("Error: 'intensity' must be a valid float value.")
    print(f"Adding smoke effect with intensity {intensity} to image of shape {image.shape}")
    image = image.to('cpu').contiguous()
    orig_shape = image.shape
    orig_dtype = image.dtype
    squeeze_batch = False
    is_sequence = False
    if image.dim() == 5:
        b, t, c, h, w = image.shape
        image = image.view(-1, c, h, w)
        is_sequence = True
    elif image.dim() == 4:
        b, c, h, w = image.shape
    elif image.dim() == 3:
        c, h, w = image.shape
        image = image.unsqueeze(0)
        b = 1
        squeeze_batch = True
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if orig_dtype == torch.uint8:
        image_f32 = image.float() / 255.0
    else:
        image_f32 = image.float().clamp(0, 1)
    image_np = image_f32.permute(0, 2, 3, 1).cpu().numpy()
    b, h, w, c = image_np.shape
    if min(h, w, c) <= 0:
        print(f"Smoke effect received invalid image shape: {image_np.shape}. Stopping and returning None.")
        return None
    # Create a more diffuse, uniform smoke effect
    # Use higher scale for Perlin noise and higher sigma for Gaussian filter
    diffuse_scale = 100  # Higher scale for larger, smoother noise features
    diffuse_sigma = 15   # Higher sigma for more diffusion
    # Optionally, add a random offset for each image in the batch
    noise_3ch = np.zeros((b, h, w, c), dtype=np.float32)
    for i in range(b):
        offset_x = np.random.randint(0, diffuse_scale)
        offset_y = np.random.randint(0, diffuse_scale)
        noise_pattern = generate_perlin_noise(h, w, scale=diffuse_scale, intensity=intensity)
        # Shift the noise pattern for each image for more variety
        noise_pattern = np.roll(noise_pattern, shift=offset_x, axis=0)
        noise_pattern = np.roll(noise_pattern, shift=offset_y, axis=1)
        noise_img = np.stack([noise_pattern] * c, axis=2)
        noise_img = gaussian_filter(noise_img, sigma=diffuse_sigma)
        noise_3ch[i] = noise_img
    corrupted = cv2.addWeighted(image_np.astype(np.float32), 1.0 - intensity, noise_3ch.astype(np.float32), intensity, 0)
    corrupted = np.clip(corrupted, 0, 1)
    corrupted_tensor = torch.from_numpy(corrupted).permute(0, 3, 1, 2).contiguous()
    if orig_dtype == torch.uint8:
        corrupted_tensor = (corrupted_tensor * 255.0).clamp(0, 255).to(torch.uint8)
    else:
        corrupted_tensor = corrupted_tensor.float().clamp(0, 1)
    if is_sequence:
        corrupted_tensor = corrupted_tensor.view(orig_shape)
    elif squeeze_batch:
        corrupted_tensor = corrupted_tensor.squeeze(0)
    out_shape = tuple(corrupted_tensor.shape)
    if any(dim == 0 for dim in out_shape):
        print(f"Smoke effect produced empty image with shape {out_shape} and dtype {corrupted_tensor.dtype}. Stopping and returning None.")
        return None
    if torch.isnan(corrupted_tensor).any():
        print(f"Smoke effect produced image with NaNs. Shape: {out_shape}, dtype: {corrupted_tensor.dtype}. Stopping and returning None.")
        return None
    min_val = corrupted_tensor.min().item() if corrupted_tensor.numel() > 0 else None
    max_val = corrupted_tensor.max().item() if corrupted_tensor.numel() > 0 else None
    print(f"Smoke effect output shape: {out_shape}, dtype: {corrupted_tensor.dtype}, min: {min_val}, max: {max_val}")
    if min_val == 0 and max_val == 0:
        print(f"Smoke effect produced all-zero image. Shape: {out_shape}, dtype: {corrupted_tensor.dtype}. Stopping and returning None.")
        return None
    # Now do plotting after all error checks
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    global _smoke_effect_save_counter
    if '_smoke_effect_save_counter' not in globals():
        _smoke_effect_save_counter = 0
    if _smoke_effect_save_counter < 1:
        inp_np = image.detach().cpu().numpy()
        out_np = corrupted_tensor.detach().cpu().numpy()
        # If batch dimension exists, select the first image
        if inp_np.ndim == 4:
            inp_np = inp_np[0]
        if out_np.ndim == 4:
            out_np = out_np[0]
        # If channel-first, transpose to HWC
        if inp_np.ndim == 3 and inp_np.shape[0] in [1,3]:
            inp_np = inp_np.transpose(1,2,0)
        if out_np.ndim == 3 and out_np.shape[0] in [1,3]:
            out_np = out_np.transpose(1,2,0)
        os.makedirs('debug_images/s_e', exist_ok=True)
        unique_id = str(uuid.uuid4())
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Input Image')
        plt.imshow(inp_np.astype('uint8') if inp_np.dtype != 'uint8' else inp_np)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title('Smoke Effect Output')
        plt.imshow(out_np.astype('uint8') if out_np.dtype != 'uint8' else out_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'debug_images/s_e/input_and_smoke_effect_{unique_id}.png')
        plt.close()
        _smoke_effect_save_counter += 1
    # --------------------------------------------------------
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