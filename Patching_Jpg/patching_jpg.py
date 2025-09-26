"""
This module handles the process of splitting large JPG/PNG images into smaller patches
with optional overlap. It supports both JPG and PNG formats and will convert PNGs
to JPG format during processing. The module supports:
- Creating patches from large images with configurable overlap
- Supporting RGB images
- Handling both JPG and PNG input formats
- Converting PNG to JPG if needed
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def pad_image(image_array: np.ndarray, patch_size: int) -> tuple:
    """
    Pads the input image to ensure it can be evenly divided into patches.
    
    Args:
        image_array (np.ndarray): Input image array of shape (height, width, channels)
        patch_size (int): Size of the patches to be created
        
    Returns:
        tuple: Contains:
            - padded_image (np.ndarray): Padded image array
            - pad_y (int): Amount of padding added in y direction
            - pad_x (int): Amount of padding added in x direction
    """
    img_height, img_width = image_array.shape[0], image_array.shape[1]
    pad_y = (patch_size - img_height % patch_size) % patch_size
    pad_x = (patch_size - img_width % patch_size) % patch_size
    padded_image = np.pad(image_array, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=0)
    return padded_image, pad_y, pad_x

def create_image_patches(input_image_path: str, patch_size: int, overlap_percent: float) -> tuple:
    """
    Creates overlapping patches from a JPG/PNG image.
    
    Args:
        input_image_path (str): Path to the input image (JPG or PNG)
        patch_size (int): Size of each square patch
        overlap_percent (float): Percentage of overlap between adjacent patches
        
    Returns:
        tuple: Contains:
            - patches (list): List of tuples containing (patch_image, (x, y))
            - img_size (tuple): Original image dimensions (width, height)
            - pad_y (int): Vertical padding added
            - pad_x (int): Horizontal padding added
    """
    # Open and convert image to RGB if needed
    image = Image.open(input_image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_width, img_height = image.size
    stride = int(patch_size * (1 - overlap_percent / 100))
    patches = []

    # Convert to numpy array
    image_array = np.array(image)
    
    padded_image, pad_y, pad_x = pad_image(image_array, patch_size)

    for y in range(0, padded_image.shape[0] - patch_size + 1, stride):
        for x in range(0, padded_image.shape[1] - patch_size + 1, stride):
            patch_array = padded_image[y:y + patch_size, x:x + patch_size]
            patch = Image.fromarray(patch_array)
            patches.append((patch, (x, y)))
    
    return patches, (img_width, img_height), pad_y, pad_x

def save_image_patches(patches, output_folder, image_name):
    """
    Saves the image patches as JPG files.
    
    Args:
        patches (list): List of tuples containing (patch_image, (x, y))
        output_folder (str): Directory where patches will be saved
        image_name (str): Base name for the patch files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for patch, (x, y) in patches:
        patch_filename = os.path.join(output_folder, f"{image_name}_{x}_{y}.jpg")
        patch.save(patch_filename, 'JPEG', quality=95)

def reconstruct_image(patches, image_size, patch_size, overlap_percent, pad_y=0, pad_x=0):
    """
    Reconstructs the original image from patches.
    
    Args:
        patches (list): List of patches with their positions
        image_size (tuple): Original image dimensions (width, height)
        patch_size (int): Size of each patch
        overlap_percent (float): Overlap percentage used in patching
        pad_y (int): Vertical padding added
        pad_x (int): Horizontal padding added
        
    Returns:
        np.ndarray: Reconstructed image array
    """
    img_width, img_height = image_size
    stride = int(patch_size * (1 - overlap_percent / 100))
    
    reconstructed_image = np.zeros((img_height + pad_y, img_width + pad_x, 3), dtype=np.float32)
    patch_count = np.zeros((img_height + pad_y, img_width + pad_x), dtype=np.uint8)

    for patch, (x, y) in patches:
        patch_array = np.array(patch, dtype=np.float32)
        reconstructed_image[y:y + patch_size, x:x + patch_size] += patch_array
        patch_count[y:y + patch_size, x:x + patch_size] += 1
    
    reconstructed_image = reconstructed_image / patch_count[:, :, None]
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    reconstructed_image = reconstructed_image[:img_height, :img_width]
    
    return reconstructed_image

def save_reconstructed_image(reconstructed_image, output_image_path):
    """
    Saves the reconstructed image as a JPG file.
    
    Args:
        reconstructed_image (np.ndarray): The reconstructed image array
        output_image_path (str): Path where the reconstructed image will be saved
    """
    Image.fromarray(reconstructed_image).save(output_image_path, 'JPEG', quality=95)

def image_patching(input_image_path, patch_size, output_folder, overlap_percent):
    """
    Main function to handle the image patching process.
    
    Args:
        input_image_path (str): Path to input image (JPG or PNG)
        patch_size (int): Size of patches
        output_folder (str): Output directory for patches
        overlap_percent (float): Overlap percentage between patches
    """
    patches, img_size, pad_y, pad_x = create_image_patches(input_image_path, patch_size, overlap_percent)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get base name without extension and convert to jpg if it's png
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    save_image_patches(patches, output_folder, base_name)

    # Optional reconstruction (commented out by default)
    # reconstructed_image = reconstruct_image(patches, img_size, patch_size, overlap_percent, pad_y, pad_x)
    # reconstructed_image_path = os.path.join(output_folder, f"reconstructed_{base_name}.jpg")
    # save_reconstructed_image(reconstructed_image, reconstructed_image_path)
    # print(f"Reconstructed image saved as '{reconstructed_image_path}'.")

if __name__ == "__main__":
    patch_size = 640
    
    src = '/home/computervision/Documents/gis/building_detection/unet/torch/aditya_test/Input_samples/jpg/palmtrees/image'
    dest = '/home/computervision/Documents/gis/building_detection/unet/torch/aditya_test/Patching_Jpg/patching_output'
    
    os.makedirs(dest, exist_ok=True)
    
    # Process all jpg and png files in the source directory
    for i in os.listdir(src):
        if i.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing {i}")
            image_patching(
                os.path.join(src, i),
                patch_size,
                dest,
                overlap_percent=50
            )
    print(f"Total patches created: {len(os.listdir(dest))}")
