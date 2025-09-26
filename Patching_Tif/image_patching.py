"""
This module handles the process of splitting large GeoTIFF images into smaller patches
with optional overlap. It's designed for processing large satellite/aerial imagery
that is too large to process at once. The module supports:
- Creating patches from large images with configurable overlap
- Preserving geospatial metadata
- Supporting RGB images
- Reconstructing the original image from patches (optional)
"""

import os
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

def pad_image(image_array: np.ndarray, patch_size: int) -> tuple:
    """
    Pads the input image to ensure it can be evenly divided into patches.
    
    Args:
        image_array (np.ndarray): Input image array of shape (channels, height, width)
        patch_size (int): Size of the patches to be created
        
    Returns:
        tuple: Contains:
            - padded_image (np.ndarray): Padded image array
            - pad_y (int): Amount of padding added in y direction
            - pad_x (int): Amount of padding added in x direction
    """
    img_height, img_width = image_array.shape[1], image_array.shape[2]  
    pad_y = (patch_size - img_height % patch_size) % patch_size
    pad_x = (patch_size - img_width % patch_size) % patch_size
    padded_image = np.pad(image_array, ((0, 0), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
    return padded_image, pad_y, pad_x  

def create_image_patches(input_image_path: str, patch_size: int, overlap_percent: float) -> tuple:
    """
    Creates overlapping patches from a large GeoTIFF image.
    
    Args:
        input_image_path (str): Path to the input GeoTIFF image
        patch_size (int): Size of each square patch
        overlap_percent (float): Percentage of overlap between adjacent patches
        
    Returns:
        tuple: Contains:
            - patches (list): List of tuples containing (patch_image, (x, y))
            - img_size (tuple): Original image dimensions (width, height)
            - metadata (dict): GeoTIFF metadata
            - pad_y (int): Vertical padding added
            - pad_x (int): Horizontal padding added
    """
    with rasterio.open(input_image_path) as src:
        img_width = src.width
        img_height = src.height
        stride = int(patch_size * (1 - overlap_percent / 100))  
        patches = []

        image_array = src.read()
        
        padded_image, pad_y, pad_x = pad_image(image_array, patch_size)

        for y in range(0, padded_image.shape[1] - patch_size + 1, stride):
            for x in range(0, padded_image.shape[2] - patch_size + 1, stride):
                patch_array = padded_image[:, y:y + patch_size, x:x + patch_size]
                patch = Image.fromarray(np.moveaxis(patch_array, 0, -1))  
                patches.append((patch, (x, y)))  
        
        return patches, (img_width, img_height), src.meta, pad_y, pad_x

def save_image_patches(patches, output_folder, metadata, patch_size, image_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (patch, (x, y)) in enumerate(patches):
        patch_filename = os.path.join(output_folder, f"{image_name}_{x}_{y}.tif")
        
        patch_metadata = metadata.copy()
        patch_metadata.update({
            'width': patch_size,
            'height': patch_size,
            'count': 3,  
            'dtype': 'uint8',
            'crs': metadata['crs'], 
            'transform': rasterio.transform.Affine(1, 0, x, 0, -1, y) 
        })
        
        with rasterio.open(patch_filename, 'w', **patch_metadata) as dst:
            dst.write(np.moveaxis(np.array(patch), -1, 0))  

def reconstruct_image(patches, image_size, patch_size, overlap_percent, pad_y=0, pad_x=0):
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

def save_reconstructed_image(reconstructed_image, output_image_path, metadata):
    metadata.update(dtype='uint8', count=3, compress='lzw')  

    with rasterio.open(output_image_path, 'w', **metadata) as dst:
        dst.write(np.moveaxis(reconstructed_image, -1, 0))  

def image_patching(input_image_path, patch_size, output_folder, overlap_percent):
    patches, img_size, metadata, pad_y, pad_x = create_image_patches(input_image_path, patch_size, overlap_percent)
    os.makedirs(output_folder, exist_ok=True)
    save_image_patches(patches, output_folder, metadata, patch_size, os.path.basename(input_image_path).split('.')[0])

    # reconstructed_image = reconstruct_image(patches, img_size, patch_size, overlap_percent=50, pad_y=pad_y, pad_x=pad_x)
    
    # reconstructed_image_path = os.path.join(output_folder, "reconstructed_image.tif")
    # save_reconstructed_image(reconstructed_image, reconstructed_image_path, metadata)
    # print(f"Reconstructed image saved as '{reconstructed_image_path}'.")


if __name__ == "__main__":

    patch_size=640
    
    # src = 'data/chakan/src'
    # dest = 'data/chakan/patches/images'
    
    src = '/home/computervision/Documents/gis/building_detection/unet/torch/aditya_test/Patching_JPG/patching_input'
    dest = '/home/computervision/Documents/gis/building_detection/unet/torch/aditya_test/Patching_JPG/patching_output'
    
    os.makedirs(dest, exist_ok = True)	
    for i in os.listdir(src):
        print(i)
        image_patching(
            os.path.join(src, i),
            patch_size,
            dest,
            overlap_percent=50
        )
    print(len(os.listdir(dest))) 
