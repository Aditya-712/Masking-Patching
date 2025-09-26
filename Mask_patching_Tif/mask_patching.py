"""
This module handles the processing of binary mask images (typically used for 
segmentation ground truth) in GeoTIFF format. It provides functionality to:
- Split large binary mask images into smaller patches
- Handle overlapping patches for better segmentation results
- Preserve geospatial metadata
- Process single-channel binary masks (0 for background, 255 for foreground)
- Optionally reconstruct the original mask from patches

This is particularly useful for preparing training data for semantic 
segmentation models where the input images are too large to process at once.
"""

import os
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

def pad_mask(image_array: np.ndarray, patch_size: int) -> tuple:
    """
    Pads a binary mask image to ensure it can be evenly divided into patches.
    
    Args:
        image_array (np.ndarray): Input binary mask array of shape (height, width)
        patch_size (int): Size of the patches to be created
        
    Returns:
        tuple: Contains:
            - padded_image (np.ndarray): Padded binary mask array
            - pad_y (int): Amount of padding added in y direction
            - pad_x (int): Amount of padding added in x direction
            
    Note:
        Uses zero padding which maintains the binary nature of the mask
    """
    img_height, img_width = image_array.shape 
    pad_y = (patch_size - img_height % patch_size) % patch_size
    pad_x = (patch_size - img_width % patch_size) % patch_size
    padded_image = np.pad(image_array, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
    return padded_image, pad_y, pad_x 

def create_mask_patches(input_image_path: str, patch_size: int, overlap_percent: float) -> tuple:
    """
    Creates overlapping patches from a binary mask GeoTIFF image.
    
    Args:
        input_image_path (str): Path to the input binary mask GeoTIFF
        patch_size (int): Size of each square patch
        overlap_percent (float): Percentage of overlap between adjacent patches
        
    Returns:
        tuple: Contains:
            - patches (list): List of tuples containing (patch_image, (x, y))
            - img_size (tuple): Original image dimensions (width, height)
            - metadata (dict): GeoTIFF metadata
            - pad_y (int): Vertical padding added
            - pad_x (int): Horizontal padding added
            
    Note:
        - Reads only the first band of the input image
        - Converts any non-zero values to 255 for binary mask consistency
    """
    with rasterio.open(input_image_path) as src:
        img_width = src.width
        img_height = src.height
        stride = int(patch_size * (1 - overlap_percent / 100))  
        patches = []

        image_array = src.read(1)  

        # Invert the mask values to get correct black/white mapping
        image_array = np.where(image_array > 0, 0, 255).astype(np.uint8)

        padded_image, pad_y, pad_x = pad_mask(image_array, patch_size)

        for y in range(0, padded_image.shape[0] - patch_size + 1, stride):
            for x in range(0, padded_image.shape[1] - patch_size + 1, stride):
                patch_array = padded_image[y:y + patch_size, x:x + patch_size]
                patch = Image.fromarray(patch_array)  
                patches.append((patch, (x, y)))  
        
        return patches, (img_width, img_height), src.meta, pad_y, pad_x

def save_mask_patches(patches, output_folder, metadata, patch_size, image_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (patch, (x, y)) in enumerate(patches):
        patch_filename = os.path.join(output_folder, f"{image_name}_{x}_{y}.tif")
        
        patch_metadata = metadata.copy()
        patch_metadata.update({
            'width': patch_size,
            'height': patch_size,
            'count': 1, 
            'dtype': 'uint8',
            'crs': metadata['crs'], 
            'transform': rasterio.transform.Affine(1, 0, x, 0, -1, y)  
        })
        
        with rasterio.open(patch_filename, 'w', **patch_metadata) as dst:
            dst.write(np.array(patch)[np.newaxis, :, :])  

def reconstruct_mask(patches, image_size, patch_size, overlap_percent, pad_y=0, pad_x=0):
    img_width, img_height = image_size
    stride = int(patch_size * (1 - overlap_percent / 100))  
    
    reconstructed_image = np.zeros((img_height + pad_y, img_width + pad_x), dtype=np.float32)
    patch_count = np.zeros((img_height + pad_y, img_width + pad_x), dtype=np.uint8)

    for patch, (x, y) in patches:
        patch_array = np.array(patch, dtype=np.float32)
        reconstructed_image[y:y + patch_size, x:x + patch_size] += patch_array
        patch_count[y:y + patch_size, x:x + patch_size] += 1
    
    reconstructed_image = reconstructed_image / patch_count
    
    # Maintain the correct black/white mapping during reconstruction
    reconstructed_image = np.where(reconstructed_image > 127, 0, 255).astype(np.uint8)
    
    reconstructed_image = reconstructed_image[:img_height, :img_width]

    return reconstructed_image

def save_reconstructed_mask(reconstructed_image, output_image_path, metadata):
    metadata.update(dtype='uint8', count=1, compress='lzw')  

    with rasterio.open(output_image_path, 'w', **metadata) as dst:
        dst.write(reconstructed_image[np.newaxis, :, :]) 

def mask_patching(input_image_path, patch_size, output_folder, overlap_percent):
    patches, img_size, metadata, pad_y, pad_x = create_mask_patches(input_image_path, patch_size, overlap_percent)
    os.makedirs(output_folder, exist_ok=True)
    save_mask_patches(patches, output_folder, metadata, patch_size, os.path.basename(input_image_path).split('.')[0])

    # reconstructed_image = reconstruct_mask(patches, img_size, patch_size, overlap_percent=50, pad_y=pad_y, pad_x=pad_x)
    
    # reconstructed_image_path = os.path.join(output_folder, "reconstructed_image.tif")
    # save_reconstructed_mask(reconstructed_image, reconstructed_image_path, metadata)
    # print(f"Reconstructed image saved as '{reconstructed_image_path}'.")

if __name__ == "__main__":
    patch_size = 640
    # src = 'data/chakan/src/mask'
    
    # dest = 'data/chakan/patches/src/sanket_masks'
    
    src = '/home/computervision/Documents/gis/building_detection/unet/torch/aditya_test/Input_samples/tiff/BN_Image.tif'

    dest = '/home/computervision/Documents/gis/building_detection/unet/torch/aditya_test/Masking/masking_output'

    for i in os.listdir(src):
        print(i)
        mask_patching(
            os.path.join(src, i),
            patch_size,
            dest,
            overlap_percent=50
        )
    print(len(os.listdir(dest))) 
    # print(len(os.listdir('data/patches/train/images')))