#!/usr/bin/env python3
"""
Image preprocessing module for goalkeeper detection
Crops the bottom center of images where the goalkeeper UI element typically appears
"""

from pathlib import Path
from PIL import Image
import shutil

def crop_bottom_center(image_path, output_path=None, crop_height_ratio=0.3, crop_width_ratio=0.5):
    """
    Crop the bottom center portion of an image
    
    Args:
        image_path: Path to input image
        output_path: Path to save cropped image (if None, returns PIL Image)
        crop_height_ratio: What proportion of height to keep from bottom (default 0.3 = bottom 30%)
        crop_width_ratio: What proportion of width to keep from center (default 0.5 = center 50%)
    
    Returns:
        PIL Image object of cropped image
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # Calculate crop dimensions
    crop_height = int(height * crop_height_ratio)
    crop_width = int(width * crop_width_ratio)
    
    # Calculate crop box (left, top, right, bottom)
    left = (width - crop_width) // 2
    top = height - crop_height
    right = left + crop_width
    bottom = height
    
    # Crop the image
    cropped = img.crop((left, top, right, bottom))
    
    if output_path:
        cropped.save(output_path)
    
    return cropped

def preprocess_directory(input_dir="data/screenshots", output_dir="data/screenshots_cropped", 
                        crop_height_ratio=0.3, crop_width_ratio=0.5):
    """
    Preprocess all images in a directory by cropping bottom center
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save cropped images
        crop_height_ratio: What proportion of height to keep from bottom
        crop_width_ratio: What proportion of width to keep from center
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to preprocess")
    
    # Process each image
    for img_path in image_files:
        output_file = output_path / img_path.name
        try:
            crop_bottom_center(img_path, output_file, crop_height_ratio, crop_width_ratio)
            print(f"Processed: {img_path.name}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    print(f"\nPreprocessing complete! Cropped images saved to: {output_dir}")
    return output_path

def cleanup_preprocessed(output_dir="data/screenshots_cropped"):
    """Remove preprocessed directory"""
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleaned up: {output_dir}")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "cleanup":
            cleanup_preprocessed()
        else:
            # Custom crop ratios can be passed as arguments
            height_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
            width_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
            preprocess_directory(crop_height_ratio=height_ratio, crop_width_ratio=width_ratio)
    else:
        preprocess_directory()