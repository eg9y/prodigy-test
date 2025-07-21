#!/usr/bin/env python3
"""
Utility script to visualize the preprocessing (cropping) applied to images.
"""

import sys
import os
from PIL import Image
from image_preprocessor import crop_bottom_center


def show_preprocessing_comparison(image_path: str, output_dir: str = "preprocessing_demo"):
    """Show original vs preprocessed image side by side."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    original = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Apply preprocessing
    cropped = crop_bottom_center(image_path, crop_height_ratio=0.3, crop_width_ratio=0.5)
    
    # Save comparison images
    original_path = os.path.join(output_dir, f"{image_name}_original.jpg")
    cropped_path = os.path.join(output_dir, f"{image_name}_cropped.jpg")
    
    original.save(original_path)
    cropped.save(cropped_path)
    
    print(f"Original image saved to: {original_path}")
    print(f"Cropped image saved to: {cropped_path}")
    print(f"Original size: {original.size}")
    print(f"Cropped size: {cropped.size}")
    
    # Calculate crop region info
    width, height = original.size
    crop_height = int(height * 0.3)
    crop_width = int(width * 0.5)
    left = (width - crop_width) // 2
    top = height - crop_height
    
    print(f"\nCrop region:")
    print(f"- Takes bottom 30% of height ({crop_height} pixels)")
    print(f"- Takes center 50% of width ({crop_width} pixels)")
    print(f"- Crop box: left={left}, top={top}, right={left + crop_width}, bottom={height}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python show_preprocessing.py <image_path>")
        print("Example: python show_preprocessing.py data/screenshots/src.png")
        return
    
    image_path = sys.argv[1]
    show_preprocessing_comparison(image_path)


if __name__ == "__main__":
    main()