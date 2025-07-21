#!/usr/bin/env python3
"""
Goalkeeper Detection Annotation Script for Prodigy

This script helps set up image classification annotation for detecting
whether a player is a goalkeeper in Rematch gameplay videos.
"""

import subprocess
import sys
import os
from pathlib import Path
from image_preprocessor import preprocess_directory, cleanup_preprocessed

def get_prodigy_command():
    """Get the appropriate command to run Prodigy"""
    # Use uv run if available, otherwise fall back to direct python
    if os.system("uv --version > /dev/null 2>&1") == 0:
        return ["uv", "run", "python", "-m", "prodigy"]
    else:
        return [sys.executable, "-m", "prodigy"]

def start_annotation(dataset_name="goalkeeper_detection", image_dir="data/screenshots", 
                    preprocess=True, crop_height_ratio=0.3, crop_width_ratio=0.5):
    """
    Start Prodigy image classification annotation session

    Args:
        dataset_name: Name for your Prodigy dataset
        image_dir: Directory containing your screenshot images
        preprocess: Whether to preprocess images by cropping (default True)
        crop_height_ratio: What proportion of height to keep from bottom (default 0.3 = bottom 30%)
        crop_width_ratio: What proportion of width to keep from center (default 0.5 = center 50%)
    """

    # Check if image directory exists and has images
    img_path = Path(image_dir)
    if not img_path.exists():
        print(f"Error: Image directory '{image_dir}' doesn't exist")
        print("Please create it and add your screenshot images first")
        return

    # Get image files
    image_files = list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")) + list(img_path.glob("*.jpeg"))

    if not image_files:
        print(f"No image files found in '{image_dir}'")
        print("Supported formats: .jpg, .png, .jpeg")
        return

    print(f"Found {len(image_files)} images to annotate")
    
    # Preprocess images if requested
    if preprocess:
        print("\nPreprocessing images (cropping bottom center)...")
        preprocessed_dir = preprocess_directory(image_dir, "data/screenshots_cropped", 
                                              crop_height_ratio, crop_width_ratio)
        image_dir = str(preprocessed_dir)

    # Start Prodigy annotation
    cmd = get_prodigy_command() + [
        "mark",
        dataset_name,
        image_dir,
        "--loader", "images",
        "--label", "GOALKEEPER",
        "--view-id", "classification"
    ]

    print(f"Starting annotation session...")
    print(f"Command: {' '.join(cmd)}")
    print("\nInstructions:")
    print("- Press Accept (✓) if you can see the goalkeeper indicator/logo")
    print("- Press Reject (✗) if no goalkeeper indicator is visible")
    print("- Press Ctrl+C to stop annotation and save progress")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nAnnotation session stopped. Progress saved!")
    except Exception as e:
        print(f"Error starting Prodigy: {e}")
        print("Make sure Prodigy is properly installed")

def export_data(dataset_name="goalkeeper_detection", output_file="goalkeeper_annotations.jsonl"):
    """Export annotated data"""
    cmd = get_prodigy_command() + ["db-out", dataset_name, output_file]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Data exported to {output_file}")
        else:
            print(f"Error exporting data: {result.stderr}")
            print("\nTroubleshooting:")
            print("1. Make sure you have annotated some data first")
            print("2. Check that the dataset name is correct")
            print(f"3. Try running manually: {sys.executable} -m prodigy db-out {dataset_name} {output_file}")
    except Exception as e:
        print(f"Error exporting data: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "export":
            export_data()
        elif sys.argv[1] == "cleanup":
            cleanup_preprocessed()
            print("Cleaned up preprocessed images")
        elif sys.argv[1] == "no-preprocess":
            start_annotation(preprocess=False)
        else:
            # Allow custom crop ratios
            try:
                height_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
                width_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
                start_annotation(crop_height_ratio=height_ratio, crop_width_ratio=width_ratio)
            except ValueError:
                print("Invalid arguments. Usage:")
                print("  python goalkeeper_annotation.py           # Start with preprocessing (default)")
                print("  python goalkeeper_annotation.py no-preprocess  # Start without preprocessing")
                print("  python goalkeeper_annotation.py export    # Export annotations")
                print("  python goalkeeper_annotation.py cleanup   # Clean preprocessed images")
                print("  python goalkeeper_annotation.py 0.3 0.5   # Custom crop ratios")
    else:
        start_annotation()
