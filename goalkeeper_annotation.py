#!/usr/bin/env python3
"""
Goalkeeper Detection Annotation Script for Prodigy

This script helps set up image classification annotation for detecting
whether a player is a goalkeeper in Rematch gameplay videos.
"""

import subprocess
import sys
from pathlib import Path

def start_annotation(dataset_name="goalkeeper_detection", image_dir="data/screenshots"):
    """
    Start Prodigy image classification annotation session

    Args:
        dataset_name: Name for your Prodigy dataset
        image_dir: Directory containing your screenshot images
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

    # Start Prodigy annotation
    cmd = [
        "uv", "run", "python", "-m", "prodigy",
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
    cmd = ["uv", "run", "python", "-m", "prodigy", "db-out", dataset_name, output_file]

    try:
        subprocess.run(cmd)
        print(f"Data exported to {output_file}")
    except Exception as e:
        print(f"Error exporting data: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export_data()
    else:
        start_annotation()
