#!/usr/bin/env python3
"""
Goalkeeper detection model training script using YOLO11 classification.
Converts Prodigy annotations to YOLO format and trains a classification model.
"""

import json
import os
import base64
import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from ultralytics import YOLO
from image_preprocessor import crop_bottom_center


def load_annotation_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load annotated data from JSONL file."""
    annotations = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    annotations.append(data)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
                    continue

    print(f"Loaded {len(annotations)} annotations from {jsonl_path}")
    return annotations


def decode_base64_image(base64_string: str) -> Image.Image:
    """Convert base64 encoded image to PIL Image."""
    # Remove data URL prefix if present
    if base64_string.startswith('data:image/'):
        base64_string = base64_string.split(',')[1]

    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)

    # Convert to PIL Image
    image = Image.open(BytesIO(image_bytes))

    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def create_yolo_dataset(annotations: List[Dict[str, Any]], dataset_path: str = "dataset"):
    """Create YOLO-format dataset from Prodigy annotations."""
    # Create dataset directory structure
    train_dir = Path(dataset_path) / "train"
    val_dir = Path(dataset_path) / "val"

    # Create class directories
    for split_dir in [train_dir, val_dir]:
        (split_dir / "goalkeeper").mkdir(parents=True, exist_ok=True)
        (split_dir / "not_goalkeeper").mkdir(parents=True, exist_ok=True)

    # Process annotations
    goalkeeper_count = 0
    not_goalkeeper_count = 0

    for i, annotation in enumerate(annotations):
        try:
            # Extract image data and answer (Prodigy format)
            image_data = annotation.get('image', '')
            answer = annotation.get('answer', '')

            if not image_data:
                print(f"Skipping annotation {i}: missing image data")
                continue

            if answer not in ['accept', 'reject']:
                print(f"Skipping annotation {i}: missing or invalid answer field (got: {answer})")
                continue

            # Convert Prodigy answer to our label format
            label = 'goalkeeper' if answer == 'accept' else 'not_goalkeeper'

            # Decode image
            image = decode_base64_image(image_data)

            # Count labels for balanced split
            if label == 'goalkeeper':
                goalkeeper_count += 1
            else:
                not_goalkeeper_count += 1

            # Split into train/val (80/20 split, balanced by class)
            if label == 'goalkeeper':
                use_val = (goalkeeper_count % 5 == 0)  # Every 5th goalkeeper image goes to val
            else:
                use_val = (not_goalkeeper_count % 5 == 0)  # Every 5th non-goalkeeper image goes to val

            split_dir = val_dir if use_val else train_dir

            # Save image
            filename = f"{label}_{i:04d}.jpg"
            image_path = split_dir / label / filename
            image.save(image_path, "JPEG", quality=95)

        except Exception as e:
            print(f"Error processing annotation {i}: {e}")
            continue

    # Print dataset statistics
    train_gk = len(list((train_dir / "goalkeeper").glob("*.jpg")))
    train_ngk = len(list((train_dir / "not_goalkeeper").glob("*.jpg")))
    val_gk = len(list((val_dir / "goalkeeper").glob("*.jpg")))
    val_ngk = len(list((val_dir / "not_goalkeeper").glob("*.jpg")))

    print(f"\nDataset created at {dataset_path}")
    print(f"Training set: {train_gk} goalkeeper, {train_ngk} not_goalkeeper")
    print(f"Validation set: {val_gk} goalkeeper, {val_ngk} not_goalkeeper")
    print(f"Total: {train_gk + val_gk} goalkeeper, {train_ngk + val_ngk} not_goalkeeper")

    return dataset_path


def create_dataset_yaml(dataset_path: str):
    """Create dataset.yaml configuration file for YOLO."""
    yaml_content = f"""# Goalkeeper Detection Dataset
path: {os.path.abspath(dataset_path)}
train: train
val: val

# Class names
names:
  0: not_goalkeeper
  1: goalkeeper

# Number of classes
nc: 2
"""

    yaml_path = Path(dataset_path) / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Dataset configuration saved to {yaml_path}")
    return str(yaml_path)


def train_yolo_model(dataset_path: str, model_size: str = "n", epochs: int = 20, imgsz: int = 224):
    """Train YOLO11 classification model."""
    # Load pretrained model
    model_name = f"yolo11{model_size}-cls.pt"
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # Train the model
    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=dataset_path,  # Pass the dataset directory, not the yaml file
        epochs=epochs,
        imgsz=imgsz,
        batch=16,  # Adjust based on your GPU memory
        patience=20,  # Early stopping patience
        save=True,
        device='cpu',  # Change to 'cuda' if you have GPU
        workers=4,
        project="goalkeeper_training",
        name="goalkeeper_model"
    )

    return model, results


def evaluate_model(model, dataset_path: str):
    """Evaluate the trained model."""
    print("Evaluating model...")
    metrics = model.val(data=dataset_path)

    print(f"Top-1 Accuracy: {metrics.top1:.3f}")
    print(f"Top-5 Accuracy: {metrics.top5:.3f}")

    return metrics


def save_model_for_inference(model, output_path: str = "models/goalkeeper_model.pt"):
    """Save the trained model for inference."""
    os.makedirs("models", exist_ok=True)

    # Copy the best model weights
    best_model_path = model.trainer.best
    shutil.copy2(best_model_path, output_path)

    print(f"Model saved to {output_path}")
    print(f"To use for inference: model = YOLO('{output_path}')")


def main():
    """Main training function."""
    # Configuration
    MODEL_SIZE = "n"  # Options: n, s, m, l, x (larger = more accurate but slower)
    EPOCHS = 20       # Reduce if overfitting, increase for better accuracy
    IMAGE_SIZE = 224  # Standard size for classification

    # Find annotation file
    annotation_files = [
        "goalkeeper_detection.jsonl",
        "-/goalkeeper_detection.jsonl",
        "goalkeeper_annotations.jsonl/goalkeeper_detection.jsonl"
    ]

    annotation_path = None
    for file_path in annotation_files:
        if os.path.exists(file_path):
            annotation_path = file_path
            break

    if annotation_path is None:
        print("Error: Could not find annotation file!")
        print("Expected files:", annotation_files)
        return

    print(f"Using annotation file: {annotation_path}")

    try:
        # Load annotations
        annotations = load_annotation_data(annotation_path)

        if len(annotations) < 10:
            print("Warning: Very few annotations found. Consider annotating more data for better results.")

        # Create YOLO dataset
        dataset_path = create_yolo_dataset(annotations)
        dataset_yaml = create_dataset_yaml(dataset_path)

        # Train model
        model, results = train_yolo_model(
            dataset_path,  # Pass dataset path instead of yaml
            model_size=MODEL_SIZE,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE
        )

        # Evaluate model
        metrics = evaluate_model(model, dataset_path)

        # Save model for inference
        save_model_for_inference(model)

        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model accuracy: {metrics.top1:.1%}")
        print("Model saved to: models/goalkeeper_model.pt")
        print("\nTo use the trained model:")
        print("from ultralytics import YOLO")
        print("model = YOLO('models/goalkeeper_model.pt')")
        print("results = model('path/to/your/image.jpg')")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
