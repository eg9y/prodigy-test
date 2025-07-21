#!/usr/bin/env python3
"""
Training script for goalkeeper detection model using YOLOv8
"""

import json
import subprocess
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def load_prodigy_data(dataset_name="goalkeeper_detection"):
    """Load annotated data from Prodigy database"""
    print(f"Loading data from Prodigy dataset: {dataset_name}")
    
    # Export data from Prodigy
    cmd = ["uv", "run", "python", "-m", "prodigy", "db-out", dataset_name, "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to export data from Prodigy: {result.stderr}")
    
    # Parse JSONL data
    examples = []
    for line in result.stdout.strip().split('\n'):
        if line:
            example = json.loads(line)
            if example.get('answer') == 'accept':
                examples.append(example)
    
    print(f"Loaded {len(examples)} annotated examples")
    return examples

def prepare_yolo_dataset(examples, output_dir="yolo_dataset", eval_split=0.2):
    """Convert Prodigy annotations to YOLO format for classification"""
    
    # Clean up existing dataset
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create directory structure for YOLO classification
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    # Create class directories
    (train_dir / "goalkeeper").mkdir(parents=True, exist_ok=True)
    (train_dir / "not_goalkeeper").mkdir(parents=True, exist_ok=True)
    (val_dir / "goalkeeper").mkdir(parents=True, exist_ok=True)
    (val_dir / "not_goalkeeper").mkdir(parents=True, exist_ok=True)
    
    # Split data
    train_examples, val_examples = train_test_split(
        examples, test_size=eval_split, random_state=42, shuffle=True
    )
    
    print(f"Preparing YOLO dataset...")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    # Copy images to appropriate directories
    def copy_examples(examples, split_dir):
        goalkeeper_count = 0
        not_goalkeeper_count = 0
        
        for idx, example in enumerate(examples):
            if 'path' in example:
                src_path = Path(example['path'])
                if src_path.exists():
                    # Determine class based on annotation
                    if 'goalkeeper' in example.get('accept', []):
                        class_name = "goalkeeper"
                        goalkeeper_count += 1
                    else:
                        class_name = "not_goalkeeper"
                        not_goalkeeper_count += 1
                    
                    # Copy image to appropriate class directory
                    dst_path = split_dir / class_name / f"{class_name}_{idx}_{src_path.name}"
                    shutil.copy2(src_path, dst_path)
        
        return goalkeeper_count, not_goalkeeper_count
    
    # Process training data
    train_gk, train_not_gk = copy_examples(train_examples, train_dir)
    val_gk, val_not_gk = copy_examples(val_examples, val_dir)
    
    print(f"\nDataset prepared:")
    print(f"Training: {train_gk} goalkeeper, {train_not_gk} not_goalkeeper")
    print(f"Validation: {val_gk} goalkeeper, {val_not_gk} not_goalkeeper")
    
    # Create dataset.yaml for YOLO
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'train',
        'val': 'val',
        'nc': 2,  # Number of classes
        'names': ['goalkeeper', 'not_goalkeeper']
    }
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    return output_path

def train_yolo_model(
    dataset_name="goalkeeper_detection",
    output_dir="models",
    model_name="goalkeeper_yolo",
    model_size="n",  # nano model for fast training
    epochs=50,
    imgsz=640,
    batch_size=16,
    eval_split=0.2
):
    """Train YOLOv8 classification model"""
    
    # Load and prepare data
    examples = load_prodigy_data(dataset_name)
    
    if len(examples) < 10:
        print("Error: Not enough annotated examples. Please annotate more images.")
        return
    
    # Prepare YOLO dataset
    dataset_path = prepare_yolo_dataset(examples, eval_split=eval_split)
    
    # Initialize YOLO model for classification
    model = YOLO(f'yolov8{model_size}-cls.pt')  # Use classification model
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nStarting YOLOv8 training...")
    print(f"Model: yolov8{model_size}-cls")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    
    # Train the model
    results = model.train(
        data=str(dataset_path / 'dataset.yaml'),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=model_name,
        project=str(output_path),
        exist_ok=True,
        patience=10,
        save=True,
        device='cpu',  # Use CPU for compatibility
        pretrained=True,
        verbose=True
    )
    
    # Save the best model
    best_model_path = output_path / model_name / 'weights' / 'best.pt'
    final_model_path = output_path / f"{model_name}.pt"
    
    if best_model_path.exists():
        shutil.copy2(best_model_path, final_model_path)
        print(f"\nModel saved to: {final_model_path}")
        
        # Save training info
        info_file = output_path / f"{model_name}_info.json"
        training_info = {
            "dataset": dataset_name,
            "model_path": str(final_model_path),
            "model_size": model_size,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "eval_split": eval_split,
            "class_names": ['goalkeeper', 'not_goalkeeper']
        }
        
        with open(info_file, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Training info saved to: {info_file}")
        
        # Print metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\nTraining Metrics:")
            print(f"Top-1 Accuracy: {metrics.get('metrics/accuracy_top1', 'N/A'):.4f}")
            print(f"Top-5 Accuracy: {metrics.get('metrics/accuracy_top5', 'N/A'):.4f}")

def evaluate_model(model_path, test_images_dir):
    """Evaluate the trained YOLO model"""
    
    model = YOLO(model_path)
    
    print(f"\nEvaluating model on test images...")
    
    # Run inference on test images
    results = model(test_images_dir, stream=True)
    
    for result in results:
        print(f"\nImage: {result.path}")
        probs = result.probs
        if probs is not None:
            class_names = ['goalkeeper', 'not_goalkeeper']
            top1_idx = probs.top1
            top1_conf = probs.top1conf
            print(f"Prediction: {class_names[top1_idx]} (confidence: {top1_conf:.4f})")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train goalkeeper detection model using YOLOv8")
    parser.add_argument("--dataset", default="goalkeeper_detection", help="Prodigy dataset name")
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    parser.add_argument("--model-name", default="goalkeeper_yolo", help="Model name")
    parser.add_argument("--model-size", default="n", choices=['n', 's', 'm', 'l', 'x'], 
                        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--eval-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training")
    parser.add_argument("--test-dir", help="Directory with test images for evaluation")
    
    args = parser.parse_args()
    
    try:
        train_yolo_model(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            model_name=args.model_name,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            eval_split=args.eval_split
        )
        
        if args.evaluate and args.test_dir:
            model_path = Path(args.output_dir) / f"{args.model_name}.pt"
            if model_path.exists():
                evaluate_model(str(model_path), args.test_dir)
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have annotated data (run annotation first)")
        print("2. Check that ultralytics is installed: uv sync")
        print("3. Ensure image paths are accessible")

if __name__ == "__main__":
    main()