#!/usr/bin/env python3
"""
Inference script for goalkeeper detection using trained YOLO model
"""

import json
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import argparse

def load_model(model_path="models/goalkeeper_yolo.pt"):
    """Load the trained YOLO model"""
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    return model

def predict_single_image(model, image_path, confidence_threshold=0.5):
    """Run inference on a single image"""
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Run inference
    results = model(str(image_path))
    
    # Get prediction
    result = results[0]
    probs = result.probs
    
    if probs is not None:
        class_names = ['goalkeeper', 'not_goalkeeper']
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        
        prediction = {
            'image': str(image_path),
            'class': class_names[top1_idx],
            'confidence': top1_conf,
            'is_goalkeeper': class_names[top1_idx] == 'goalkeeper' and top1_conf >= confidence_threshold
        }
        
        return prediction
    
    return None

def predict_directory(model, directory_path, confidence_threshold=0.5):
    """Run inference on all images in a directory"""
    
    directory_path = Path(directory_path)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found at {directory_path}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory_path.glob(f'*{ext}'))
        image_files.extend(directory_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in {directory_path}")
    
    predictions = []
    goalkeeper_count = 0
    
    for image_path in image_files:
        prediction = predict_single_image(model, image_path, confidence_threshold)
        if prediction:
            predictions.append(prediction)
            if prediction['is_goalkeeper']:
                goalkeeper_count += 1
            
            print(f"{image_path.name}: {prediction['class']} ({prediction['confidence']:.2f})")
    
    print(f"\nSummary:")
    print(f"Total images: {len(predictions)}")
    print(f"Goalkeeper images: {goalkeeper_count}")
    print(f"Non-goalkeeper images: {len(predictions) - goalkeeper_count}")
    
    return predictions

def batch_inference(model, image_list, confidence_threshold=0.5):
    """Run batch inference on multiple images"""
    
    # Run inference on all images at once
    results = model(image_list)
    
    predictions = []
    class_names = ['goalkeeper', 'not_goalkeeper']
    
    for i, result in enumerate(results):
        probs = result.probs
        if probs is not None:
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            
            prediction = {
                'image': image_list[i],
                'class': class_names[top1_idx],
                'confidence': top1_conf,
                'is_goalkeeper': class_names[top1_idx] == 'goalkeeper' and top1_conf >= confidence_threshold
            }
            predictions.append(prediction)
    
    return predictions

def save_predictions(predictions, output_path="predictions.json"):
    """Save predictions to a JSON file"""
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\nPredictions saved to: {output_path}")

def main():
    """Main inference function"""
    
    parser = argparse.ArgumentParser(description="Run goalkeeper detection inference")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--model", default="models/goalkeeper_yolo.pt", help="Path to trained model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--save", help="Save predictions to JSON file")
    parser.add_argument("--batch", action="store_true", help="Use batch inference for directories")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = load_model(args.model)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image inference
            prediction = predict_single_image(model, input_path, args.confidence)
            if prediction:
                print(f"\nPrediction for {input_path.name}:")
                print(f"Class: {prediction['class']}")
                print(f"Confidence: {prediction['confidence']:.4f}")
                print(f"Is Goalkeeper: {prediction['is_goalkeeper']}")
                
                if args.save:
                    save_predictions([prediction], args.save)
        
        elif input_path.is_dir():
            # Directory inference
            if args.batch:
                # Batch inference (faster for many images)
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_files.extend([str(p) for p in input_path.glob(f'*{ext}')])
                    image_files.extend([str(p) for p in input_path.glob(f'*{ext.upper()}')])
                
                if image_files:
                    predictions = batch_inference(model, image_files, args.confidence)
                    
                    if args.save:
                        save_predictions(predictions, args.save)
            else:
                # Sequential inference
                predictions = predict_directory(model, input_path, args.confidence)
                
                if args.save:
                    save_predictions(predictions, args.save)
        
        else:
            print(f"Error: {input_path} is neither a file nor a directory")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()