#!/usr/bin/env python3
"""
Evaluate the trained goalkeeper detection model on new images.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from image_preprocessor import crop_bottom_center


def load_trained_model(model_path: str = "models/goalkeeper_model.pt"):
    """Load the trained goalkeeper detection model."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python train_model.py")
        return None
    
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    return model


def predict_single_image(model, image_path: str, confidence_threshold: float = 0.5, 
                         crop_height_ratio: float = 0.3, crop_width_ratio: float = 0.5):
    """Predict goalkeeper status for a single image."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Apply the same preprocessing as during training
    try:
        # Crop the image to focus on the goalkeeper UI area
        cropped_image = crop_bottom_center(
            image_path, 
            output_path=None,  # Don't save, just return PIL Image
            crop_height_ratio=crop_height_ratio,
            crop_width_ratio=crop_width_ratio
        )
        
        # Run prediction on the cropped image
        results = model(cropped_image, verbose=False)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None
    
    # Extract results
    result = results[0]
    
    # Get the predicted class and confidence
    probs = result.probs
    if probs is not None:
        top_class_idx = probs.top1
        confidence = probs.top1conf.item()
        class_names = model.names
        predicted_class = class_names[top_class_idx]
        
        # Determine if goalkeeper based on confidence threshold
        is_goalkeeper = (predicted_class == "goalkeeper" and confidence >= confidence_threshold)
        
        return {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_goalkeeper": is_goalkeeper,
            "all_probabilities": {class_names[i]: probs.data[i].item() for i in range(len(probs.data))}
        }
    
    return None


def evaluate_directory(model, directory_path: str, confidence_threshold: float = 0.5,
                      crop_height_ratio: float = 0.3, crop_width_ratio: float = 0.5):
    """Evaluate all images in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(directory_path).glob(f"*{ext}"))
        image_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {directory_path}")
        return
    
    print(f"Found {len(image_files)} images to evaluate")
    print(f"Applying preprocessing: crop bottom {crop_height_ratio:.0%}, center {crop_width_ratio:.0%}")
    print("-" * 70)
    
    goalkeeper_count = 0
    total_count = len(image_files)
    
    for image_file in sorted(image_files):
        result = predict_single_image(model, str(image_file), confidence_threshold, 
                                    crop_height_ratio, crop_width_ratio)
        
        if result:
            if result["is_goalkeeper"]:
                goalkeeper_count += 1
                status = "🥅 GOALKEEPER"
            else:
                status = "⚽ NOT GOALKEEPER"
            
            print(f"{image_file.name:40} | {status:15} | Confidence: {result['confidence']:.3f}")
    
    print("-" * 70)
    print(f"Summary: {goalkeeper_count}/{total_count} images classified as goalkeeper")
    print(f"Goalkeeper rate: {goalkeeper_count/total_count:.1%}")


def evaluate_test_dataset(model, test_dir: str = "evaluate_pics", confidence_threshold: float = 0.5,
                         crop_height_ratio: float = 0.3, crop_width_ratio: float = 0.5):
    """Evaluate model on test dataset with known ground truth labels."""
    test_path = Path(test_dir)
    
    if not test_path.exists():
        print(f"Error: Test directory {test_dir} not found")
        return
    
    goalkeeper_dir = test_path / "goalkeeper"
    not_goalkeeper_dir = test_path / "not_goalkeeper"
    
    if not goalkeeper_dir.exists() or not not_goalkeeper_dir.exists():
        print(f"Error: Expected 'goalkeeper' and 'not_goalkeeper' subdirectories in {test_dir}")
        return
    
    # Get all test images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    goalkeeper_files = []
    not_goalkeeper_files = []
    
    for ext in image_extensions:
        goalkeeper_files.extend(goalkeeper_dir.glob(f"*{ext}"))
        goalkeeper_files.extend(goalkeeper_dir.glob(f"*{ext.upper()}"))
        not_goalkeeper_files.extend(not_goalkeeper_dir.glob(f"*{ext}"))
        not_goalkeeper_files.extend(not_goalkeeper_dir.glob(f"*{ext.upper()}"))
    
    total_images = len(goalkeeper_files) + len(not_goalkeeper_files)
    
    if total_images == 0:
        print(f"No images found in {test_dir}")
        return
    
    print(f"Evaluating on test dataset: {test_dir}")
    print(f"Ground truth: {len(goalkeeper_files)} goalkeeper, {len(not_goalkeeper_files)} not_goalkeeper")
    print(f"Applying preprocessing: crop bottom {crop_height_ratio:.0%}, center {crop_width_ratio:.0%}")
    print("=" * 80)
    
    # Track results for metrics calculation
    true_positives = 0   # Correctly predicted goalkeeper
    true_negatives = 0   # Correctly predicted not_goalkeeper  
    false_positives = 0  # Incorrectly predicted goalkeeper
    false_negatives = 0  # Incorrectly predicted not_goalkeeper
    
    detailed_results = []
    
    # Evaluate goalkeeper images (ground truth = goalkeeper)
    print("GOALKEEPER IMAGES (Should predict: 🥅 GOALKEEPER)")
    print("-" * 80)
    
    for image_file in sorted(goalkeeper_files):
        result = predict_single_image(model, str(image_file), confidence_threshold, 
                                    crop_height_ratio, crop_width_ratio)
        
        if result:
            is_correct = result["is_goalkeeper"]
            if is_correct:
                true_positives += 1
                status = "✅ CORRECT"
            else:
                false_negatives += 1
                status = "❌ WRONG"
            
            prediction_text = "🥅 GOALKEEPER" if result["is_goalkeeper"] else "⚽ NOT GOALKEEPER"
            print(f"{image_file.name:40} | {prediction_text:15} | {status:10} | Conf: {result['confidence']:.3f}")
            
            detailed_results.append({
                "file": image_file.name,
                "ground_truth": "goalkeeper",
                "predicted": result["predicted_class"],
                "confidence": result["confidence"],
                "correct": is_correct
            })
    
    print()
    print("NOT_GOALKEEPER IMAGES (Should predict: ⚽ NOT GOALKEEPER)")
    print("-" * 80)
    
    # Evaluate not_goalkeeper images (ground truth = not_goalkeeper)
    for image_file in sorted(not_goalkeeper_files):
        result = predict_single_image(model, str(image_file), confidence_threshold, 
                                    crop_height_ratio, crop_width_ratio)
        
        if result:
            is_correct = not result["is_goalkeeper"]  # Correct if NOT predicted as goalkeeper
            if is_correct:
                true_negatives += 1
                status = "✅ CORRECT"
            else:
                false_positives += 1
                status = "❌ WRONG"
            
            prediction_text = "🥅 GOALKEEPER" if result["is_goalkeeper"] else "⚽ NOT GOALKEEPER"
            print(f"{image_file.name:40} | {prediction_text:15} | {status:10} | Conf: {result['confidence']:.3f}")
            
            detailed_results.append({
                "file": image_file.name,
                "ground_truth": "not_goalkeeper", 
                "predicted": result["predicted_class"],
                "confidence": result["confidence"],
                "correct": is_correct
            })
    
    # Calculate metrics
    total_correct = true_positives + true_negatives
    total_predictions = true_positives + true_negatives + false_positives + false_negatives
    
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print()
    print("=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Images:     {total_predictions}")
    print(f"Correct:          {total_correct}")
    print(f"Incorrect:        {total_predictions - total_correct}")
    print()
    print("🎯 CONFUSION MATRIX:")
    print(f"True Positives:   {true_positives:2d} (Correctly identified goalkeeper)")
    print(f"True Negatives:   {true_negatives:2d} (Correctly identified not_goalkeeper)")
    print(f"False Positives:  {false_positives:2d} (Incorrectly predicted goalkeeper)")
    print(f"False Negatives:  {false_negatives:2d} (Missed goalkeeper)")
    print()
    print("📈 METRICS:")
    print(f"Accuracy:         {accuracy:.1%} ({total_correct}/{total_predictions})")
    print(f"Precision:        {precision:.1%} (of predicted goalkeepers, how many were correct)")
    print(f"Recall:           {recall:.1%} (of actual goalkeepers, how many were found)")
    print(f"F1-Score:         {f1_score:.1%} (harmonic mean of precision and recall)")
    
    # Performance assessment
    if accuracy >= 0.9:
        performance = "🟢 EXCELLENT"
    elif accuracy >= 0.8:
        performance = "🟡 GOOD"
    elif accuracy >= 0.7:
        performance = "🟠 FAIR"
    else:
        performance = "🔴 NEEDS IMPROVEMENT"
    
    print()
    print(f"Overall Performance: {performance}")
    print("=" * 80)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_images": total_predictions,
        "detailed_results": detailed_results
    }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate goalkeeper detection model")
    parser.add_argument("input", nargs="?", help="Path to image file or directory")
    parser.add_argument("--model", default="models/goalkeeper_model.pt", 
                       help="Path to trained model")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for goalkeeper detection")
    parser.add_argument("--crop-height", type=float, default=0.3,
                       help="Height ratio for bottom crop (default: 0.3 = bottom 30%)")
    parser.add_argument("--crop-width", type=float, default=0.5,
                       help="Width ratio for center crop (default: 0.5 = center 50%)")
    parser.add_argument("--test", action="store_true", 
                       help="Evaluate on test dataset in evaluate_pics folder")
    parser.add_argument("--test-dir", default="evaluate_pics",
                       help="Directory containing test images with ground truth")
    
    args = parser.parse_args()
    
    # Load model
    model = load_trained_model(args.model)
    if model is None:
        return
    
    print(f"Model classes: {model.names}")
    print(f"Using confidence threshold: {args.confidence}")
    print()
    
    # Check if test mode is requested
    if args.test:
        # Evaluate on test dataset with ground truth
        evaluate_test_dataset(model, args.test_dir, args.confidence, 
                             args.crop_height, args.crop_width)
        return
    
    # Require input if not in test mode
    if not args.input:
        print("Error: input path required when not using --test mode")
        print("Usage: python evaluate_model.py <image_path_or_directory>")
        print("   or: python evaluate_model.py --test")
        return
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single image
        result = predict_single_image(model, args.input, args.confidence, 
                                    args.crop_height, args.crop_width)
        if result:
            print(f"Image: {result['image_path']}")
            print(f"Preprocessing: crop bottom {args.crop_height:.0%}, center {args.crop_width:.0%}")
            print(f"Prediction: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Is Goalkeeper: {result['is_goalkeeper']}")
            print("\nAll class probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
    
    elif os.path.isdir(args.input):
        # Directory of images
        evaluate_directory(model, args.input, args.confidence, 
                          args.crop_height, args.crop_width)
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, evaluate the screenshots directory
        model = load_trained_model()
        if model is not None:
            screenshots_dir = "data/screenshots"
            if os.path.exists(screenshots_dir):
                print("No arguments provided. Evaluating all screenshots in data/screenshots/")
                print()
                evaluate_directory(model, screenshots_dir)
            else:
                print("Usage: python evaluate_model.py <image_path_or_directory>")
                print("Example: python evaluate_model.py data/screenshots/")
                print("Example: python evaluate_model.py my_image.jpg")
    else:
        main()
