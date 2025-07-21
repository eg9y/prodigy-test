#!/usr/bin/env python3
"""
Quick prediction script for goalkeeper detection.
Simple interface for testing the trained model on new images.
"""

import sys
import os
from ultralytics import YOLO
from image_preprocessor import crop_bottom_center


def predict_goalkeeper(image_path: str, model_path: str = "models/goalkeeper_model.pt", 
                      confidence_threshold: float = 0.5) -> dict:
    """
    Quick goalkeeper prediction with preprocessing.
    
    Args:
        image_path: Path to the image to predict
        model_path: Path to the trained model
        confidence_threshold: Threshold for goalkeeper classification
    
    Returns:
        Dictionary with prediction results
    """
    # Check if model exists
    if not os.path.exists(model_path):
        return {"error": f"Model not found at {model_path}. Please train the model first."}
    
    # Check if image exists
    if not os.path.exists(image_path):
        return {"error": f"Image not found at {image_path}"}
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Apply preprocessing (same as training)
        cropped_image = crop_bottom_center(
            image_path, 
            output_path=None,  # Don't save, just return PIL Image
            crop_height_ratio=0.3,  # Bottom 30%
            crop_width_ratio=0.5    # Center 50%
        )
        
        # Make prediction
        results = model(cropped_image, verbose=False)
        result = results[0]
        
        # Extract prediction info
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
                "goalkeeper_probability": probs.data[0].item() if predicted_class == "goalkeeper" else probs.data[1].item(),
                "not_goalkeeper_probability": probs.data[1].item() if predicted_class == "goalkeeper" else probs.data[0].item(),
                "preprocessing_applied": "bottom 30%, center 50%"
            }
        else:
            return {"error": "Could not extract prediction probabilities"}
            
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python quick_predict.py <image_path> [confidence_threshold]")
        print("Example: python quick_predict.py data/screenshots/image.jpg")
        print("Example: python quick_predict.py data/screenshots/image.jpg 0.7")
        return
    
    image_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    print("🥅 Goalkeeper Detection")
    print("="*40)
    
    result = predict_goalkeeper(image_path, confidence_threshold=confidence_threshold)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Display results
    print(f"📷 Image: {result['image_path']}")
    print(f"🔄 Preprocessing: {result['preprocessing_applied']}")
    print(f"🎯 Prediction: {result['predicted_class']}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    print(f"🥅 Is Goalkeeper: {'YES' if result['is_goalkeeper'] else 'NO'}")
    print("\n📈 Detailed Probabilities:")
    print(f"   Goalkeeper: {result['goalkeeper_probability']:.1%}")
    print(f"   Not Goalkeeper: {result['not_goalkeeper_probability']:.1%}")
    
    # Confidence indicator
    if result['confidence'] > 0.9:
        confidence_icon = "🟢"
        confidence_text = "Very Confident"
    elif result['confidence'] > 0.7:
        confidence_icon = "🟡"
        confidence_text = "Confident"
    else:
        confidence_icon = "🟠"
        confidence_text = "Less Confident"
    
    print(f"\n{confidence_icon} Confidence Level: {confidence_text}")


if __name__ == "__main__":
    main()