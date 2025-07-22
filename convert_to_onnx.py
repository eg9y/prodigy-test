#!/usr/bin/env python3
"""
Convert trained PyTorch goalkeeper detection model to ONNX format for web inference.
"""

import torch
from ultralytics import YOLO
from pathlib import Path

def convert_model_to_onnx(
    model_path: str = "models/goalkeeper_model.pt", 
    output_path: str = "models/goalkeeper_model.onnx",
    imgsz: int = 224
):
    """Convert YOLO model to ONNX format."""
    
    # Check if model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Export to ONNX
    print(f"Exporting to ONNX format: {output_path}")
    success = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=11,  # ONNX opset version compatible with ONNX.js
        simplify=True,  # Simplify the model
        dynamic=False,  # Fixed input size for better web performance
    )
    
    if success:
        print(f"✅ Model successfully exported to {output_path}")
        print(f"Input shape: [1, 3, {imgsz}, {imgsz}] (batch_size, channels, height, width)")
        print("Model is ready for web inference with ONNX.js")
        
        # Print model info
        print("\nModel classes:")
        print("0: not_goalkeeper")
        print("1: goalkeeper")
        
        return output_path
    else:
        raise RuntimeError("Failed to export model to ONNX")

def main():
    """Main conversion function."""
    try:
        convert_model_to_onnx()
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())