# Goalkeeper Detection ML Pipeline

A complete machine learning pipeline for detecting goalkeeper status in Rematch gameplay videos using YOLO11 image classification. This project provides an end-to-end workflow from data collection to model deployment.

## 🚀 Quick Start

```bash
# 1. Setup
uv sync

# 2. Collect & annotate data
python goalkeeper_annotation.py

# 3. Export annotations
python goalkeeper_annotation.py export

# 4. Train model
python train_model.py

# 5. Evaluate on test set
python evaluate_model.py --test
```

## 📋 Complete Workflow

### Step 1: Data Collection

**Goal**: Collect screenshots from Rematch gameplay videos for annotation.

1. **Extract screenshots** from your gameplay videos at 1-2 second intervals
2. **Place them** in the `data/screenshots/` directory
3. **Supported formats**: `.jpg`, `.png`, `.jpeg`

**Tips for quality data:**
- Include variety: different game scenarios, lighting, camera angles
- Balance the dataset: aim for roughly equal goalkeeper/non-goalkeeper examples
- Ensure clear visibility of the goalkeeper UI indicator (glove icon) when present
- Extract 100-500+ screenshots for a robust dataset

```bash
# Example directory structure
data/
└── screenshots/
    ├── gameplay_001.png
    ├── gameplay_002.png
    └── ...
```

### Step 2: Data Annotation

**Goal**: Label screenshots to create training data using Prodigy's web interface.

```bash
# Start annotation session
python goalkeeper_annotation.py
```

**What you'll do:**
- Open your browser to the Prodigy interface (usually `http://localhost:8080`)
- For each screenshot, choose:
  - **Accept** (✅): Goalkeeper UI indicator is visible
  - **Reject** (❌): No goalkeeper indicator present
- The images are automatically preprocessed (cropped to bottom 30%, center 50%) to focus on the UI area

**Best practices:**
- Take your time to ensure accurate labels
- Look specifically for the goalkeeper glove icon in the UI
- Consistent labeling is crucial for model performance

### Step 3: Export Annotations

**Goal**: Convert Prodigy annotations to a format suitable for training.

```bash
# Export annotated data to JSONL format
python goalkeeper_annotation.py export
```

This creates a JSONL file (e.g., `goalkeeper_annotations.jsonl`) containing your labeled data with the following structure:
```json
{"image": "data:image/png;base64,...", "answer": "accept", "label": "GOALKEEPER", ...}
```

### Step 4: Model Training

**Goal**: Train a YOLO11 classification model on your annotated data.

```bash
# Train the model
python train_model.py
```

**What happens during training:**
1. **Data conversion**: Prodigy JSONL → YOLO dataset format
2. **Preprocessing**: Images are cropped consistently (bottom 30%, center 50%)
3. **Dataset split**: 80% training, 20% validation, balanced by class
4. **Training**: YOLO11n-cls model trains for 20 epochs with early stopping
5. **Model saving**: Trained model saved to `models/goalkeeper_model.pt`

**Training output example:**
```
Loaded 72 annotations from goalkeeper_annotations.jsonl
Training images: 58, Validation images: 14
Classes: {'not_goalkeeper': 36, 'goalkeeper': 36}
...
Training complete! Model saved to models/goalkeeper_model.pt
Final accuracy: 92.9%
```

### Step 5: Model Evaluation

**Goal**: Test your trained model on various datasets to assess performance.

#### 5a. Test on Original Screenshots
```bash
# Evaluate on all training screenshots
python evaluate_model.py data/screenshots/
```

#### 5b. Test on Ground Truth Dataset
```bash
# Evaluate on test set with known labels
python evaluate_model.py --test
```

**Test dataset structure:**
```
evaluate_pics/
├── goalkeeper/          # Images that should predict "goalkeeper"
│   ├── test_gk_001.png
│   └── test_gk_002.png
└── not_goalkeeper/      # Images that should predict "not_goalkeeper"
    ├── test_nogk_001.png
    └── test_nogk_002.png
```

**Evaluation output:**
```
📊 EVALUATION RESULTS
================================================================================
Total Images:     24
Correct:          22
Incorrect:        2

🎯 CONFUSION MATRIX:
True Positives:   11 (Correctly identified goalkeeper)
True Negatives:   11 (Correctly identified not_goalkeeper)
False Positives:   1 (Incorrectly predicted goalkeeper)
False Negatives:   1 (Missed goalkeeper)

📈 METRICS:
Accuracy:         91.7% (22/24)
Precision:        91.7% (of predicted goalkeepers, how many were correct)
Recall:           91.7% (of actual goalkeepers, how many were found)
F1-Score:         91.7% (harmonic mean of precision and recall)

Overall Performance: 🟢 EXCELLENT
```

#### 5c. Test Single Images
```bash
# Test a specific image
python evaluate_model.py "path/to/test_image.jpg"

# Test with custom confidence threshold
python evaluate_model.py "test_image.jpg" --confidence 0.7
```

## 🛠️ Setup & Dependencies

### Requirements
- **Python 3.12+** (strictly required)
- **UV Package Manager** for dependency management
- **Valid Prodigy License** with authentication token

### Installation
```bash
# Install all dependencies
uv sync

# Verify setup
python --version  # Should show 3.12+
python -c "import prodigy; print('Prodigy OK')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

## 🔧 Advanced Usage

### Custom Training Parameters
```bash
# Train with custom settings
python train_model.py --epochs 50 --batch-size 16
```

### Custom Evaluation
```bash
# Test with different confidence threshold
python evaluate_model.py --test --confidence 0.8

# Test with different preprocessing
python evaluate_model.py --test --crop-height 0.4 --crop-width 0.6

# Test custom directory
python evaluate_model.py --test --test-dir "my_test_images/"
```

### Visualize Preprocessing
```bash
# See how images are cropped before training/prediction
python show_preprocessing.py "data/screenshots/example.png"
```

### Quick Predictions
```bash
# Single image prediction with details
python quick_predict.py "new_image.jpg"
```

## 📊 Understanding the Pipeline

### Image Preprocessing
All images are preprocessed consistently across training and inference:
- **Crop bottom 30%**: Focus on the lower portion where goalkeeper UI appears
- **Crop center 50%**: Center horizontally to focus on main UI elements
- This preprocessing is **critical** for accurate predictions

### Model Architecture
- **Base model**: YOLO11n-cls (lightweight classification model)
- **Classes**: 2 (goalkeeper, not_goalkeeper)
- **Input**: 224x224 RGB images (after preprocessing)
- **Training**: 20 epochs with early stopping, patience=5

### Data Format
- **Prodigy export**: JSONL with base64-encoded images
- **YOLO format**: Directory structure with class folders
- **Model input**: Preprocessed PIL Images

## 🐛 Troubleshooting

### Common Issues

**1. "Only 23 annotations loaded" (but Prodigy shows more)**
```bash
# Re-export your annotations
python goalkeeper_annotation.py export
```

**2. Low prediction confidence**
- Ensure you're applying preprocessing: `crop_bottom_center()`
- Check if test images are similar to training data
- Consider retraining with more diverse data

**3. "Model not found" error**
```bash
# Train the model first
python train_model.py
```

**4. Prodigy license issues**
- Verify your license token in `pyproject.toml`
- Check with `python -c "import prodigy; print('OK')"`

### Performance Guidelines
- **90%+ accuracy**: 🟢 Excellent performance
- **80-90% accuracy**: 🟡 Good performance  
- **70-80% accuracy**: 🟠 Fair, consider more data
- **<70% accuracy**: 🔴 Needs improvement, check data quality

## 📁 Project Structure

```
prodigy-test/
├── data/
│   └── screenshots/          # Input screenshots for annotation
├── evaluate_pics/            # Test dataset with ground truth
│   ├── goalkeeper/           # Positive examples
│   └── not_goalkeeper/       # Negative examples
├── dataset/                  # Generated YOLO dataset
│   ├── train/               # Training images by class
│   ├── val/                 # Validation images by class
│   └── dataset.yaml         # YOLO dataset configuration
├── models/
│   └── goalkeeper_model.pt   # Trained model
├── goalkeeper_annotation.py  # Prodigy annotation workflow
├── train_model.py           # Model training script
├── evaluate_model.py        # Model evaluation tools
├── quick_predict.py         # Simple prediction interface
├── image_preprocessor.py    # Image preprocessing utilities
└── show_preprocessing.py    # Preprocessing visualization
```

## 🚀 Production Usage

```python
from ultralytics import YOLO
from image_preprocessor import crop_bottom_center

# Load trained model
model = YOLO('models/goalkeeper_model.pt')

# Predict on new image
def predict_goalkeeper(image_path, confidence_threshold=0.5):
    # Apply same preprocessing as training
    cropped_image = crop_bottom_center(
        image_path, 
        crop_height_ratio=0.3, 
        crop_width_ratio=0.5
    )
    
    # Get prediction
    results = model(cropped_image, verbose=False)
    result = results[0]
    
    predicted_class = model.names[result.probs.top1]
    confidence = result.probs.top1conf.item()
    
    is_goalkeeper = (predicted_class == "goalkeeper" and 
                    confidence >= confidence_threshold)
    
    return {
        "is_goalkeeper": is_goalkeeper,
        "confidence": confidence,
        "predicted_class": predicted_class
    }

# Usage
result = predict_goalkeeper("gameplay_screenshot.png")
print(f"Goalkeeper detected: {result['is_goalkeeper']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 🌐 Web-Based Video Analysis

In addition to the ML training pipeline, this project includes a **streamlined web interface** for analyzing gameplay videos in real-time.

### Features
- **Interactive Video Player**: Upload and analyze videos with Plyr video player
- **Real-time Detection**: See goalkeeper indicators during video playback  
- **Timeline Visualization**: Visual timeline showing continuous goalkeeper periods
- **Customizable Settings**: Adjust frame interval, confidence threshold, analysis parameters
- **Export Results**: Download detailed frame-by-frame analysis as JSON

### Quick Start (Web Interface)
```bash
# Navigate to web interface directory
cd js/

# Start local server
python serve.py

# Open browser to http://localhost:8000
# Upload a video and click "Analyze Video"
```

### Web Interface Structure
```
js/
├── index.html              # Main web interface
├── serve.py               # Local development server  
├── models/
│   └── goalkeeper_model.onnx  # Web-compatible model
└── Core JavaScript modules:
    ├── worker-interface.js     # Web worker communication
    ├── video-processor.js      # Video frame extraction
    ├── inference-worker.js     # ML inference worker
    └── gameplay-analyzer.js    # Analysis orchestration
```

### Model Conversion
The trained PyTorch model is converted to ONNX format for web deployment:
```bash
# Convert trained model to ONNX (web-compatible)
python convert_to_onnx.py
```

---

## 📚 Additional Resources

- [Prodigy Documentation](https://prodi.gy/)
- [YOLO11 Documentation](https://docs.ultralytics.com/)
- [UV Package Manager](https://github.com/astral-sh/uv)
- [ONNX Runtime](https://onnxruntime.ai/) - For web deployment

For questions or contributions, please refer to the project documentation or create an issue.