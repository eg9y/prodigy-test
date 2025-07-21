# Goalkeeper Detection with Prodigy

A project to detect goalkeeper status in Rematch gameplay videos using image classification.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Collect screenshot data:
   - Extract screenshots from your gameplay videos
   - Place them in `data/screenshots/` directory
   - Supported formats: .jpg, .png, .jpeg

## Annotation Workflow

### Step 1: Start Annotation
```bash
python goalkeeper_annotation.py
```

This will start Prodigy's web interface for image classification. You'll see each screenshot and can label it as:
- **goalkeeper**: Player is currently a goalkeeper (logo/indicator visible)
- **not_goalkeeper**: Player is not a goalkeeper

### Step 2: Export Data
```bash
python goalkeeper_annotation.py export
```

This exports your annotations to `goalkeeper_annotations.jsonl`

## Data Collection Tips

1. **Screenshot Extraction**: Extract frames at 1-2 second intervals from your gameplay videos
2. **Quality**: Ensure screenshots are clear and the goalkeeper indicator is visible when present
3. **Variety**: Include different game scenarios, lighting conditions, and camera angles
4. **Balance**: Try to get roughly equal numbers of goalkeeper vs non-goalkeeper examples

## Training the Model

### Step 3: Train Model
After annotating and exporting data:
```bash
python train_model.py
```

Options:
- `--batch-size 16`: Adjust batch size
- `--epochs 20`: Number of training epochs
- `--evaluate`: Run evaluation after training
- `--pretrained`: Use transfer learning

Example with custom settings:
```bash
python train_model.py --batch-size 8 --epochs 30 --evaluate
```

## Running Inference

### Single Image Prediction
```bash
python inference.py path/to/image.jpg
```

### Batch Prediction
```bash
python inference.py data/screenshots/ --output predictions.jsonl
```

### Live Classification Interface
```bash
python inference.py --live --port 8080
```

Options:
- `--no-preprocess`: Skip image preprocessing
- `--model models/goalkeeper_model`: Specify model path
- `--analyze`: Show prediction statistics

## Model Evaluation

### Basic Evaluation
```bash
python evaluate_model.py
```

### Cross-Validation
```bash
python evaluate_model.py --cross-validate --folds 5
```

### Error Analysis
```bash
python evaluate_model.py --analyze-errors
```

### Full Performance Report
```bash
python evaluate_model.py --report
```

## Complete Workflow Example

1. **Prepare data**: Place screenshots in `data/screenshots/`
2. **Annotate**: `python goalkeeper_annotation.py`
3. **Export**: `python goalkeeper_annotation.py export`
4. **Train**: `python train_model.py --evaluate`
5. **Predict**: `python inference.py data/new_screenshots/`
6. **Evaluate**: `python evaluate_model.py --report`

## Image Preprocessing

The scripts automatically crop the bottom center of images where the goalkeeper UI typically appears. To skip preprocessing:
- Annotation: `python goalkeeper_annotation.py no-preprocess`
- Inference: `python inference.py image.jpg --no-preprocess`

Custom crop ratios:
```bash
python goalkeeper_annotation.py 0.3 0.5  # height_ratio width_ratio
```