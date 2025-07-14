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

## Next Steps

After annotation, you can:
1. Train a classification model using the exported data
2. Integrate the model into your app for real-time detection
3. Use the model to automatically detect goalkeeper periods in new videos