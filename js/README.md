# ⚽ Goalkeeper Detection Video Player

A streamlined web-based video analysis tool for detecting goalkeeper moments in gameplay videos using machine learning.

## Features

- **Video Analysis**: Upload and analyze gameplay videos to detect goalkeeper moments
- **Interactive Timeline**: Visual timeline showing continuous goalkeeper periods
- **Video Player with Markers**: Click markers to jump to specific goalkeeper moments
- **Customizable Settings**: Adjust frame interval, confidence threshold, and analysis parameters
- **Export Results**: Download detailed analysis results as JSON

## Quick Start

1. **Start the server**:
   ```bash
   python serve.py
   ```

2. **Open in browser**:
   ```
   http://localhost:8000
   ```

3. **Analyze videos**:
   - Upload or drag & drop a gameplay video
   - Adjust settings if needed (⚙️ Settings button)
   - Click "🔍 Analyze Video"
   - View results with interactive timeline and video player

## Settings

### Frame Interval
- **Range**: 0.5 - 10 seconds
- **Default**: 1 second
- **Description**: How often to analyze frames. Smaller values = more accurate but slower processing.

### Confidence Threshold
- **Range**: 0.1 - 1.0
- **Default**: 0.5
- **Description**: Minimum confidence required to classify a frame as containing a goalkeeper.

### Max Frames (Optional)
- **Description**: Limit total frames for quick testing. Leave empty for full analysis.

### Skip First Seconds
- **Default**: 0
- **Description**: Skip initial seconds of video (useful for skipping intros/menus).

## Project Structure

```
js/
├── index.html              # Main web interface
├── serve.py               # Local development server
├── plyr.css              # Video player styles  
├── plyr.polyfilled.js    # Video player library
├── models/
│   └── goalkeeper_model.onnx  # Trained ML model
└── Core JavaScript modules:
    ├── worker-interface.js     # Web worker communication
    ├── video-processor.js      # Video frame extraction
    ├── inference-worker.js     # ML inference worker
    └── gameplay-analyzer.js    # Main analysis orchestration
```

## Model Details

- **Input**: 224x224 RGB images (cropped to bottom 30%, center 50% of frame)
- **Output**: Binary classification (goalkeeper vs not_goalkeeper)
- **Framework**: ONNX Runtime for web browsers
- **Classes**: Index 0 = goalkeeper, Index 1 = not_goalkeeper

## Data Export

The analysis results include:
- **Metadata**: Filename, timestamp, settings used
- **Statistics**: Total frames, goalkeeper count, confidence scores
- **Goalkeeper Segments**: Continuous periods with start/end times
- **Detailed Frames**: Frame-by-frame analysis results

## Browser Compatibility

- Modern browsers with WebAssembly support
- Chrome, Firefox, Safari, Edge
- Requires JavaScript enabled

## Development

The codebase uses vanilla JavaScript with Web Workers for ML inference to keep the UI responsive during analysis.

Key components:
- **GameplayAnalyzer**: Orchestrates the analysis pipeline
- **VideoProcessor**: Extracts frames from video files
- **InferenceWorker**: Runs ML model in background thread
- **WorkerInterface**: Manages communication with worker

## Performance Notes

- **Frame Interval**: Lower values (0.5s) provide more detail but take longer
- **Video Length**: Processing time scales linearly with video duration
- **Hardware**: Inference runs on CPU in browser (WebAssembly)
- **Memory**: Large videos may require more RAM for processing