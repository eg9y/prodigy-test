# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Prodigy-based data annotation project for **goalkeeper detection in Rematch gameplay videos**. The goal is to build a labeled dataset by annotating screenshots to train an image classification model that can automatically detect when a player is acting as goalkeeper (indicated by the goalkeeper UI element with glove icon).

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies using UV package manager
uv sync

# Verify Python version (requires 3.12+)
python --version
```

### Annotation Workflow
```bash
# Start Prodigy annotation session
python goalkeeper_annotation.py

# Export annotated data to JSONL format
python goalkeeper_annotation.py export
```

### Data Preparation
- Place gameplay screenshots in `data/screenshots/` directory
- Supported formats: `.jpg`, `.png`, `.jpeg`
- Extract frames at 1-2 second intervals from gameplay videos
- Aim for balanced dataset (equal goalkeeper/non-goalkeeper examples)

## Architecture

**Core Components:**
- `goalkeeper_annotation.py`: Main annotation workflow controller that manages Prodigy sessions
- `data/screenshots/`: Input directory for gameplay screenshots to be annotated
- Exported JSONL files: Training data for downstream ML models

**Data Flow:**
1. Screenshots placed in `data/screenshots/`
2. Script validates images and launches Prodigy web interface
3. Manual annotation via web UI (goalkeeper vs not_goalkeeper classification)
4. Export to JSONL format for ML training

## Prodigy Integration

This project uses Prodigy's `image.classification` recipe with two labels:
- `goalkeeper`: When goalkeeper UI indicator (glove icon) is visible
- `not_goalkeeper`: When no goalkeeper indicator is present

The annotation script uses subprocess calls to invoke Prodigy commands rather than direct API integration.

## Technical Requirements

- **Python 3.12+** (strictly required)
- **UV Package Manager** for dependency management
- **Valid Prodigy License** with authentication token in pyproject.toml
- Web browser for Prodigy annotation interface

## Data Collection Guidelines

- Extract screenshots showing clear goalkeeper UI indicators when present
- Include variety: different game scenarios, lighting, camera angles
- Balance the dataset with roughly equal positive/negative examples
- Ensure image quality allows clear visibility of UI elements
- Recommended extraction interval: 1-2 seconds from gameplay videos