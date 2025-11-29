# FVC Binary Video Classifier

A deep learning project for binary classification of videos (real vs. fake) using 3D Convolutional Neural Networks with transfer learning.

## Overview

This project implements a binary video classifier for the FVC (Fake Video Classification) dataset. The model uses a pretrained 3D ResNet backbone (Kinetics-400 weights) with a custom Inception-like head for binary classification.

## Key Features

- **Pretrained 3D CNN**: Uses `torchvision.models.video.r3d_18` with Kinetics-400 weights
- **Fixed-Size Preprocessing**: 224x224 with letterboxing for memory efficiency
- **Comprehensive Augmentations**: Pre-generated spatial and temporal augmentations
- **K-Fold Cross-Validation**: Stratified 5-fold CV for robust evaluation
- **MLOps Infrastructure**: Experiment tracking, checkpointing, and data versioning
- **Memory Optimized**: Aggressive GC, OOM handling, progressive batch size fallback
- **Class Imbalance Handling**: Balanced batch sampling and inverse-frequency weights

## Project Structure

```
fvc/
├── archive/                    # Original dataset archives
├── data/                       # Processed metadata
├── docs/                       # Documentation
├── lib/                        # Core library modules
├── src/                        # Scripts and notebooks
├── videos/                     # Extracted video files
├── runs/                       # Experiment outputs
├── models/                     # Saved models
└── logs/                       # Log files
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Dataset

```bash
python3 src/setup_fvc_dataset.py
```

### 3. Run Training

**Option A: Using MLOps Pipeline (Recommended)**
```bash
python3 src/run_mlops_pipeline.py
```

**Option B: Using 5-Stage Pipeline**
```bash
python3 src/run_new_pipeline.py
```

**Option C: Using Individual Stage Scripts**
```bash
# Run stages individually
python3 src/scripts/run_stage1_augmentation.py
python3 src/scripts/run_stage2_features.py
python3 src/scripts/run_stage3_downscaling.py
python3 src/scripts/run_stage4_downscaled_features.py
python3 src/scripts/run_stage5_training.py
```

See [src/scripts/README.md](src/scripts/README.md) for detailed usage.

**Option D: Using SLURM (Cluster)**
```bash
sbatch src/scripts/run_fvc_training.sh
```

## Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) - Comprehensive project documentation
- [MLOps Optimizations](docs/MLOPS_OPTIMIZATIONS.md) - MLOps system design details
- [OOM and K-Fold Optimizations](docs/OOM_AND_KFOLD_OPTIMIZATIONS.md) - Memory management and CV
- [Changelog](docs/CHANGELOG.md) - Version history

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- `torch`, `torchvision`: Deep learning framework
- `polars`: Fast DataFrame operations
- `opencv-python`, `av`, `torchcodec`: Video decoding
- `papermill`, `ipykernel`: Notebook execution

## Model Architecture

- **Backbone**: 3D ResNet-18 (r3d_18) pretrained on Kinetics-400
- **Head**: Inception3DBlock → AdaptiveAvgPool3d → Dropout(0.5) → Linear(512, 1)
- **Input**: (N, C=3, T=16, H=224, W=224)
- **Output**: Binary logits (N, 1)

## Training Configuration

- **Frames per video**: 16 (uniformly sampled)
- **Input size**: 224x224 (fixed with letterboxing)
- **Batch size**: 32 (with progressive fallback: 16 → 8 → 4 → 2 → 1)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Mixed Precision**: Enabled (AMP)
- **Cross-Validation**: 5-fold stratified

## Results

Results are saved in `runs/` directory with:
- Experiment configurations (`config.json`)
- Metrics logs (`metrics.jsonl`)
- Model checkpoints (`checkpoints/`)
- System metadata (`metadata.json`)

## License

This project is part of SI670 Applied Machine Learning course.

## Author

Horopter (santoshdesai12@hotmail.com)

