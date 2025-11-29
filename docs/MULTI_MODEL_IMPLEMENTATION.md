# Multi-Model Implementation for AURA Project

## Overview

This document describes the implementation of 7 model architectures as specified in the AURA project proposal, integrated into the existing MLOps pipeline with robust checkpointing, OOM resistance, and dead kernel recovery.

## Implemented Models

### Baselines (3 models)
1. **Logistic Regression** (`logistic_regression`) - Handcrafted features + sklearn LogisticRegression
2. **Linear SVM** (`svm`) - Handcrafted features + sklearn LinearSVC
3. **Naive CNN** (`naive_cnn`) - Simple 2D CNN over uniformly sampled frames

### Frame→Temporal Models (2 models)
4. **ViT-B/16 + GRU** (`vit_gru`) - Vision Transformer backbone with GRU temporal head
5. **ViT-B/16 + Transformer** (`vit_transformer`) - Vision Transformer backbone with Transformer encoder temporal head

### Spatiotemporal Models (2 models)
6. **SlowFast** (`slowfast`) - Dual-pathway network (slow + fast pathways)
7. **X3D** (`x3d`) - Efficient 3D CNN for video recognition

## Architecture Details

### Handcrafted Features (`lib/handcrafted_features.py`)

**Features Extracted**:
- **Noise Residual Energy**: High-pass filtering to extract noise patterns
- **DCT Band Statistics**: DCT coefficients with DC/AC separation, low/high frequency analysis
- **Blur/Sharpness Metrics**: Laplacian variance, gradient magnitude, Tenengrad, Brenner gradient
- **Block Boundary Inconsistency**: Detection of compression artifacts at block boundaries
- **Codec Cues**: Metadata extraction (bitrate, fps) via ffprobe

**Caching**: Features are cached to disk to avoid recomputation across runs/folds.

### Baseline Models (`lib/baseline_models.py`)

**LogisticRegressionBaseline**:
- Extracts handcrafted features per video
- Trains sklearn LogisticRegression
- Saves feature extractor + model + scaler

**SVMBaseline**:
- Same feature extraction
- Trains sklearn LinearSVC
- Converts decision function to probabilities via sigmoid

**NaiveCNNBaseline**:
- Simple 2D CNN: Conv2D blocks → GlobalAvgPool → FC
- Processes frames independently, averages predictions
- PyTorch nn.Module (can be trained with standard training loop)

### Frame→Temporal Models (`lib/frame_temporal_models.py`)

**ViTGRUModel**:
- Backbone: `timm.create_model('vit_base_patch16_224', pretrained=True)`
- Extracts [CLS] token (768-dim) per frame
- Temporal head: GRU (hidden_dim=256, num_layers=2)
- Classification: Linear(256, 1)

**ViTTransformerModel**:
- Same ViT backbone
- Temporal head: Transformer encoder (d_model=768, nhead=8, num_layers=2)
- Mean pooling over temporal dimension
- Classification: Linear(768, 1)

### Spatiotemporal Models (`lib/spatiotemporal_models.py`)

**SlowFastModel**:
- Uses `torchvision.models.video.slowfast_r50` if available
- Fallback: Simplified SlowFast implementation
- Slow pathway: 16 frames at 2 fps
- Fast pathway: 64 frames at 8 fps
- Fusion and binary classification head

**X3DModel**:
- Uses `torchvision.models.video.x3d_m` (pretrained on Kinetics-400)
- Fallback: Uses `r3d_18` as approximation
- Binary classification head

## Model Factory (`lib/model_factory.py`)

**Centralized Model Creation**:
- `create_model(model_type, config)`: Creates model instance
- `get_model_config(model_type)`: Returns memory-optimized config
- `list_available_models()`: Lists all available model types
- `is_pytorch_model(model_type)`: Checks if model is PyTorch or sklearn

**Memory-Optimized Configs**:
```python
MODEL_MEMORY_CONFIGS = {
    "logistic_regression": {"batch_size": 64, "num_workers": 2, ...},
    "svm": {"batch_size": 64, "num_workers": 2, ...},
    "naive_cnn": {"batch_size": 16, "num_workers": 2, ...},
    "vit_gru": {"batch_size": 4, "num_workers": 1, "gradient_accumulation": 4, ...},
    "vit_transformer": {"batch_size": 2, "num_workers": 1, "gradient_accumulation": 8, ...},
    "slowfast": {"batch_size": 2, "num_workers": 1, "gradient_accumulation": 8, ...},
    "x3d": {"batch_size": 4, "num_workers": 1, "gradient_accumulation": 4, ...},
}
```

## Multi-Model Pipeline (`lib/mlops_pipeline_multimodel.py`)

**Pipeline Stages**:
1. **Load Data** (shared): Load and filter metadata
2. **Create K-Fold Splits** (shared): Stratified k-fold splits
3. **Generate Shared Augmentations** (shared): Pre-generate augmentations once for all videos
4. **Train All Models** (sequential):
   - For each model type:
     - Check if already complete (skip if done)
     - Get model-specific memory config
     - For each fold:
       - Filter augmentations for fold's training videos
       - Create datasets and loaders
       - Train model (PyTorch or sklearn)
       - Evaluate and save metrics
       - Cleanup after fold
     - Save fold results and mark as complete
     - Aggressive GC after model

**Key Features**:
- **Resume Capability**: Checks `training_complete.pt` to skip completed models
- **Per-Model Checkpoints**: Each model has its own checkpoint directory
- **Shared Data Pipeline**: Data loading, splitting, and augmentation done once
- **Memory Efficient**: Aggressive GC after each model/fold
- **OOM Handling**: Wrapped in `safe_execute` with retry logic

## Model Comparison (`lib/model_comparison.py`)

**Evaluation Utilities**:
- `evaluate_all_models()`: Evaluate all models on same test set
- `compare_models()`: Create comparison DataFrame
- `plot_model_comparison()`: Generate comparison plots (if matplotlib available)

## Usage

### Running Multi-Model Pipeline

**Default (all models)**:
```bash
python3 src/run_mlops_pipeline.py
```

**Custom model selection**:
```bash
export MODELS_TO_TRAIN="vit_gru,slowfast,x3d"
python3 src/run_mlops_pipeline.py
```

**Disable multi-model (single model)**:
```bash
export USE_MULTIMODEL=false
python3 src/run_mlops_pipeline.py
```

### Model Training Order

Models are trained in this order (by default):
1. `logistic_regression` (fast, low memory)
2. `svm` (fast, low memory)
3. `naive_cnn` (medium memory)
4. `vit_gru` (medium-high memory)
5. `vit_transformer` (high memory)
6. `slowfast` (very high memory)
7. `x3d` (high memory)

### Checkpoint Structure

```
runs/run_XXX/
  checkpoints/
    logistic_regression/
      fold_1/
        best_model.pt
        latest_checkpoint.pt
      ...
    vit_gru/
      fold_1/
        ...
  models/
    logistic_regression/
      fold_1/
        model.joblib
        scaler.joblib
      fold_results.csv
      training_complete.pt
    vit_gru/
      fold_1/
        ...
      fold_results.csv
      training_complete.pt
```

### Resume Capability

The pipeline automatically:
- Checks `training_complete.pt` for each model
- Skips completed models
- Resumes from latest checkpoint for incomplete models
- Can be interrupted and restarted at any time

## Memory Optimizations

### Per-Model Configurations

Each model has conservative memory settings:
- **Baselines**: Large batch sizes (64), minimal memory
- **Frame→Temporal**: Small batches (2-4), gradient accumulation (4-8x)
- **Spatiotemporal**: Very small batches (2-4), high gradient accumulation (4-8x)

### Aggressive Garbage Collection

- After each model
- After each fold
- After each epoch (during training)
- After each batch (for augmentation generation)

### OOM Handling

- `safe_execute()` wrapper with retry logic
- Automatic batch size reduction on OOM
- Detailed memory logging
- Graceful error handling

## Dependencies

**New dependencies added**:
- `timm>=0.9.0`: For ViT models
- `scikit-learn>=1.3.0`: For baseline models
- `scipy>=1.11.0`: For DCT, signal processing
- `joblib>=1.3.0`: For model serialization

## File Structure

```
lib/
  handcrafted_features.py      # Feature extraction
  baseline_models.py           # LogisticRegression, SVM, NaiveCNN
  frame_temporal_models.py    # ViT+GRU, ViT+Transformer
  spatiotemporal_models.py   # SlowFast, X3D
  model_factory.py            # Model creation and config
  mlops_pipeline_multimodel.py # Multi-model pipeline
  model_comparison.py          # Evaluation and comparison
  mlops_core.py               # Extended RunConfig (model_type, model_specific_config)
```

## Training Strategy

### Sequential Training
1. Train baselines first (fast, low memory)
2. Train frame→temporal models (medium memory)
3. Train spatiotemporal models (high memory, last)

### Checkpointing
- Each model has separate checkpoint directory
- Resume from last incomplete model on restart
- Skip completed models automatically
- Per-fold checkpoints for PyTorch models

### Evaluation
- All models evaluated on same k-fold splits
- Same metrics (accuracy, precision, recall, F1)
- Per-model fold results saved to CSV
- Average metrics computed across folds

## OOM and Dead Kernel Resistance

1. **Per-model memory limits**: Conservative batch sizes
2. **Gradient accumulation**: Compensate for small batches
3. **Aggressive GC**: After each model, fold, epoch
4. **Checkpointing**: Save after each fold, resume from latest
5. **Error handling**: Catch OOM, reduce batch size, retry
6. **Progress tracking**: Log which models/folds are complete

## Notes

- All models use same data pipeline (shared augmentations)
- All models evaluated on same splits for fair comparison
- Handcrafted features cached to disk to avoid recomputation
- Model checkpoints are independent (can resume any model)
- Pipeline designed to run over multiple sessions (checkpointed)
- Augmentations are deterministic (seeded by video path hash)

