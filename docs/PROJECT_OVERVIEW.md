# AURA - Authenticity Understanding of Real vs Artificial Shortform Videos - Project Overview

## Project Description

AURA (Authenticity Understanding of Real vs Artificial) is a comprehensive system for detecting fake videos in shortform content. This project implements multiple model architectures for binary video classification using the **Fake Video Corpus (FVC)** dataset. The goal is to distinguish between real and fake videos using deep learning techniques, specifically 3D Convolutional Neural Networks (3D CNNs) with transfer learning, handcrafted features, and ensemble methods.

### Dataset

This project uses the **Fake Video Corpus (FVC)** dataset from [MKLab-ITI](https://github.com/MKLab-ITI/fake-video-corpus), created by Olga Papadopoulou, Markos Zampoglou, Symeon Papadopoulos, and Ioannis Kompatsiaris. The dataset consists of 3957 videos annotated as fake and 2458 annotated as real, including near-duplicate versions collected from YouTube, Facebook, and Twitter.

**Citation**:
```bibtex
@article{papadopoulou2018corpus,
  author = "Papadopoulou, Olga and Zampoglou, Markos and Papadopoulos, Symeon and Kompatsiaris, Ioannis",
  title = "A corpus of debunked and verified user-generated videos",
  journal = "Online Information Review",
  doi = "10.1108/OIR-03-2018-0101",
  year={2018},
  publisher={Emerald Publishing Limited}
}
```

For questions about the dataset, please contact Olga Papadopoulou at olgapapa@iti.gr.

## Initial Approach

### 1. Problem Statement
- **Task**: Binary classification of videos (real vs. fake)
- **Challenge**: Videos have varying properties (resolution, aspect ratio, frame count, fps, codec, bitrate)
- **Dataset**: FVC dataset with metadata including platform, video_id, labels, and technical properties

### 2. Initial Architecture Decisions

#### Model Architecture
- **Backbone**: Pretrained 3D ResNet (`torchvision.models.video.r3d_18`) with Kinetics-400 weights
- **Head**: Custom Inception-like 3D block + global average pooling + binary classification head
- **Rationale**: 
  - Transfer learning from Kinetics-400 (large-scale video dataset) provides strong feature representations
  - 3D CNNs capture both spatial and temporal information
  - Inception blocks increase model capacity without excessive parameters

#### Video Preprocessing Strategy
- **Initial Approach**: Variable aspect ratio preservation
  - Resize longer edge to `max_size` while maintaining aspect ratio
  - Use adaptive pooling to handle variable dimensions
  - Custom collate function for batch padding
  
- **Evolution**: Fixed-size preprocessing with letterboxing
  - Resize to fixed 112x112 (configurable via `FVC_FIXED_SIZE`, default 112) with letterboxing
  - Benefits: Consistent batch dimensions, better GPU memory utilization, 4x memory reduction vs 224x224
  - Trade-off: Slight information loss from letterboxing and lower resolution, but significant memory gains

#### Frame Sampling
- **Uniform Sampling**: Extract `num_frames` frames uniformly across video duration
- **Rolling Window**: Optional temporal windowing for longer videos (not used in final implementation)
- **Frame Count**: Started with 8 frames, increased to 16, then reduced to 6 frames for ultra-conservative memory usage

### 3. Data Augmentation Strategy

#### Initial Approach: On-the-Fly Augmentation
- Augmentations applied during training
- Pros: No disk space required
- Cons: Slower training, non-reproducible

#### Final Approach: Pre-Generated Augmentations
- **Spatial Augmentations** (per-frame):
  - Geometric: Rotation, affine transformations (translation, scale, shear), horizontal flip
  - Color: Color jitter (brightness, contrast, saturation, hue)
  - Noise: Gaussian noise injection
  - Blur: Gaussian blur
  - Occlusion: Random cutout (random erasing)
  - Distortion: Elastic transform (simplified)
  
- **Temporal Augmentations** (sequence-level):
  - Frame dropping: Randomly remove frames (up to 25%)
  - Frame duplication: Randomly duplicate frames (slow motion effect)
  - Temporal reversal: Reverse video sequence
  
- **Implementation**: 
  - Pre-generate 1 augmented version per training video (reduced from 3 for memory efficiency)
  - Store as `.pt` (PyTorch tensor) files
  - **Frame-by-frame decoding**: Decode only the 6 needed frames instead of loading entire videos (50x memory reduction)
  - **Incremental CSV writing**: Write metadata directly to CSV to avoid memory accumulation
  - Benefits: Faster training, reproducibility, can cache on disk, minimal memory usage

### 4. Training Strategy

#### Class Imbalance Handling
- **Problem**: Dataset may have imbalanced real/fake classes
- **Solution**: 
  - Inverse-frequency class weights
  - Balanced batch sampling (equal number of each class per batch)
  - Weighted random sampler as fallback

#### Optimization
- **Optimizer**: Adam with learning rate 1e-4, weight decay 1e-4
- **Scheduler**: StepLR (not actively used in final implementation)
- **Mixed Precision**: Automatic Mixed Precision (AMP) for faster training and lower memory
- **Gradient Accumulation**: Dynamic based on batch size
  - `batch_size >= 8`: No accumulation
  - `batch_size >= 4`: No accumulation
  - `batch_size >= 2`: 2x accumulation
  - `batch_size = 1`: 4x accumulation

#### Batch Size Strategy
- **Progressive Fallback**: Try larger batch sizes first, fallback on OOM
  - Start: `batch_size=32` (16 real + 16 fake per batch)
  - Fallbacks: 16 → 8 → 4 → 2 → 1
- **Balanced Sampling**: Ensures equal class representation per batch

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC (not implemented in final version)
- Per-class metrics

## Key Issues Encountered and Solutions

### Issue 1: Out-of-Memory (OOM) Errors

**Problem**: 
- GPU memory exhaustion during training
- Kernel death on SLURM cluster
- Loss reporting as 0.0000 due to memory issues

**Root Causes**:
- Large video resolutions (some videos > 1920x1080)
- Variable batch dimensions requiring padding
- Accumulated memory from data loading
- Insufficient garbage collection

**Solutions Implemented**:
1. **Fixed-Size Preprocessing**: Downscale all videos to 112x112 with letterboxing (configurable)
   - Reduces memory per sample by ~40-80x vs original high-res videos
   - Enables larger batch sizes (though we use conservative batch sizes 1-8)
   
2. **Frame-by-Frame Video Decoding**: Decode only the 6 needed frames instead of loading entire videos
   - Reduces per-video memory from ~1.87 GB to ~37 MB (50x reduction)
   - Uses PyAV to seek and decode specific frames
   - Fallback to full video loading if frame-by-frame decoding fails
   
3. **Incremental CSV Writing**: Write augmented metadata directly to CSV
   - Eliminates unbounded memory growth from accumulating metadata lists
   - Memory stays constant regardless of dataset size
   
4. **Aggressive Garbage Collection**:
   - `aggressive_gc()` function: Multiple passes of `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`
   - Called after every pipeline stage, every video, every batch, after each epoch
   
5. **Memory-Optimized Data Loading**:
   - Set `num_workers=0` (CPU-only or test mode to avoid multiprocessing memory overhead)
   - Process videos one at a time during augmentation generation
   - Clear video tensors and clips immediately after processing
   
6. **OOM Error Handling**:
   - `check_oom_error()`: Detects OOM from multiple error messages
   - `handle_oom_error()`: Performs cleanup and logging
   - `safe_execute()`: Wraps functions with retry logic
   - Automatic batch size reduction on OOM

### Issue 2: Overfitting

**Problem**:
- Loss reported as 0.0000 (suspiciously low)
- Identical logits across batches
- Low gradient norms
- Only 12 trainable parameters (backbone frozen)

**Root Causes**:
- Backbone frozen too aggressively
- Insufficient data augmentation
- No regularization
- Single train/val split (not robust)

**Solutions Implemented**:
1. **K-Fold Cross-Validation**:
   - Stratified 5-fold CV to prevent overfitting/underfitting
   - More robust performance estimates
   - Better generalization assessment
   
2. **Comprehensive Augmentations**:
   - Pre-generated augmentations (3x per video)
   - Spatial + temporal augmentations
   - Configurable augmentation parameters
   
3. **Backbone Unfreezing**:
   - Option to unfreeze backbone for fine-tuning
   - Gradual unfreezing strategy (not implemented, but available)
   
4. **Enhanced Diagnostics**:
   - Logits statistics (min, max, mean, std, unique count)
   - Gradient norms
   - Parameter counts
   - Loss verification (manual computation)

### Issue 3: Data Pipeline Issues

**Problem**:
- Missing video files causing crashes
- Inconsistent path resolution
- Partial unzip issues (zip bomb detection)

**Solutions Implemented**:
1. **Path Resolution**:
   - Centralized `lib/video_paths.py` with `resolve_video_path()`
   - Multiple fallback strategies
   - Consistent path handling across codebase
   
2. **Video Filtering**:
   - `filter_existing_videos()` before splitting
   - Optional frame count verification
   - Graceful handling of missing files
   
3. **Dataset Setup**:
   - `setup_fvc_dataset.py`: Handles unzipping with `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`
   - Path verification for `fvc/videos/FVC[1-3]`
   - Metadata generation and validation

### Issue 4: Reproducibility and Experiment Management

**Problem**:
- No experiment tracking
- No configuration versioning
- Metrics scattered in logs
- No resume capability

**Solutions Implemented**:
1. **MLOps Core Infrastructure**:
   - `RunConfig`: Centralized configuration with deterministic hashing
   - `ExperimentTracker`: Structured metrics logging (JSONL format)
   - `CheckpointManager`: Full state saving (model, optimizer, scheduler, epoch)
   - `DataVersionManager`: Data lineage tracking
   
2. **Pipeline Orchestration**:
   - `PipelineStage`: Modular stages with dependencies
   - `MLOpsPipeline`: Automatic dependency resolution
   - Stage-level checkpointing for fault tolerance
   - Resume from latest checkpoint
   
3. **K-Fold Integration**:
   - `build_kfold_pipeline()`: Stratified K-fold with per-fold tracking
   - Per-fold checkpointing and metrics
   - Average metrics across folds

### Issue 5: Code Quality and Maintainability

**Problem**:
- Syntax errors (`=== None` instead of `is None`)
- Missing imports
- Function signature mismatches
- Inconsistent error handling

**Solutions Implemented**:
1. **Comprehensive Code Quality Checks**:
   - Python syntax validation
   - Indentation checks
   - Function signature verification
   - Import validation
   - Version compatibility checks
   
2. **Error Handling**:
   - Try-except blocks with proper logging
   - Graceful degradation (fallback strategies)
   - Warning capture (non-breaking warnings logged, not raised)
   
3. **Code Organization**:
   - Modular library structure (`lib/`)
   - Clear separation of concerns
   - Comprehensive documentation

## 5-Stage Pipeline Architecture

### Stage 1: Video Augmentation
**Input**: N original videos  
**Output**: 11N videos (N original + 10N augmented)  
**Location**: `lib/augmentation/pipeline.py`

- Generates 10 augmentations per video (configurable, default: 1 for memory efficiency)
- Preserves original resolution and aspect ratio
- Augmentation types: rotation, flip, brightness, contrast, saturation, gaussian_noise, gaussian_blur, affine, elastic
- Saves augmented videos as MP4 files
- Creates metadata: `data/augmented_videos/augmented_metadata.arrow` (or `.parquet`/`.csv`)

### Stage 2: Extract Handcrafted Features (M features)
**Input**: 11N videos (from Stage 1)  
**Output**: M features per video  
**Location**: `lib/features/pipeline.py`

- Extracts handcrafted features from all 11N videos
- Features: Noise residual energy, DCT band statistics, blur/sharpness metrics, block boundary inconsistency, codec cues
- Saves features as `.npy` files
- Creates metadata: `data/features_stage2/features_metadata.arrow` (or `.parquet`/`.csv`)
- M = number of features extracted (typically ~50)

### Stage 3: Scale Videos
**Input**: 11N videos (from Stage 1)  
**Output**: 11N scaled videos  
**Location**: `lib/scaling/pipeline.py`

- Scales all videos to target max dimension (can both downscale and upscale)
- Methods:
  - **Letterbox resize**: Simple resize with letterboxing (default)
  - **Autoencoder**: Pretrained Hugging Face VAE for high-quality scaling (optional)
- Target max dimension: 112 pixels (max(width, height) = 112, configurable via `FVC_FIXED_SIZE`)
- Preserves aspect ratio for all videos
- Creates metadata: `data/scaled_videos/scaled_metadata.arrow` (or `.parquet`/`.csv`)

### Stage 4: Extract Additional Features from Scaled Videos (P features)
**Input**: 11N scaled videos (from Stage 3)  
**Output**: P features per video  
**Location**: `lib/features/scaled.py`

- Extracts additional features from scaled videos
- Same feature types as Stage 2, but from scaled videos
- Saves features as `.npy` files
- Creates metadata: `data/features_stage4/features_scaled_metadata.arrow` (or `.parquet`/`.csv`)
- P = number of features extracted (typically ~50)

### Stage 5: Model Training
**Input**: Scaled videos (Stage 3) + Features (Stage 2 & 4)  
**Output**: Trained models and metrics  
**Location**: `lib/training/pipeline.py`

- Trains multiple model architectures
- Uses 5-fold stratified cross-validation
- Models: Logistic Regression, SVM, Naive CNN, ViT-GRU, ViT-Transformer, SlowFast, X3D
- Output: `data/stage5/<model_type>/` with checkpoints, metrics, and plots

## Final Architecture

### Model
- **Backbone**: `torchvision.models.video.r3d_18` (Kinetics-400 weights)
- **Head**: Inception3DBlock → AdaptiveAvgPool3d → Dropout(0.5) → Linear(512, 1)
- **Input**: (N, C=3, T=6, H=112, W=112) - Ultra-conservative for memory efficiency
- **Output**: Binary logits (N, 1)

### Data Pipeline
1. **Setup**: `setup_fvc_dataset.py` extracts videos and generates metadata
2. **Preprocessing**: Fixed-size 112x112 with letterboxing, uniform frame sampling (6 frames)
3. **Augmentation**: Pre-generated (1x per video) with spatial + temporal augmentations, frame-by-frame decoding
4. **Loading**: Polars DataFrames, PyTorch DataLoader with balanced batch sampling

### Training Pipeline
1. **Data Loading**: Load metadata, filter existing videos, stratified split
2. **Augmentation**: Pre-generate augmented clips (or skip if exists)
3. **Dataset Creation**: VideoDataset with pre-generated augmentations
4. **Model Initialization**: PretrainedInceptionVideoModel
5. **Training**: K-fold cross-validation with aggressive GC and OOM handling
6. **Evaluation**: Per-fold metrics, average across folds

### MLOps Integration
- **Experiment Tracking**: Unique run IDs, config hashing, metrics logging
- **Checkpointing**: Full state saving, resume capability, best model tracking
- **Pipeline Orchestration**: Modular stages, dependency management, fault tolerance
- **Data Versioning**: Split and augmentation versioning with config linkage

## Performance Optimizations

### Memory
- Fixed-size preprocessing (112x112) reduces memory by 40-80x vs original videos
- Frame-by-frame decoding reduces per-video memory by 50x (only 6 frames loaded)
- Incremental CSV writing eliminates memory accumulation
- Ultra-conservative batch sizes (1-8) with high gradient accumulation (8-16 steps)
- Aggressive GC after every stage/video/batch/epoch
- OOM error detection and automatic retry
- Comprehensive memory profiling with detailed object breakdowns

### Speed
- Pre-generated augmentations (no augmentation overhead during training)
- Frame-by-frame decoding (only decode needed frames, not entire videos)
- Mixed precision training (AMP)
- Sequential processing (one video at a time) to minimize memory spikes

### Robustness
- K-fold cross-validation for better generalization
- Comprehensive error handling
- Resume capability from checkpoints
- Data validation at every stage

## File Structure

```
fvc/
├── archive/                    # Original dataset archives (DO NOT DELETE)
├── data/                       # Processed metadata
│   ├── video_index_input.csv
│   └── video_index_input.json
├── docs/                       # Documentation
│   ├── PROJECT_OVERVIEW.md     # This file
│   ├── CHANGELOG.md            # Project changelog
│   ├── MEMORY_OPTIMIZATIONS.md  # Comprehensive memory optimizations
│   ├── MLOPS_OPTIMIZATIONS.md  # MLOps system design
│   ├── MULTI_MODEL_IMPLEMENTATION.md  # Multi-model architecture details
│   └── GIT_SETUP.md            # Git repository setup
├── lib/                        # Core library modules
│   ├── mlops_core.py          # MLOps infrastructure
│   ├── mlops_utils.py         # MLOps utilities (GC, OOM handling)
│   ├── mlops_pipeline.py      # Single-split pipeline
│   ├── mlops_pipeline_kfold.py # K-fold pipeline
│   ├── video_modeling.py     # Models and datasets
│   ├── video_data.py          # Data loading and splitting
│   ├── video_training.py     # Training loop
│   ├── video_augmentations.py # Augmentation utilities
│   ├── video_augmentation_pipeline.py # Pre-generation pipeline
│   ├── video_inference.py     # Inference utilities
│   ├── video_explain.py       # Interpretability (Grad-CAM)
│   └── video_paths.py         # Path resolution
├── src/                        # Scripts and notebooks
│   ├── setup_fvc_dataset.py   # Dataset setup
│   ├── teardown_fvc_dataset.py # Dataset cleanup
│   ├── run_new_pipeline.py  # 5-stage pipeline runner
│   ├── dashboard_results.py # Streamlit results dashboard
│   └── scripts/
│       └── run_fvc_training.sh # SLURM batch script
├── videos/                     # Extracted video files
├── runs/                       # Experiment outputs
├── models/                     # Saved models
├── logs/                       # Log files
└── requirements.txt            # Dependencies
```

## Usage

### 1. Setup Dataset
```bash
python3 src/setup_fvc_dataset.py
```

### 2. Run Training (5-Stage Pipeline)
```bash
# Using the unified pipeline runner
python3 src/run_new_pipeline.py

# Or run individual stages
python3 src/scripts/run_stage1_augmentation.py
python3 src/scripts/run_stage2_features.py
python3 src/scripts/run_stage3_scaling.py
python3 src/scripts/run_stage4_scaled_features.py
python3 src/scripts/run_stage5_training.py

# Or using SLURM (cluster)
sbatch scripts/slurm_jobs/slurm_stage5a.sh  # For specific models
```

### 4. Cleanup (Optional)
```bash
python3 src/teardown_fvc_dataset.py
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `torch`, `torchvision`: Deep learning framework
- `polars`: Fast DataFrame operations (replaces pandas)
- `opencv-python`, `av`, `torchcodec`: Video decoding
- `papermill`, `ipykernel`: Notebook execution
- `pyarrow`: Data serialization

## Future Improvements

1. **Model Architecture**:
   - Try Video Transformers (TimeSformer, ViViT)
   - Two-stream networks (RGB + optical flow)
   - SlowFast networks

2. **Training**:
   - Learning rate scheduling (cosine decay, warmup)
   - Advanced loss functions (Focal loss, class-balanced loss)
   - Multi-GPU training

3. **Evaluation**:
   - Per-class precision/recall
   - PR-AUC, calibration curves
   - Robustness checks (by platform, resolution, duration)

4. **Deployment**:
   - Model packaging
   - CLI interface
   - Batch inference API

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

## Model Architecture Details

### Handcrafted Features (`lib/handcrafted_features.py`)

**Features Extracted**:
- **Noise Residual Energy**: High-pass filtering to extract noise patterns
- **DCT Band Statistics**: DCT coefficients with DC/AC separation, low/high frequency analysis
- **Blur/Sharpness Metrics**: Laplacian variance, gradient magnitude, Tenengrad, Brenner gradient
- **Block Boundary Inconsistency**: Detection of compression artifacts at block boundaries
- **Codec Cues**: Metadata extraction (bitrate, fps) via ffprobe

**Caching**: Features are cached to disk to avoid recomputation across runs/folds.

### Baseline Models

**LogisticRegressionBaseline** (`lib/training/_linear.py`):
- Extracts handcrafted features per video
- Trains sklearn LogisticRegression
- Saves feature extractor + model + scaler

**SVMBaseline** (`lib/training/_svm.py`):
- Same feature extraction
- Trains sklearn LinearSVC
- Converts decision function to probabilities via sigmoid

**NaiveCNNBaseline** (`lib/training/_cnn.py`):
- Simple 2D CNN: Conv2D blocks → GlobalAvgPool → FC
- Processes frames independently, averages predictions
- PyTorch nn.Module (can be trained with standard training loop)

### Frame→Temporal Models

**ViTGRUModel** (`lib/training/vit_gru.py`):
- Backbone: `timm.create_model('vit_base_patch16_224', pretrained=True)`
- Extracts [CLS] token (768-dim) per frame
- Temporal head: GRU (hidden_dim=256, num_layers=2)
- Classification: Linear(256, 1)

**ViTTransformerModel** (`lib/training/vit_transformer.py`):
- Same ViT backbone
- Temporal head: Transformer encoder (d_model=768, nhead=8, num_layers=2)
- Mean pooling over temporal dimension
- Classification: Linear(768, 1)

### Spatiotemporal Models

**SlowFastModel** (`lib/training/slowfast.py`):
- Uses `torchvision.models.video.slowfast_r50` if available
- Fallback: Simplified SlowFast implementation
- Slow pathway: 16 frames at 2 fps
- Fast pathway: 64 frames at 8 fps
- Fusion and binary classification head

**X3DModel** (`lib/training/x3d.py`):
- Uses `torchvision.models.video.x3d_m` (pretrained on Kinetics-400)
- Fallback: Uses `r3d_18` as approximation
- Binary classification head

## Reusable Components

### Training Utilities (`lib/training/trainer.py`)

**`fit()`** - High-level training with experiment tracking
- Handles optimizer/scheduler creation
- Full training loop with validation
- Checkpointing and resume capability
- Metrics logging
- Early stopping

**`train_one_epoch()`** - Core training loop
- Handles mixed precision (AMP)
- Gradient accumulation
- Dynamic loss function selection (BCE vs CrossEntropy)
- Memory-efficient with aggressive GC

**`evaluate()`** - Evaluation loop
- Computes loss and accuracy
- Memory-efficient evaluation
- Dynamic loss function selection

### Model Factory (`lib/training/model_factory.py`)

**Centralized Model Creation**:
- `create_model(model_type, config)`: Creates model instance
- `get_model_config(model_type)`: Returns memory-optimized config
- `list_available_models()`: Lists all available model types
- `is_pytorch_model(model_type)`: Checks if model is PyTorch or sklearn

**Memory-Optimized Configs** (Ultra-Conservative):
- Baselines: Moderate batch sizes (8), minimal memory, `num_workers=0`
- Frame→Temporal: Very small batches (1), high gradient accumulation (16x), `num_workers=0`
- Spatiotemporal: Very small batches (1), 6 frames (reduced from 8), high gradient accumulation (16x), `num_workers=0`

### Data Utilities (`lib/data/loading.py`)

**`stratified_kfold()`** - K-fold cross-validation
- Stratified splitting
- Balanced classes
- Handles duplicate groups (prevents data leakage)

**`train_val_test_split()`** - Data splitting
- Stratified splitting
- Duplicate group awareness
- Configurable ratios

**`filter_existing_videos()`** - Video filtering
- Filters missing videos
- Optional frame count verification

## References

- Kinetics-400 Dataset: https://deepmind.com/research/open-source/kinetics
- 3D ResNet: "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (Tran et al., 2018)
- Inception: "Going Deeper with Convolutions" (Szegedy et al., 2015)

