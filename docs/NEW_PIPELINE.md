# New 5-Stage Pipeline Architecture

## Overview

This document describes the new 5-stage pipeline architecture for the FVC Binary Video Classifier project.

## Pipeline Stages

### Stage 1: Video Augmentation
**Input**: N original videos  
**Output**: 11N videos (N original + 10N augmented)  
**Location**: `lib/pipeline_stage1_augmentation.py`

- Generates 10 augmentations per video
- Preserves original resolution and aspect ratio
- Augmentation types: rotation, flip, brightness, contrast, saturation, gaussian_noise, gaussian_blur, affine, elastic
- Saves augmented videos as MP4 files
- Creates metadata CSV: `data/augmented_videos/augmented_metadata.csv`

### Stage 2: Extract Handcrafted Features (M features)
**Input**: 11N videos (from Stage 1)  
**Output**: M features per video  
**Location**: `lib/pipeline_stage2_features.py`

- Extracts handcrafted features from all 11N videos
- Uses existing feature extraction functions:
  - Noise residual energy
  - DCT band statistics
  - Blur/sharpness metrics
  - Block boundary inconsistency
  - Codec cues
- Saves features as `.npy` files
- Creates metadata CSV: `data/features_stage2/features_metadata.csv`
- M = number of features extracted (typically ~50)

### Stage 3: Downscale Videos
**Input**: 11N videos (from Stage 1)  
**Output**: 11N downscaled videos  
**Location**: `lib/pipeline_stage3_downscale.py`

- Downscales all videos to manageable sizes
- Methods:
  - **Resolution reduction**: Simple resize with letterboxing (default)
  - **Autoencoder**: Pretrained autoencoder for dimensionality reduction (optional)
- Target size: 224x224 (configurable)
- Preserves all videos (original + augmented)
- Creates metadata CSV: `data/downscaled_videos/downscaled_metadata.csv`

### Stage 4: Extract Additional Features from Downscaled Videos (P features)
**Input**: 11N downscaled videos (from Stage 3)  
**Output**: P additional features per video  
**Location**: `lib/pipeline_stage4_features_downscaled.py`

- Extracts new features specific to downscaled videos:
  - Block artifact strength
  - Edge preservation metrics
  - Texture uniformity
  - Frequency domain features (DCT on downscaled frames)
  - Compression artifact visibility
  - Color consistency
- These features are different from Stage 2 features (detectable on downscaled videos)
- Saves features as `.npy` files
- Creates metadata CSV: `data/features_stage4/features_downscaled_metadata.csv`
- P = number of additional features (typically ~20)

### Stage 5: Training
**Input**: 
- 11N downscaled videos (from Stage 3)
- M features (from Stage 2)
- P features (from Stage 4)

**Output**: Trained models  
**Location**: `lib/pipeline_stage5_training.py`

- Trains models using combined inputs:
  - Downscaled video frames
  - M handcrafted features (from original videos)
  - P additional features (from downscaled videos)
- Supports:
  - Baseline models (Logistic Regression, SVM) - use features only
  - PyTorch models (CNN, ViT, etc.) - use videos + features
- K-fold cross-validation
- No augmentation or feature generation in this stage

## Data Flow

```
N original videos
    ↓
Stage 1: Augmentation
    ↓
11N videos (N original + 10N augmented)
    ↓
    ├─→ Stage 2: Extract M features → M features per video
    │
    └─→ Stage 3: Downscale → 11N downscaled videos
            ↓
        Stage 4: Extract P features → P features per video
            ↓
        Stage 5: Training (videos + M features + P features)
```

## Usage

### Run All Stages

```bash
python src/run_new_pipeline.py \
    --project-root /path/to/project \
    --num-augmentations 10 \
    --model-types logistic_regression svm naive_cnn \
    --n-splits 5
```

### Run Specific Stages

```bash
# Only run Stage 1 (augmentation)
python src/run_new_pipeline.py --only-stage 1

# Skip Stage 1 (use existing augmentations)
python src/run_new_pipeline.py --skip-stage 1

# Run only Stage 5 (training)
python src/run_new_pipeline.py --only-stage 5
```

### Stage-Specific Options

**Stage 1:**
- `--num-augmentations`: Number of augmentations per video (default: 10)

**Stage 3:**
- Method: `resolution` (default) or `autoencoder`
- Target size: Configurable (default: 224x224)

**Stage 5:**
- `--model-types`: List of models to train
- `--n-splits`: Number of k-fold splits (default: 5)

## Output Structure

```
data/
├── augmented_videos/
│   ├── video1_original.mp4
│   ├── video1_aug0.mp4
│   ├── video1_aug1.mp4
│   ├── ...
│   └── augmented_metadata.csv
├── features_stage2/
│   ├── video1_original_features.npy
│   ├── video1_aug0_features.npy
│   ├── ...
│   └── features_metadata.csv
├── downscaled_videos/
│   ├── video1_original_downscaled.mp4
│   ├── video1_aug0_downscaled.mp4
│   ├── ...
│   └── downscaled_metadata.csv
├── features_stage4/
│   ├── video1_original_downscaled_features_downscaled.npy
│   ├── video1_aug0_downscaled_features_downscaled.npy
│   ├── ...
│   └── features_downscaled_metadata.csv
└── training_results/
    ├── logistic_regression/
    ├── svm/
    └── ...
```

## Key Differences from Old Pipeline

1. **No fixed-size preprocessing during augmentation**: Original resolution preserved
2. **Two-stage feature extraction**: 
   - M features from original videos
   - P features from downscaled videos
3. **Downscaling as separate stage**: Happens after augmentation and feature extraction
4. **Combined inputs for training**: Videos + M features + P features
5. **No augmentation during training**: All augmentations pre-generated in Stage 1
6. **No feature generation during training**: All features pre-extracted in Stages 2 and 4

## Memory Considerations

- Each stage processes one video at a time
- Aggressive garbage collection after each video
- Features saved to disk immediately
- Videos saved to disk immediately
- No accumulation of data in memory

## Resuming Pipeline

The pipeline can be resumed from any stage:
- Use `--skip-stage` to skip completed stages
- Use `--only-stage` to run only specific stages
- Each stage checks for existing outputs and can be re-run independently

## Integration with Existing Code

- Uses existing `handcrafted_features.py` for Stage 2
- Uses model factory (`lib/training/model_factory.py`) to create models:
  - `logistic_regression.py` - LogisticRegressionBaseline
  - `svm.py` - SVMBaseline
  - `naive_cnn.py` - NaiveCNNBaseline
  - `vit_gru.py` - ViTGRUModel
  - `vit_transformer.py` - ViTTransformerModel
  - `slowfast.py` - SlowFastModel
  - `x3d.py` - X3DModel
- Uses existing `model_factory.py` for model creation
- New pipeline modules are separate from old pipeline code

