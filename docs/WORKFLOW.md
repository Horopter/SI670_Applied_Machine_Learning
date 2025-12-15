# End-to-End Workflow: Archive to Notebook Generation

## Overview

This document describes the complete workflow from raw archive data through all 5 pipeline stages to generating presentation-ready Jupyter notebooks.

## Prerequisites

- Python 3.10+
- Dependencies installed (`pip install -r requirements.txt`)
- Archive files: `FVC1.zip`, `FVC2.zip`, `FVC3.zip`, `Metadata.zip` in `archive/` directory
- Password for archives: `m3v3r!@`

## Complete Workflow

### Step 0: Dataset Setup

**Script**: `src/setup_fvc_dataset.py`

**Purpose**: Extract videos from password-protected ZIP archives and generate video index manifest.

**Process**:
1. Locates ZIP archives in `archive/` or project root
2. Extracts videos to `videos/FVC1/`, `videos/FVC2/`, `videos/FVC3/`
3. Copies metadata CSV files (`FVC_dup.csv`, `FVC.csv`) to `videos/Metadata/`
4. Generates video index: `data/video_index_input.csv` and `data/video_index_input.json`
5. Validates video paths and metadata integrity

**Output**:
- `data/video_index_input.csv`: Video metadata with paths, labels, and properties
- `data/video_index_input.json`: JSON version of metadata
- `videos/FVC1/`, `videos/FVC2/`, `videos/FVC3/`: Extracted video files

**Usage**:
```bash
python src/setup_fvc_dataset.py
```

---

### Step 1: Video Augmentation

**Script**: `src/scripts/run_stage1_augmentation.py` or `src/run_new_pipeline.py --only-stage 1`

**Function**: `lib.augmentation.pipeline.stage1_augment_videos()`

**Purpose**: Generate augmented versions of each video using spatial and temporal transformations.

**Input**: `data/video_index_input.csv` (or `data/FVC_dup.csv`)

**Process**:
1. Loads video metadata from input CSV
2. For each video:
   - Loads video using PyAV (frame-by-frame decoding)
   - Samples frames uniformly (default: 6 frames)
   - Applies 10 augmentations per video:
     - Spatial: rotation, flip, brightness, contrast, saturation, noise, blur, affine, cutout
     - Temporal: frame drop, frame duplicate, temporal reverse
   - Saves augmented videos as MP4 files
   - Writes metadata row to CSV/Arrow
3. Processes videos in chunks to prevent OOM
4. Uses incremental CSV writing to avoid memory accumulation

**Output**:
- `data/augmented_videos/augmented_metadata.arrow` (or `.parquet`/`.csv`)
- `data/augmented_videos/*.mp4`: Augmented video files
- Total videos: N original + 10N augmented = 11N videos

**Key Features**:
- Frame-by-frame decoding (50x memory reduction)
- Deterministic seeding for reproducibility
- Resume capability (skips existing augmentations)
- Chunked processing for long videos

**Usage**:
```bash
python src/scripts/run_stage1_augmentation.py --num-augmentations 10
```

---

### Step 2: Handcrafted Feature Extraction

**Script**: `src/scripts/run_stage2_features.py` or `src/run_new_pipeline.py --only-stage 2`

**Function**: `lib.features.pipeline.stage2_extract_features()`

**Purpose**: Extract handcrafted features from original augmented videos (M features).

**Input**: `data/augmented_videos/augmented_metadata.arrow`

**Process**:
1. Loads augmented video metadata
2. For each video:
   - Samples frames uniformly (default: 6 frames)
   - Extracts features:
     - Noise residual (5 features)
     - DCT statistics (3 features)
     - Blur/sharpness (2 features)
     - Boundary inconsistency (2 features)
     - Codec cues (3 features)
   - Saves features as `.npy` file
   - Writes metadata row to CSV/Arrow
3. Total: ~15 features per video (M features)

**Output**:
- `data/features_stage2/features_metadata.arrow` (or `.parquet`/`.csv`)
- `data/features_stage2/*.npy`: Feature files (one per video)

**Usage**:
```bash
python src/scripts/run_stage2_features.py --num-frames 6
```

---

### Step 3: Video Scaling

**Script**: `src/scripts/run_stage3_scaling.py` or `src/run_new_pipeline.py --only-stage 3`

**Function**: `lib.scaling.pipeline.stage3_scale_videos()`

**Purpose**: Scale all videos to target max dimension (can downscale or upscale).

**Input**: `data/augmented_videos/augmented_metadata.arrow`

**Process**:
1. Loads augmented video metadata
2. For each video:
   - Loads frames using PyAV
   - Determines scaling direction:
     - If max(width, height) > target_size: downscale
     - If max(width, height) < target_size: upscale
   - Applies scaling method:
     - `resolution`: Letterbox resize (bilinear interpolation)
     - `autoencoder`: Hugging Face VAE (optional, higher quality)
   - Preserves aspect ratio
   - Saves scaled video as MP4
   - Writes metadata row with original and scaled dimensions
3. Processes videos in chunks to prevent OOM

**Output**:
- `data/scaled_videos/scaled_metadata.arrow` (or `.parquet`/`.csv`)
- `data/scaled_videos/*.mp4`: Scaled video files (max dimension = target_size, default: 256px)
- Metadata includes: `original_width`, `original_height`, `is_upscaled`, `is_downscaled`

**Usage**:
```bash
python src/scripts/run_stage3_scaling.py --target-size 256 --method resolution
```

---

### Step 4: Scaled Video Feature Extraction

**Script**: `src/scripts/run_stage4_scaled_features.py` or `src/run_new_pipeline.py --only-stage 4`

**Function**: `lib.features.scaled.stage4_extract_scaled_features()`

**Purpose**: Extract additional features from scaled videos (P features).

**Input**: `data/scaled_videos/scaled_metadata.arrow`

**Process**:
1. Loads scaled video metadata
2. For each scaled video:
   - Samples frames uniformly (default: 6 frames)
   - Extracts all features:
     - Base handcrafted features (15 features, same as Stage 2)
     - Scaled-specific features (6 features):
       - Edge preservation metrics
       - Texture uniformity
       - Compression artifact visibility
       - Color consistency
   - Adds scaling indicators:
     - `is_upscaled`: Binary (1 if upscaled, 0 otherwise)
     - `is_downscaled`: Binary (1 if downscaled, 0 otherwise)
   - Saves features as `.npy` file
   - Writes metadata row to CSV/Arrow
3. Total: ~23 features per video (P features)

**Output**:
- `data/features_stage4/features_scaled_metadata.arrow` (or `.parquet`/`.csv`)
- `data/features_stage4/*.npy`: Scaled feature files (one per video)

**Usage**:
```bash
python src/scripts/run_stage4_scaled_features.py --num-frames 6
```

---

### Step 5: Model Training

**Script**: `src/scripts/run_stage5_training.py` or `src/run_new_pipeline.py --only-stage 5`

**Function**: `lib.training.pipeline.stage5_train_models()`

**Purpose**: Train models using scaled videos and extracted features.

**Input**:
- `data/scaled_videos/scaled_metadata.arrow` (Stage 3)
- `data/features_stage2/features_metadata.arrow` (Stage 2)
- `data/features_stage4/features_scaled_metadata.arrow` (Stage 4)

**Process**:
1. Validates prerequisites (all stages must be complete)
2. Loads metadata and features
3. Filters corrupted videos and videos with no frames
4. For each model type:
   - Creates 5-fold stratified cross-validation splits
   - For each fold:
     - Splits data into train/validation (group-aware to prevent data leakage)
     - Trains model:
       - Baseline models (sklearn): Use Stage 2 + Stage 4 features
       - XGBoost models: Extract features from pretrained models, then train XGBoost
       - PyTorch models: Train on scaled videos directly
     - Evaluates on validation set
     - Saves model checkpoint and metrics
   - Aggregates results across folds
   - Generates plots and visualizations
5. Optionally trains ensemble model

**Output**:
- `data/stage5/<model_type>/fold_*/model.pt` (or `.joblib`): Model checkpoints
- `data/stage5/<model_type>/fold_*/metrics.json`: Per-fold metrics
- `data/stage5/<model_type>/metrics.json`: Aggregated metrics
- `data/stage5/<model_type>/plots/`: Visualization plots

**Available Models**:
- Baseline: `logistic_regression`, `svm`
- Feature-based: `logistic_regression_stage2`, `svm_stage2`, `logistic_regression_stage2_stage4`, `svm_stage2_stage4`
- XGBoost: `xgboost_pretrained_inception`, `xgboost_i3d`, `xgboost_r2plus1d`, `xgboost_vit_gru`, `xgboost_vit_transformer`
- PyTorch: `naive_cnn`, `pretrained_inception`, `variable_ar_cnn`, `vit_gru`, `vit_transformer`, `timesformer`, `vivit`, `i3d`, `r2plus1d`, `x3d`, `slowfast`, `slowfast_attention`, `slowfast_multiscale`, `two_stream`

**Usage**:
```bash
python src/scripts/run_stage5_training.py --model-types x3d slowfast --n-splits 5
```

---

### Step 6: Notebook Generation

**Script**: `src/notebooks/generate_model_notebooks.py`

**Purpose**: Generate presentation-ready Jupyter notebooks for all models.

**Input**: Model configurations defined in `MODEL_CONFIGS` dictionary

**Process**:
1. For each model (5c-5u):
   - Generates notebook structure:
     - Title and description
     - Setup cell (imports, project root)
     - Architecture deep-dive
     - Model checkpoint verification
     - Training code (commented)
     - Hyperparameter configuration
     - MLOps integration (MLflow, DuckDB, Airflow)
     - Training methodology
     - Feature engineering (for XGBoost models)
     - Video demonstration
     - Model inference example
     - Results visualization
     - Conclusion
   - Saves notebook as `src/notebooks/{model_id}_{model_type}.ipynb`

**Output**:
- `src/notebooks/5c_naive_cnn.ipynb` through `src/notebooks/5u_two_stream.ipynb`
- `src/notebooks/00_MASTER_PIPELINE_JOURNEY.ipynb` (manually created, not generated)

**Usage**:
```bash
python src/notebooks/generate_model_notebooks.py
```

---

## Complete Pipeline Execution

### Option 1: Full Pipeline (All Stages)

```bash
python src/run_new_pipeline.py
```

This runs all 5 stages sequentially:
1. Stage 1: Augmentation
2. Stage 2: Feature extraction
3. Stage 3: Scaling
4. Stage 4: Scaled feature extraction
5. Stage 5: Training

### Option 2: Individual Stages

```bash
# Stage 1
python src/scripts/run_stage1_augmentation.py

# Stage 2
python src/scripts/run_stage2_features.py

# Stage 3
python src/scripts/run_stage3_scaling.py

# Stage 4
python src/scripts/run_stage4_scaled_features.py

# Stage 5
python src/scripts/run_stage5_training.py --model-types x3d slowfast

# Notebook Generation
python src/notebooks/generate_model_notebooks.py
```

### Option 3: SLURM Cluster Execution

```bash
# Submit each stage as a SLURM job
sbatch scripts/slurm_jobs/slurm_stage1_augmentation.sh
sbatch scripts/slurm_jobs/slurm_stage2_features.sh
sbatch scripts/slurm_jobs/slurm_stage3_scaling.sh
sbatch scripts/slurm_jobs/slurm_stage4_scaled_features.sh
sbatch scripts/slurm_jobs/slurm_stage5_training.sh
```

---

## Data Flow Diagram

```
Archive (ZIP files)
    │
    ▼
[Step 0: Setup]
    │
    ▼
data/video_index_input.csv (N videos)
    │
    ▼
[Step 1: Augmentation]
    │
    ▼
data/augmented_videos/augmented_metadata.arrow (11N videos)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
[Step 2: Features]                  [Step 3: Scaling]
    │                                     │
    ▼                                     ▼
data/features_stage2/              data/scaled_videos/
features_metadata.arrow            scaled_metadata.arrow
(M features)                       (11N scaled videos)
    │                                     │
    │                                     ▼
    │                              [Step 4: Scaled Features]
    │                                     │
    │                                     ▼
    │                              data/features_stage4/
    │                              features_scaled_metadata.arrow
    │                              (P features)
    │                                     │
    └─────────────────────────────────────┘
                    │
                    ▼
            [Step 5: Training]
                    │
                    ▼
            data/stage5/<model_type>/
            (Trained models + metrics)
                    │
                    ▼
            [Step 6: Notebook Generation]
                    │
                    ▼
            src/notebooks/*.ipynb
```

---

## Key Design Decisions

### Memory Optimization
- **Frame-by-frame decoding**: Decode only needed frames (50x memory reduction)
- **Incremental CSV writing**: Write metadata directly to avoid accumulation
- **Chunked processing**: Process videos in chunks to prevent OOM
- **Ultra-small batch sizes**: Model-specific batch sizes (1-8) for memory-intensive models
- **Gradient checkpointing**: For X3D models to trade computation for memory

### Reproducibility
- **Deterministic seeding**: Fixed random seeds for all operations
- **Pre-generated augmentations**: Store augmentations on disk for reproducibility
- **Fixed splits**: 5-fold stratified CV with fixed seed

### Data Integrity
- **Group-aware splitting**: Prevents data leakage via duplicate groups
- **Corruption detection**: Filters corrupted videos before training
- **Metadata validation**: Validates all metadata files before use

### MLOps Integration
- **MLflow tracking**: Experiment tracking, model registry, artifact management
- **DuckDB analytics**: Fast SQL queries on training results
- **Airflow orchestration**: Pipeline orchestration with dependency management

---

## File Locations

### Input Files
- Archives: `archive/FVC1.zip`, `archive/FVC2.zip`, `archive/FVC3.zip`, `archive/Metadata.zip`
- Metadata: `archive/FVC_dup.csv`, `archive/FVC.csv`

### Intermediate Files
- Stage 1: `data/augmented_videos/augmented_metadata.arrow`
- Stage 2: `data/features_stage2/features_metadata.arrow`
- Stage 3: `data/scaled_videos/scaled_metadata.arrow`
- Stage 4: `data/features_stage4/features_scaled_metadata.arrow`

### Output Files
- Stage 5: `data/stage5/<model_type>/fold_*/model.pt`, `data/stage5/<model_type>/metrics.json`
- Notebooks: `src/notebooks/*.ipynb`

---

## Troubleshooting

### Stage 1 fails: "Metadata file not found"
- Run `python src/setup_fvc_dataset.py` first

### Stage 2/3/4/5 fails: "Metadata file not found"
- Ensure previous stages completed successfully
- Check metadata files exist in expected locations

### Out of Memory (OOM) errors
- Reduce `--num-augmentations` in Stage 1
- Reduce `--num-frames` in Stages 2 and 4
- Reduce `--target-size` in Stage 3
- Use smaller batch sizes (configured in model factory)

### Notebooks not generated
- Ensure Stage 5 training completed
- Run `python src/notebooks/generate_model_notebooks.py`

---

## Next Steps

After completing all stages:
1. Review training results in `data/stage5/`
2. Compare model performance across architectures
3. View notebooks in `src/notebooks/`
4. Use best models for inference
5. Generate paper figures: `python src/generate_paper_figures.py`
6. View dashboard: `streamlit run src/dashboard_results.py`
