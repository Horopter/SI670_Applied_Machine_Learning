# OOM Handling and K-Fold Cross-Validation Optimizations

## Overview

This document describes the aggressive memory management, OOM handling, and K-fold cross-validation optimizations implemented to ensure robust training.

## Key Features

### 1. Aggressive Garbage Collection

**Implementation:**
- `aggressive_gc()`: Performs 3 passes of garbage collection + CUDA cache clearing
- Called after every pipeline stage
- Called every 5 batches during training
- Called after each epoch
- Called after each K-fold fold

**Benefits:**
- Prevents memory accumulation
- Reduces OOM risk
- Frees GPU memory proactively

### 2. OOM Error Detection and Handling

**Implementation:**
- `check_oom_error()`: Detects OOM errors from multiple indicators
- `handle_oom_error()`: Performs aggressive cleanup on OOM
- `safe_execute()`: Wraps functions with OOM retry logic

**OOM Indicators Detected:**
- "out of memory"
- "cuda out of memory"
- "oom"
- "memory allocation failed"
- "allocation failed"

**Recovery Strategy:**
- Automatic retry after aggressive cleanup
- Detailed memory logging
- Graceful degradation (skip batch if needed)

### 3. Per-Stage Checkpointing

**Implementation:**
- Every pipeline stage saves a checkpoint
- Checkpoints include stage results and completion status
- Automatic resume from last completed stage

**Stages Checkpointed:**
1. `load_data`: Metadata DataFrame
2. `create_kfold_splits`: Fold splits
3. `generate_augmentations`: Augmentation metadata
4. `create_datasets`: Dataset configurations
5. `create_loaders`: DataLoader configurations
6. `initialize_model`: Model architecture
7. `train_kfold_models`: Training results per fold

**Benefits:**
- Resume from any stage after failure
- No need to recompute expensive operations
- Full pipeline state preservation

### 4. Stratified K-Fold Cross-Validation

**Implementation:**
- `build_kfold_pipeline()`: Complete K-fold pipeline
- Stratified splitting ensures class balance
- 5-fold CV by default (configurable)
- Per-fold metrics tracking
- Average metrics across folds

**Benefits:**
- Prevents overfitting (model sees all data)
- Prevents underfitting (multiple train/val splits)
- More robust performance estimates
- Better generalization

### 5. Enhanced Training Loop

**Memory Management:**
- Aggressive GC every 5 batches
- CUDA cache clearing every 20 batches
- Memory stats logging at epoch boundaries
- OOM error detection and handling

**Checkpointing:**
- Full state checkpoints (model + optimizer + scheduler)
- Best model tracking
- Latest checkpoint for resume
- Per-epoch checkpointing

## Usage

The SLURM script (`run_fvc_training.sh`) now defaults to using the MLOps pipeline:

```bash
# Default: Uses MLOps pipeline with K-fold CV
sbatch src/scripts/run_fvc_training.sh

# To use notebook instead:
USE_MLOPS_PIPELINE=false sbatch src/scripts/run_fvc_training.sh
```

## Pipeline Flow

```
1. Load Data
   ↓ (checkpoint)
2. Create K-Fold Splits (5 folds)
   ↓ (checkpoint)
3. For each fold:
   a. Generate Augmentations
      ↓ (checkpoint)
   b. Create Datasets
      ↓ (checkpoint)
   c. Create Loaders
      ↓ (checkpoint)
   d. Initialize Model
      ↓ (checkpoint)
   e. Train Model (with aggressive GC, OOM handling)
      ↓ (checkpoint per epoch)
   f. Evaluate
   g. Aggressive cleanup
4. Compute Average Metrics
5. Save Results
```

## Memory Management Strategy

1. **Proactive GC**: Before expensive operations
2. **Reactive GC**: After OOM errors
3. **Periodic GC**: Every N batches/epochs
4. **Stage GC**: After each pipeline stage
5. **Fold GC**: After each K-fold fold

## OOM Recovery

1. **Detection**: Automatic OOM error detection
2. **Cleanup**: Aggressive GC + CUDA cache clear
3. **Retry**: Automatic retry (up to max_retries)
4. **Logging**: Detailed memory stats
5. **Fallback**: Skip batch or reduce batch size

## Checkpoint Structure

```
runs/
└── experiment_name/
    └── run_20240101_120000_abcd1234/
        ├── checkpoints/
        │   ├── stage_load_data.pt
        │   ├── stage_create_kfold_splits.pt
        │   ├── stage_generate_augmentations.pt
        │   ├── fold_1/
        │   │   ├── best_model.pt
        │   │   ├── latest_checkpoint.pt
        │   │   └── checkpoint_epoch_*.pt
        │   ├── fold_2/
        │   └── ...
        └── kfold_results.feather
```

## Benefits Summary

1. **Robustness**: Handles OOM errors gracefully
2. **Efficiency**: Aggressive GC prevents memory leaks
3. **Reliability**: Per-stage checkpointing enables resume
4. **Generalization**: K-fold CV prevents overfitting/underfitting
5. **Observability**: Detailed memory and metrics logging

## Configuration

All settings are in `RunConfig`:
- `n_splits`: Number of K-fold splits (default: 5)
- `batch_size`: Batch size (auto-reduced on OOM)
- `num_workers`: DataLoader workers
- `use_amp`: Mixed precision training

## Monitoring

Memory stats are logged:
- Before/after each stage
- At epoch boundaries
- After OOM errors
- After each fold

Metrics are tracked:
- Per-fold metrics
- Average metrics across folds
- Standard deviation across folds
- Best model per fold

