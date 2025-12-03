# Reusable Components from Previous Pipeline

This document lists what we can borrow/reuse from the previous pipeline for the new 5-stage pipeline.

## Already Being Used

### Stage 1-4 (Preprocessing)
- ✅ `handcrafted_features.py` - Feature extraction (Stage 2 & 4)
- ✅ `video_paths.py` - Path resolution
- ✅ `video_data.py` - Metadata loading, filtering, stratified_kfold
- ✅ `video_modeling.py` - `_read_video_wrapper()`, `uniform_sample_indices()`
- ✅ `mlops_utils.py` - Memory management (`aggressive_gc`, `log_memory_stats`)

### Stage 5 (Training)
- ✅ `_linear.py` - LogisticRegressionBaseline
- ✅ `_svm.py` - SVMBaseline
- ✅ `_cnn.py` - NaiveCNNBaseline
- ✅ `_transformer_gru.py` - ViTGRUModel
- ✅ `_transformer.py` - ViTTransformerModel
- ✅ `slowfast.py` - SlowFastModel
- ✅ `x3d.py` - X3DModel
- ✅ `model_factory.py` - Model creation and configuration
- ✅ `trainer.py` - Training utilities

## Components to Integrate into Stage 5

### 1. Training Utilities (`lib/video_training.py`)

**`fit_with_tracking()`** - High-level training with experiment tracking
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

**`build_optimizer()` / `build_scheduler()`** - Optimizer/scheduler creation
- Adam optimizer with configurable parameters
- StepLR scheduler

**`EarlyStopping`** - Early stopping logic
- Patience-based early stopping
- Mode selection (max for accuracy, min for loss)

### 2. Metrics (`lib/video_metrics.py`)

**`collect_logits_and_labels()`** - Collect model predictions
- Runs model over DataLoader
- Handles both binary and multi-class outputs
- Returns logits and labels tensors

**`basic_classification_metrics()`** - Compute classification metrics
- Accuracy, precision, recall, F1
- Handles both binary and multi-class cases
- Configurable threshold

**`confusion_matrix()`** - Confusion matrix
- Computes TP, TN, FP, FN
- Returns as dictionary

**`roc_auc()`** - ROC AUC score
- Handles edge cases (single class, etc.)
- Returns NaN when undefined

### 3. MLOps Infrastructure (`lib/mlops_core.py`)

**`ExperimentTracker`** - Experiment tracking
- Logs metrics to JSONL
- Tracks best metrics
- Saves configuration
- Query metrics as DataFrame

**`CheckpointManager`** - Checkpoint management
- Saves best model checkpoints
- Full state saving (model, optimizer, scheduler, epoch)
- Resume capability

**`DataVersionManager`** - Data versioning
- Tracks data splits
- Tracks augmentation versions
- Links data to config hashes

**`RunConfig`** - Configuration management
- Centralized configuration
- Deterministic hashing
- JSON serialization

### 4. Model Architectures

**Frame→Temporal Models**
- `ViTGRUModel` (`lib/training/vit_gru.py`) - ViT + GRU
- `ViTTransformerModel` (`lib/training/vit_transformer.py`) - ViT + Transformer

**Spatiotemporal Models**
- `SlowFastModel` (`lib/training/slowfast.py`) - SlowFast network
- `X3DModel` (`lib/training/x3d.py`) - X3D network

**Baseline Models**
- `LogisticRegressionBaseline` (`lib/training/logistic_regression.py`)
- `SVMBaseline` (`lib/training/svm.py`)
- `NaiveCNNBaseline` (`lib/training/naive_cnn.py`)

### 5. Memory Management (`lib/mlops_utils.py`)

**`aggressive_gc()`** - Aggressive garbage collection
- Multiple GC passes
- CUDA cache clearing
- Threshold manipulation

**`log_memory_stats()`** - Memory profiling
- CPU memory (RSS, VMS)
- GPU memory (allocated, reserved, free)
- Detailed object breakdown
- Top memory consumers

**`safe_execute()`** - OOM handling
- Wraps functions with retry logic
- OOM error detection
- Automatic cleanup and retry

### 6. Data Utilities (`lib/video_data.py`)

**`stratified_kfold()`** - K-fold cross-validation
- Stratified splitting
- Balanced classes
- Handles duplicate groups

**`train_val_test_split()`** - Data splitting
- Stratified splitting
- Duplicate group awareness
- Configurable ratios

**`filter_existing_videos()`** - Video filtering
- Filters missing videos
- Optional frame count verification

## Integration Status

### ✅ Fully Integrated
- Training utilities (`fit_with_tracking`, `train_one_epoch`, `evaluate`)
- Metrics (`basic_classification_metrics`, `confusion_matrix`, `roc_auc`)
- Baseline models
- Model factory
- Memory management

### 🔄 Partially Integrated
- Experiment tracking (structure in place, needs full integration)
- Checkpoint management (structure in place, needs full integration)

### ❌ Not Yet Integrated
- `DataVersionManager` - Could track data versions across stages
- `RunConfig` - Could use for unified configuration
- Advanced model architectures (ViT, SlowFast, X3D) - Need adaptation for combined inputs

## Recommended Next Steps

1. **Full Experiment Tracking Integration**
   - Use `ExperimentTracker` for all stages
   - Log metrics for each stage
   - Track data versions

2. **Enhanced Checkpointing**
   - Checkpoint after each stage
   - Resume from any stage
   - Save intermediate results

3. **Model Architecture Adaptation**
   - Adapt ViT/GRU/Transformer models to accept combined inputs (video + features)
   - Create fusion layers for combining video and features

4. **Unified Configuration**
   - Use `RunConfig` for all stages
   - Single configuration file for entire pipeline

5. **Enhanced Error Handling**
   - Use `safe_execute()` for all stages
   - OOM handling throughout pipeline
   - Graceful degradation

## Example: Using fit_with_tracking in Stage 5

```python
from .video_training import fit_with_tracking, OptimConfig, TrainConfig
from .mlops_core import ExperimentTracker, CheckpointManager

# Create tracker and checkpoint manager
tracker = ExperimentTracker(str(fold_output_dir))
ckpt_manager = CheckpointManager(str(fold_output_dir))

# Configure training
optim_cfg = OptimConfig(lr=1e-4, weight_decay=1e-4)
train_cfg = TrainConfig(
    num_epochs=20,
    device=device,
    use_amp=True,
    gradient_accumulation_steps=16
)

# Train with tracking
trained_model = fit_with_tracking(
    model,
    train_loader,
    val_loader,
    optim_cfg,
    train_cfg,
    tracker,
    ckpt_manager
)
```

This provides:
- Full training loop
- Metrics logging
- Checkpointing
- Early stopping
- Resume capability

