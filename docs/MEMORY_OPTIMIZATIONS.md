# Memory Optimizations and Stability Improvements

## Overview

This document describes the memory optimizations and stability improvements implemented to address OOM (Out of Memory) errors and ensure reproducible augmentations across runs and k-fold splits.

## Memory Optimizations

### 1. Reduced Resource Usage

**Batch Size Reduction**:
- **Before**: `batch_size=32` (16 real + 16 fake per batch)
- **After**: Ultra-conservative batch sizes per model (1-8 depending on model)
- **Impact**: ~4-32x reduction in memory per batch
- **Compensation**: Increased `gradient_accumulation_steps` (8-16) to maintain effective batch size

**Number of Workers**:
- **Before**: `num_workers=4`
- **After**: `num_workers=0` (CPU-only or test mode to avoid multiprocessing memory overhead)
- **Impact**: Eliminates memory overhead from parallel data loading workers

**Frame Count**:
- **Before**: `num_frames=16`
- **After**: `num_frames=6` (ultra-conservative)
- **Impact**: ~2.7x reduction in memory per video sample

**Resolution**:
- **Before**: `fixed_size=224` (224×224 pixels)
- **After**: `fixed_size=112` (112×112 pixels, configurable via `FVC_FIXED_SIZE`)
- **Impact**: ~4x reduction in memory per frame (224² → 112²)

### 2. Frame-by-Frame Video Decoding (CRITICAL)

**Problem**: Loading entire videos into memory causes massive memory spikes
- Large video (1920×1080, 30fps, 10s = 300 frames): ~1.87 GB per video
- With base memory (~31GB), loading one large video can spike past 80GB → OOM

**Solution**: Decode only the frames we need using PyAV
- **Before**: Load entire video → extract 6 frames → delete video
- **After**: Seek to specific frames → decode only those 6 frames → never load full video
- **Memory per video**: ~37 MB (6 frames) instead of ~1.87 GB (300 frames)
- **Memory reduction**: ~50x reduction per video

**Implementation**:
```python
# Get frame count without loading video
container = av.open(video_path)
stream = container.streams.video[0]
total_frames = stream.frames
container.close()

# Decode only selected frames
for frame_idx in sorted(indices):
    container.seek(timestamp_pts, stream=stream)
    for packet in container.demux(stream):
        for frame in packet.decode():
            # Process only this frame
            frame_array = frame.to_ndarray(format='rgb24')
            # ... apply transforms ...
            break
```

**Fallback**: If frame-by-frame decoding fails, falls back to full video loading with warning

**Benefits**:
- Prevents OOM during augmentation generation
- Allows processing large videos without memory spikes
- Stable memory usage (~1-2 GB instead of 30+ GB)

### 3. Incremental CSV Writing for Metadata

**Problem**: Accumulating all augmented clip metadata in memory
- 298 videos × 1 augmentation = 298 metadata rows
- Each row: video_path, label, original_video, augmentation_idx
- Memory accumulation as list grows

**Solution**: Write metadata directly to CSV incrementally
- **Before**: `augmented_rows.append({...})` → `pl.DataFrame(augmented_rows)` at end
- **After**: Write each row immediately to CSV file
- **Memory**: Constant (no accumulation)

**Implementation**:
```python
# Initialize CSV with header
with open(metadata_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "label", "original_video", "augmentation_idx"])

# Write each row immediately
for video in videos:
    # ... process video ...
    with open(metadata_path, 'a') as f:
        writer.writerow([clip_path_rel, label, video_rel, aug_idx])
    
# Load final DataFrame from CSV at end
augmented_df = pl.read_csv(metadata_path)
```

**Benefits**:
- Eliminates unbounded memory growth from metadata list
- Memory stays constant regardless of dataset size

### 4. One Video at a Time Processing

**Implementation**:
- Process videos one at a time (`batch_size=1`) during augmentation generation
- Aggressive GC after each video
- Clear video tensors and clips from memory immediately after processing
- Delete frames after stacking into clip
- Delete clip after saving to disk

**Benefits**:
- Prevents memory accumulation during augmentation generation
- Allows processing large datasets without OOM
- Minimal peak memory usage

### 5. Shared Augmentations Across K-Fold

**Before**: Each fold generated its own augmentations
- Memory: 5x augmentation generation (one per fold)
- Disk: 5x storage space
- Time: 5x generation time

**After**: Single shared augmentation generation
- Generate augmentations once for all unique videos
- Filter augmentations per fold based on training videos
- Memory: 1x generation
- Disk: 1x storage (shared across folds)
- Time: 1x generation time

**Implementation**:
```python
# Generate augmentations once for all videos
shared_aug_dir = "augmented_clips/shared"
shared_aug_df = pregenerate_augmented_dataset(all_train_videos, ...)

# Filter per fold
for fold_idx, (train_df, val_df) in enumerate(folds):
    aug_df = shared_aug_df.filter(
        pl.col("original_video").is_in(train_df["video_path"])
    )
```

### 6. Aggressive Garbage Collection

**Enhanced GC Strategy**:
- 3 passes of `gc.collect()` instead of 1
- CUDA cache clearing after every batch
- CUDA synchronization to ensure cleanup
- GC after every pipeline stage
- GC after every epoch
- GC after every k-fold fold

**Implementation**:
```python
def aggressive_gc(clear_cuda: bool = True):
    for _ in range(3):
        gc.collect()
    if clear_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

## Augmentation Stability

### 1. Deterministic Seeds

**Problem**: Augmentations were random, causing:
- Different augmentations across runs
- Different augmentations per fold for same video
- Non-reproducible results

**Solution**: Deterministic seed generation from video path
```python
# Generate seed from video path hash
import hashlib
video_path_str = str(video_path)
seed = int(hashlib.md5(video_path_str.encode()).hexdigest()[:8], 16) % (2**31)

# Use seed for each augmentation
for aug_idx in range(num_augmentations):
    aug_seed = seed + aug_idx
    random.seed(aug_seed)
    np.random.seed(aug_seed)
    torch.manual_seed(aug_seed)
    # Generate augmentation...
```

**Benefits**:
- Same video → same augmentations across runs
- Same video → same augmentations across folds
- Reproducible results
- Can cache augmentations safely

### 2. Shared Augmentation Storage

**Directory Structure**:
```
runs/
  run_XXX/
    augmented_clips/
      shared/              # Shared across all folds
        video1_aug0.pt
        video1_aug1.pt
        video1_aug2.pt
        ...
        augmented_metadata.csv
```

**Usage**:
- Generate once in `shared/` directory
- All folds reference same augmentations
- Filter by fold's training videos

## Cleanup Before Runs

### 1. Automatic Cleanup

**Implementation**:
```python
from lib.cleanup_utils import cleanup_runs_and_logs

# Before starting pipeline
cleanup_runs_and_logs(project_root, keep_models=False)
```

**What Gets Deleted**:
- `runs/` directory (all previous experiment outputs)
- `logs/` directory (all previous log files)
- `models/` directory (all previous model checkpoints)

**What Gets Preserved**:
- `archive/` folder (original dataset archives)
- `data/video_index_input.csv` (metadata)
- `videos/` folder (extracted video files)

### 2. Benefits

- Prevents disk space accumulation
- Ensures clean start for each run
- Avoids confusion from old results
- Reduces risk of loading wrong checkpoints

## Memory Usage Estimates

### Before All Optimizations
- Batch size 32: ~2-3 GB per batch
- 16 frames: ~1.5 GB per batch
- 4 workers: ~500 MB overhead
- Full video loading: ~1.87 GB per video during augmentation
- **Total**: ~4-5 GB per batch + model + overhead + video loading = **~80 GB** (OOM)

### After Initial Optimizations
- Batch size 8: ~0.5-0.75 GB per batch
- 8 frames: ~0.4 GB per batch
- 2 workers: ~200 MB overhead
- Full video loading: ~1.87 GB per video during augmentation
- **Total**: ~1-1.5 GB per batch + model + overhead + video loading = **~30-40 GB** (still OOM)

### After Latest Optimizations (Current)
- Batch size 1-8 (model-dependent): ~0.1-0.75 GB per batch
- 6 frames: ~0.3 GB per batch
- 0 workers: 0 MB overhead
- **Frame-by-frame decoding**: ~37 MB per video (only 6 frames loaded)
- **Incremental CSV writing**: Constant memory (no accumulation)
- **One video at a time**: Minimal peak memory
- **Total**: ~1-2 GB per batch + model + overhead = **~5-10 GB** (well within limits)

### Effective Batch Size
- Actual batch: 1-8 (model-dependent)
- Gradient accumulation: 8-16 steps (compensates for smaller batches)
- **Effective batch**: 8-16 (maintains training effectiveness)

## Configuration Changes

### `src/run_mlops_pipeline.py`

```python
config = RunConfig(
    # Ultra-conservative for memory efficiency
    num_frames=6,           # Was 16 (reduced to 6)
    fixed_size=112,         # Was 224 (reduced to 112, configurable via FVC_FIXED_SIZE)
    num_augmentations_per_video=1,  # Was 3 (reduced to 1)
    # Batch sizes are model-specific (see MODEL_MEMORY_CONFIGS in model_factory.py)
    # num_workers=0 (CPU-only or test mode)
    # gradient_accumulation_steps=8-16 (compensates for smaller batches)
    ...
)
```

### `lib/model_factory.py` - MODEL_MEMORY_CONFIGS

Ultra-conservative batch sizes per model:
- `logistic_regression`: batch_size=8
- `svm`: batch_size=8
- `naive_cnn`: batch_size=4
- `vit_gru`: batch_size=1, gradient_accumulation_steps=16
- `vit_transformer`: batch_size=1, gradient_accumulation_steps=16
- `slowfast`: batch_size=1, num_frames=6, gradient_accumulation_steps=16
- `x3d`: batch_size=1, num_frames=6, gradient_accumulation_steps=16

## Testing Recommendations

1. **Monitor Memory Usage**:
   ```bash
   # On SLURM cluster
   sacct -j <JOBID> --format=MaxRSS,MaxVMSize
   ```

2. **Check Logs for OOM**:
   ```bash
   grep -i "out of memory" logs/*.log
   ```

3. **Verify Augmentation Stability**:
   - Run same video through augmentation twice
   - Check that augmentations are identical
   - Verify same augmentations across folds

4. **Monitor Disk Space**:
   ```bash
   du -sh runs/ logs/ models/
   ```

## Memory Profiling

### Detailed Memory Tracking

Added comprehensive memory profiling to identify hotspots:
- CPU memory (RSS and VMS)
- GPU memory (allocated, reserved, free)
- Top memory consumers by object type
- Polars DataFrame count
- PyTorch tensor count and total memory
- PyTorch model count

**Usage**:
```python
from lib.mlops_utils import log_memory_stats

# Basic memory stats
log_memory_stats("after loading data")

# Detailed breakdown
log_memory_stats("after augmentation batch 5", detailed=True)
```

**Output**:
- Memory stats logged at key points in pipeline
- Detailed breakdowns show what's consuming memory
- Helps identify memory leaks and hotspots

## Future Optimizations

1. **Further Reduce Frame Count**: If still OOM, reduce to `num_frames=4`
2. **Further Reduce Resolution**: If still OOM, reduce to `fixed_size=96` or `64`
3. **Mixed Precision**: Already enabled, but can be more aggressive
4. **Gradient Checkpointing**: Trade compute for memory (if needed)
5. **Model Pruning**: Reduce model size if needed
6. **Streaming Video Processing**: Process videos in chunks instead of loading all frames

## Summary

These optimizations reduce memory usage by approximately **8-16x** while maintaining:
- Training effectiveness (via increased gradient accumulation)
- Augmentation quality (same augmentations, just fewer frames and lower resolution)
- Reproducibility (deterministic seeds)
- Efficiency (shared augmentations across folds)

**Key Breakthrough**: Frame-by-frame decoding eliminates the memory spike from loading entire videos, reducing per-video memory from ~1.87 GB to ~37 MB (50x reduction).

The pipeline should now run comfortably within 80 GB memory limits on the SLURM cluster, with typical usage around 5-10 GB during augmentation generation and 10-20 GB during training.

