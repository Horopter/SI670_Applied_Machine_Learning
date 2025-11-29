# Changelog

All notable changes to the FVC Binary Video Classifier project are documented in this file.

## [Unreleased]

### Added - Latest Memory Optimizations (2025-11-29)
- **Frame-by-frame video decoding**: Decode only the 6 needed frames instead of loading entire videos (50x memory reduction per video)
- **Incremental CSV writing**: Write augmented metadata directly to CSV to avoid memory accumulation
- **One video at a time processing**: Process videos sequentially with aggressive cleanup after each
- **Ultra-conservative batch sizes**: Model-specific batch sizes (1-8) with increased gradient accumulation (8-16 steps)
- **Reduced resolution**: Default `fixed_size=112` (was 224) for 4x memory reduction per frame
- **Reduced frame count**: `num_frames=6` (was 8) for additional memory savings
- **Reduced augmentations**: `num_augmentations_per_video=1` (was 3) to minimize augmentation memory
- **Comprehensive memory profiling**: Detailed memory tracking with object breakdowns to identify hotspots
- **Zero workers**: `num_workers=0` to eliminate multiprocessing memory overhead

### Added
- Comprehensive project documentation (`docs/PROJECT_OVERVIEW.md`)
- MLOps infrastructure for experiment tracking and versioning
- K-fold cross-validation pipeline
- Pre-generated data augmentation pipeline
- Aggressive garbage collection and OOM handling
- Fixed-size video preprocessing with letterboxing
- Balanced batch sampling for class imbalance
- Comprehensive video augmentations (spatial + temporal)
- Path resolution utilities for consistent file handling
- SLURM batch script for cluster execution
- MLOps pipeline runner script

### Changed
- Switched from variable aspect ratio to fixed-size preprocessing (224x224 → 112x112)
- Moved from on-the-fly to pre-generated augmentations
- **Reduced batch sizes**: Ultra-conservative model-specific batch sizes (1-8) instead of 32
- **Reduced frame count**: From 16 → 8 → 6 frames per video for memory efficiency
- **Reduced resolution**: From 224×224 → 112×112 for 4x memory reduction
- **Reduced augmentations**: From 3 → 1 augmentation per video
- Enhanced error handling and logging throughout
- Improved memory management with aggressive GC
- **Video loading**: Changed from full video loading to frame-by-frame decoding

### Fixed
- **OOM errors**: Resolved through frame-by-frame decoding, incremental CSV writing, and ultra-conservative resource usage
- **Memory spikes during augmentation**: Eliminated by decoding only needed frames instead of loading entire videos
- **Memory accumulation**: Fixed by writing metadata incrementally instead of accumulating in memory
- Overfitting through K-fold cross-validation
- Path resolution inconsistencies
- Missing video file handling
- Zip bomb detection issues during dataset setup
- Syntax errors (`=== None` → `is None`)
- Function signature mismatches
- Import errors
- Loss reporting issues (0.0000 loss)

### Removed
- Single-use debug scripts (`debug_unzip.sh`, `fix_incomplete_unzip.sh`, `diagnose_job.sh`)
- Unused dependencies (scikit-learn replaced by Polars)

## Initial Implementation

### Architecture
- Pretrained 3D ResNet (r3d_18) backbone with Kinetics-400 weights
- Custom Inception-like head for binary classification
- Variable aspect ratio support with adaptive pooling

### Data Pipeline
- Polars-based metadata loading
- Stratified train/val/test splits with dup_group awareness
- Uniform frame sampling
- Basic data augmentation (horizontal flip, color jitter)

### Training
- Adam optimizer with StepLR scheduler
- Mixed precision training (AMP)
- Early stopping
- Basic checkpointing

### Issues Encountered
1. **OOM Errors**: Large video resolutions causing GPU memory exhaustion
2. **Overfitting**: Loss dropping to 0.0000, identical logits
3. **Data Issues**: Missing files, path inconsistencies, partial unzip
4. **Reproducibility**: No experiment tracking or versioning

## Evolution

### Phase 1: Memory Optimization
- Implemented fixed-size preprocessing (224x224)
- Added aggressive garbage collection
- Optimized data loading (parallel workers, pin_memory, prefetching)
- Progressive batch size fallback strategy

### Phase 2: Robustness
- Added K-fold cross-validation
- Implemented comprehensive augmentations
- Enhanced error handling and OOM detection
- Added data validation at every stage

### Phase 3: MLOps
- Built experiment tracking infrastructure
- Implemented pipeline orchestration
- Added checkpoint management with resume capability
- Created data versioning system

### Phase 4: Code Quality
- Fixed syntax and import errors
- Standardized path resolution
- Enhanced logging and diagnostics
- Comprehensive code quality checks

