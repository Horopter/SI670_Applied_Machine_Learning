# Changelog

All notable changes to the FVC Binary Video Classifier project are documented in this file.

## [Unreleased]

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
- Switched from variable aspect ratio to fixed-size preprocessing (224x224)
- Moved from on-the-fly to pre-generated augmentations
- Increased default batch size from 2-4 to 32 (with fallbacks)
- Increased frame count from 8 to 16 frames per video
- Enhanced error handling and logging throughout
- Improved memory management with aggressive GC

### Fixed
- OOM errors through fixed-size preprocessing and aggressive GC
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

