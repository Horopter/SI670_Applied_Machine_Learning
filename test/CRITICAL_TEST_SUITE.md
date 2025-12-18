# Critical Test Suite

## Overview

This document describes the critical test suite for the FVC project,
focusing on the most important functionality that must work correctly.

## Test Files

### `test_critical_model_factory.py`
Tests model creation and configuration:
- ✅ Model creation (logistic regression, SVM)
- ✅ PyTorch/XGBoost model detection
- ✅ Model configuration retrieval
- ✅ Invalid model type handling
- ✅ All model types have configurations

**Status**: 7/7 tests passing

### `test_critical_metrics.py`
Tests metrics computation:
- ✅ Perfect predictions
- ✅ Worst predictions
- ✅ Balanced predictions
- ✅ All zeros/ones edge cases
- ✅ Missing probabilities
- ✅ Per-class metrics
- ✅ Empty inputs

**Status**: 9/9 tests passing

### `test_critical_pipeline.py`
Tests training pipeline constants and validation:
- ✅ Baseline models defined
- ✅ Stage 4 models defined
- ✅ Memory-intensive models have batch limits
- ✅ Empty dataframe handling
- ✅ Missing columns handling
- ✅ Label validation

**Status**: 6/6 tests passing

### `test_critical_data_loading.py`
Tests data loading and splitting:
- ✅ CSV metadata loading
- ✅ JSON metadata loading
- ✅ Stratified k-fold basic functionality
- ✅ Label distribution preservation
- ✅ No train/val overlap
- ✅ Duplicate group handling

**Status**: 6/6 tests passing

### `test_critical_mlops.py`
Tests MLOps integration:
- ✅ ExperimentTracker initialization
- ✅ Metric logging
- ✅ Epoch metrics logging
- ✅ CheckpointManager initialization
- ✅ Checkpoint saving
- ✅ Checkpoint loading

**Status**: 6/6 tests passing

## Total Status

**34/34 critical tests passing** ✅

## Running Tests

```bash
# Run all critical tests
pytest test/test_critical_*.py -v

# Run specific test file
pytest test/test_critical_metrics.py -v

# Run with coverage
pytest test/test_critical_*.py --cov=lib --cov-report=term-missing
```

## Next Steps

1. Expand test coverage for:
   - Feature extraction pipelines
   - Video preprocessing
   - Augmentation transforms
   - Training loop edge cases
   - Memory management
   - Error handling

2. Add integration tests for:
   - End-to-end training pipeline
   - MLflow integration
   - Model inference
   - Data pipeline stages

3. Add performance tests for:
   - Training speed benchmarks
   - Memory usage validation
   - Batch processing efficiency

