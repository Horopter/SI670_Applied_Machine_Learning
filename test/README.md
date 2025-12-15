# AURA Unit Tests

Comprehensive unit test suite for the FVC video classifier project.

## Structure

- `conftest.py`: Shared pytest fixtures and configuration
- `test_mlops_core.py`: Tests for MLOps core functionality (RunConfig, ExperimentTracker, CheckpointManager)
- `test_video_data.py`: Tests for data loading, splitting, and k-fold cross-validation
- `test_video_metrics.py`: Tests for metrics computation (accuracy, precision, recall, F1, ROC-AUC)
- `test_handcrafted_features.py`: Tests for handcrafted feature extraction
- `test_video_augmentations.py`: Tests for video augmentation transforms
- `pytest.ini`: Pytest configuration
- `run_tests.sh`: Test runner script

## Running Tests

### Basic Test Run
```bash
pytest test/ -v
```

### With Coverage
```bash
pytest test/ --cov=lib --cov-report=html --cov-report=term-missing
```

### Using Test Runner Script
```bash
./test/run_tests.sh
```

### Run Specific Test File
```bash
pytest test/test_video_data.py -v
```

### Run Specific Test Class
```bash
pytest test/test_video_data.py::TestStratifiedKfold -v
```

### Run Specific Test Function
```bash
pytest test/test_video_data.py::TestStratifiedKfold::test_basic_kfold -v
```

## Test Coverage

The test suite includes:

1. **Core MLOps Tests** (`test_mlops_core.py`):
   - RunConfig serialization/deserialization
   - ExperimentTracker metrics logging
   - CheckpointManager save/load/resume
   - DataVersionManager version tracking

2. **Data Tests** (`test_video_data.py`):
   - CSV loading and validation
   - Train/val/test splitting
   - K-fold cross-validation
   - Duplicate group handling
   - Test subset limiting

3. **Metrics Tests** (`test_video_metrics.py`):
   - Binary classification metrics
   - Multiclass metrics
   - Confusion matrix
   - ROC-AUC computation
   - Edge cases (perfect predictions, worst predictions)

4. **Feature Extraction Tests** (`test_handcrafted_features.py`):
   - Noise residual extraction
   - DCT statistics
   - Blur/sharpness detection
   - Boundary inconsistency
   - Codec cues

5. **Augmentation Tests** (`test_video_augmentations.py`):
   - Spatial augmentations (rotation, affine, color jitter)
   - Letterbox resize
   - Transform composition

## Stress Testing

All tests include stress testing scenarios:
- Large datasets (10,000+ samples)
- Small datasets (edge cases)
- Extreme values
- Empty inputs
- Imbalanced classes
- Missing data

## Fixtures

Common fixtures available in `conftest.py`:
- `temp_dir`: Temporary directory for test outputs
- `sample_video_df`: Sample video DataFrame
- `sample_video_df_with_dups`: Sample DataFrame with duplicate groups
- `sample_run_config`: Sample RunConfig
- `sample_metrics`: Sample metrics dictionary
- `sample_logits_labels`: Sample logits and labels for binary classification
- `sample_logits_labels_multiclass`: Sample logits and labels for multiclass
- `mock_video_tensor`: Mock video tensor
- `sample_handcrafted_features`: Sample handcrafted features

## Adding New Tests

When adding new tests:

1. Create a new file `test_<module_name>.py`
2. Import the module functions/classes to test
3. Create test classes following the pattern `Test<ClassName>`
4. Use descriptive test function names: `test_<what_it_tests>`
5. Include stress tests for edge cases
6. Use fixtures from `conftest.py` when possible

Example:
```python
"""Tests for my_module."""
import pytest
from lib.my_module import my_function

class TestMyFunction:
    """Tests for my_function."""
    
    def test_basic_usage(self):
        """Test basic function usage."""
        result = my_function(input_data)
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case."""
        result = my_function(edge_case_input)
        assert result == expected_value
```

## Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Notes

- Tests use deterministic random seeds for reproducibility
- All tests should be independent (no shared state)
- Use temporary directories for file I/O tests
- Clean up resources in fixtures (automatic with `yield`)

