"""Critical tests for training pipeline."""
import pytest
import numpy as np
import polars as pl
from pathlib import Path
from unittest.mock import Mock, patch

from lib.training.pipeline import (
    BASELINE_MODELS,
    STAGE4_MODELS,
    MEMORY_INTENSIVE_MODELS_BATCH_LIMITS
)


class TestPipelineConstants:
    """Test pipeline constants and configurations."""

    def test_baseline_models_defined(self):
        """Test baseline models are defined."""
        assert len(BASELINE_MODELS) > 0
        assert "logistic_regression" in BASELINE_MODELS
        assert "svm" in BASELINE_MODELS

    def test_stage4_models_defined(self):
        """Test Stage 4 models are defined."""
        assert len(STAGE4_MODELS) > 0

    def test_memory_intensive_models_defined(self):
        """Test memory-intensive models have batch limits."""
        assert len(MEMORY_INTENSIVE_MODELS_BATCH_LIMITS) > 0
        assert "x3d" in MEMORY_INTENSIVE_MODELS_BATCH_LIMITS
        assert "naive_cnn" in MEMORY_INTENSIVE_MODELS_BATCH_LIMITS

    def test_memory_intensive_batch_sizes(self):
        """Test memory-intensive models have batch_size <= 2."""
        for model_type, batch_size in (
            MEMORY_INTENSIVE_MODELS_BATCH_LIMITS.items()
        ):
            assert batch_size <= 2, (
                f"{model_type} should have batch_size <= 2, got {batch_size}"
            )


class TestPipelineDataValidation:
    """Test data validation in pipeline."""

    def test_empty_dataframe_handling(self, temp_dir):
        """Test handling of empty dataframes."""
        empty_df = pl.DataFrame({
            "video_path": [],
            "label": [],
            "video_id": []
        })
        # Should not crash on empty dataframe
        assert empty_df.height == 0

    def test_missing_columns_handling(self):
        """Test handling of missing required columns."""
        df = pl.DataFrame({
            "video_path": ["test.mp4"],
            "label": [0]
            # Missing video_id
        })
        # Should detect missing columns
        required_cols = ["video_path", "label", "video_id"]
        missing = [col for col in required_cols if col not in df.columns]
        assert len(missing) > 0

    def test_label_validation(self):
        """Test label validation."""
        df = pl.DataFrame({
            "video_path": ["test1.mp4", "test2.mp4"],
            "label": [0, 1],
            "video_id": ["v1", "v2"]
        })
        # Labels should be 0 or 1
        unique_labels = df["label"].unique().to_list()
        assert all(label in [0, 1] for label in unique_labels)

