"""Pytest configuration and shared fixtures for unit tests."""
import pytest
import tempfile
import shutil
import os
from pathlib import Path
import numpy as np
import polars as pl
import torch
from typing import Dict, Any

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_video_df():
    """Create a sample video DataFrame for testing."""
    data = {
        "video_path": [
            "FVC1/twitter/video1.mp4",
            "FVC1/youtube/video2.mp4",
            "FVC2/twitter/video3.mp4",
            "FVC2/youtube/video4.mp4",
            "FVC3/twitter/video5.mp4",
        ],
        "label": [0, 1, 0, 1, 0],
        "video_id": ["vid1", "vid2", "vid3", "vid4", "vid5"],
        "subset": ["train", "train", "val", "val", "test"],
        "platform": ["twitter", "youtube", "twitter", "youtube", "twitter"],
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_video_df_with_dups():
    """Create a sample video DataFrame with duplicate groups."""
    data = {
        "video_path": [
            "FVC1/twitter/video1.mp4",
            "FVC1/twitter/video1_dup.mp4",
            "FVC1/youtube/video2.mp4",
            "FVC2/twitter/video3.mp4",
            "FVC2/youtube/video4.mp4",
        ],
        "label": [0, 0, 1, 0, 1],
        "video_id": ["vid1", "vid1_dup", "vid2", "vid3", "vid4"],
        "dup_group": [1, 1, 2, 3, 4],
        "subset": ["train", "train", "train", "val", "val"],
        "platform": ["twitter", "twitter", "youtube", "twitter", "youtube"],
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_run_config():
    """Create a sample RunConfig for testing."""
    from lib.mlops_core import RunConfig
    
    return RunConfig(
        run_id="test_run_123",
        experiment_name="test_experiment",
        description="Test run",
        data_csv="data/video_index_input.csv",
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
        random_seed=42,
        num_frames=8,
        fixed_size=224,
        batch_size=4,
        num_epochs=2,
        learning_rate=1e-4,
        project_root="/tmp/test",
        output_dir="/tmp/test/output",
    )


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary."""
    return {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "loss": 0.45,
        "val_accuracy": 0.83,
        "val_loss": 0.47,
    }


@pytest.fixture
def sample_logits_labels():
    """Create sample logits and labels for testing."""
    # Binary classification: 10 samples
    logits = torch.randn(10)
    labels = torch.randint(0, 2, (10,))
    return logits, labels


@pytest.fixture
def sample_logits_labels_multiclass():
    """Create sample logits and labels for multiclass testing."""
    # 3 classes, 10 samples
    logits = torch.randn(10, 3)
    labels = torch.randint(0, 3, (10,))
    return logits, labels


@pytest.fixture
def mock_video_tensor():
    """Create a mock video tensor for testing."""
    # Shape: (batch, channels, frames, height, width)
    return torch.randn(2, 3, 8, 224, 224)


@pytest.fixture
def sample_handcrafted_features():
    """Create sample handcrafted features."""
    return {
        "noise_residual_energy": np.random.rand(10),
        "dct_band_stats": np.random.rand(20),
        "blur_sharpness": np.random.rand(5),
        "boundary_inconsistency": np.random.rand(8),
        "codec_cues": np.random.rand(12),
    }

