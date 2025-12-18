"""Critical tests for MLOps integration."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from lib.mlops.config import ExperimentTracker, CheckpointManager


class TestExperimentTracker:
    """Critical tests for ExperimentTracker."""

    def test_tracker_initialization(self, temp_dir):
        """Test tracker initialization."""
        tracker = ExperimentTracker(
            run_dir=str(Path(temp_dir) / "output"),
            run_id="test_run_123"
        )
        assert tracker.run_id == "test_run_123"
        assert tracker.run_dir.exists()

    def test_tracker_log_metrics(self, temp_dir):
        """Test logging metrics."""
        tracker = ExperimentTracker(
            run_dir=str(Path(temp_dir) / "output"),
            run_id="test_run_123"
        )
        tracker.log_metric(step=1, metric_name="accuracy", value=0.85)
        tracker.log_metric(step=1, metric_name="loss", value=0.5)
        # Should not raise exception
        assert True

    def test_tracker_log_epoch_metrics(self, temp_dir):
        """Test logging epoch metrics."""
        tracker = ExperimentTracker(
            run_dir=str(Path(temp_dir) / "output"),
            run_id="test_run_123"
        )
        metrics = {"accuracy": 0.85, "loss": 0.5}
        tracker.log_epoch_metrics(epoch=1, metrics=metrics, phase="train")
        # Should not raise exception
        assert True


class TestCheckpointManager:
    """Critical tests for CheckpointManager."""

    def test_checkpoint_manager_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=str(Path(temp_dir) / "checkpoints"),
            run_id="test_run_123"
        )
        assert manager.checkpoint_dir.exists()

    def test_save_checkpoint(self, temp_dir):
        """Test saving checkpoint."""
        import torch
        import torch.nn as nn
        manager = CheckpointManager(
            checkpoint_dir=str(Path(temp_dir) / "checkpoints"),
            run_id="test_run_123"
        )
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = None
        metrics = {"val_loss": 0.5, "val_accuracy": 0.85}
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics=metrics
        )
        # Should create checkpoint file
        checkpoints = list(manager.checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0

    def test_load_checkpoint(self, temp_dir):
        """Test loading checkpoint."""
        import torch
        import torch.nn as nn
        manager = CheckpointManager(
            checkpoint_dir=str(Path(temp_dir) / "checkpoints"),
            run_id="test_run_123"
        )
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = None
        metrics = {"val_loss": 0.5, "val_accuracy": 0.85}
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            metrics=metrics
        )
        
        model2 = nn.Linear(10, 1)
        optimizer2 = torch.optim.Adam(model2.parameters())
        loaded = manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=None
        )
        assert loaded is not None
        assert loaded["epoch"] == 1

