"""Comprehensive unit tests for mlops_core module."""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import polars as pl
from lib.mlops_core import (
    RunConfig,
    ExperimentTracker,
    CheckpointManager,
    DataVersionManager,
    create_run_directory,
)


class TestRunConfig:
    """Stress tests for RunConfig."""
    
    def test_basic_creation(self, sample_run_config):
        """Test basic RunConfig creation."""
        assert sample_run_config.run_id == "test_run_123"
        assert sample_run_config.experiment_name == "test_experiment"
        assert sample_run_config.num_frames == 8
        assert sample_run_config.fixed_size == 224
    
    def test_default_values(self):
        """Test RunConfig with default values."""
        config = RunConfig(
            run_id="test",
            experiment_name="test",
        )
        assert config.tags == []
        assert config.model_specific_config == {}
        assert config.train_split == 0.8
        assert config.val_split == 0.2
        assert config.random_seed == 42
    
    def test_serialization_roundtrip(self, sample_run_config):
        """Test JSON serialization and deserialization."""
        json_str = sample_run_config.to_json()
        assert isinstance(json_str, str)
        
        config_dict = json.loads(json_str)
        assert config_dict["run_id"] == "test_run_123"
        
        restored = RunConfig.from_json(json_str)
        assert restored.run_id == sample_run_config.run_id
        assert restored.experiment_name == sample_run_config.experiment_name
        assert restored.num_frames == sample_run_config.num_frames
    
    def test_dict_roundtrip(self, sample_run_config):
        """Test dictionary conversion roundtrip."""
        config_dict = sample_run_config.to_dict()
        assert isinstance(config_dict, dict)
        
        restored = RunConfig.from_dict(config_dict)
        assert restored.run_id == sample_run_config.run_id
        assert restored.num_frames == sample_run_config.num_frames
    
    def test_hash_deterministic(self, sample_run_config):
        """Test that hash is deterministic and excludes run_id."""
        hash1 = sample_run_config.compute_hash()
        hash2 = sample_run_config.compute_hash()
        assert hash1 == hash2
        
        # Same config with different run_id should have same hash
        # Create config from dict to ensure all fields are copied
        config_dict = sample_run_config.to_dict()
        config_dict["run_id"] = "different_id"
        config_dict["project_root"] = "/different/path"
        config_dict["output_dir"] = "/different/output"
        config2 = RunConfig.from_dict(config_dict)
        
        # Hashes should match (excluding run_id and paths)
        assert config2.compute_hash() == hash1
    
    def test_hash_different_configs(self, sample_run_config):
        """Test that different configs have different hashes."""
        hash1 = sample_run_config.compute_hash()
        
        config2 = RunConfig(
            run_id=sample_run_config.run_id,
            experiment_name=sample_run_config.experiment_name,
            num_frames=16,  # Different
            fixed_size=sample_run_config.fixed_size,
        )
        hash2 = config2.compute_hash()
        assert hash1 != hash2
    
    def test_extreme_values(self):
        """Test RunConfig with extreme values."""
        config = RunConfig(
            run_id="test",
            experiment_name="test",
            num_frames=1,  # Minimum
            fixed_size=32,  # Small
            batch_size=1,  # Minimum
            num_epochs=1,  # Minimum
            learning_rate=1e-8,  # Very small
            weight_decay=1e-8,
        )
        assert config.num_frames == 1
        assert config.fixed_size == 32
    
    def test_none_values(self):
        """Test RunConfig with None values."""
        config = RunConfig(
            run_id="test",
            experiment_name="test",
            description=None,
            fixed_size=None,
            augmentation_config=None,
        )
        assert config.description is None
        assert config.fixed_size is None


class TestExperimentTracker:
    """Stress tests for ExperimentTracker."""
    
    def test_basic_tracking(self, temp_dir):
        """Test basic experiment tracking."""
        tracker = ExperimentTracker(temp_dir, run_id="test_123")
        assert tracker.run_id == "test_123"
        assert tracker.run_dir.exists()
        assert tracker.metrics_file.exists()
    
    def test_auto_run_id(self, temp_dir):
        """Test automatic run ID generation."""
        tracker1 = ExperimentTracker(temp_dir)
        tracker2 = ExperimentTracker(temp_dir)
        assert tracker1.run_id != tracker2.run_id
        assert tracker1.run_id.startswith("run_")
    
    def test_log_config(self, temp_dir, sample_run_config):
        """Test logging configuration."""
        tracker = ExperimentTracker(temp_dir)
        tracker.log_config(sample_run_config)
        
        assert tracker.config_file.exists()
        with open(tracker.config_file) as f:
            config_data = json.load(f)
        assert config_data["run_id"] == tracker.run_id
        assert config_data["experiment_name"] == sample_run_config.experiment_name
    
    def test_log_metrics(self, temp_dir, sample_metrics):
        """Test logging metrics."""
        tracker = ExperimentTracker(temp_dir)
        
        # Log multiple metrics using log_epoch_metrics
        for i in range(5):
            metrics = {k: v + i * 0.01 for k, v in sample_metrics.items()}
            tracker.log_epoch_metrics(epoch=i, metrics=metrics, phase="train")
        
        # Verify metrics file
        assert tracker.metrics_file.exists()
        with open(tracker.metrics_file) as f:
            lines = f.readlines()
        # Each metric in the dict creates one line
        assert len(lines) >= 5
    
    def test_log_metrics_stress(self, temp_dir):
        """Stress test logging many metrics."""
        tracker = ExperimentTracker(temp_dir)
        
        # Log 1000 metrics using log_epoch_metrics
        for i in range(1000):
            metrics = {
                "loss": float(i),
                "accuracy": 0.5 + (i % 100) / 200.0,
            }
            tracker.log_epoch_metrics(epoch=i, metrics=metrics, phase="train")
        
        # Verify file exists and has content
        assert tracker.metrics_file.exists()
        assert tracker.metrics_file.stat().st_size > 0
    
    def test_get_best_metric(self, temp_dir):
        """Test getting best metric."""
        tracker = ExperimentTracker(temp_dir)
        
        # Log metrics with varying accuracy using log_epoch_metrics
        for i in range(10):
            metrics = {"val_accuracy": 0.5 + i * 0.05}
            tracker.log_epoch_metrics(epoch=i, metrics=metrics, phase="val")
        
        best = tracker.get_best_metric("val_accuracy", phase="val", maximize=True)
        assert best is not None
        assert best["value"] == pytest.approx(0.95, abs=0.01)
    
    def test_metadata_logging(self, temp_dir):
        """Test metadata logging."""
        tracker = ExperimentTracker(temp_dir)
        
        metadata = {
            "model_type": "test_model",
            "dataset_size": 1000,
            "training_time": 3600,
        }
        tracker.log_metadata(metadata)
        
        assert tracker.metadata_file.exists()
        with open(tracker.metadata_file) as f:
            data = json.load(f)
        assert data["model_type"] == "test_model"


class TestCheckpointManager:
    """Stress tests for CheckpointManager."""
    
    def test_basic_checkpointing(self, temp_dir):
        """Test basic checkpoint saving and loading."""
        ckpt_manager = CheckpointManager(temp_dir, "test_run")
        
        # Create a simple model
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # Save checkpoint
        metrics = {"val_accuracy": 0.85, "val_loss": 0.5}
        ckpt_path = ckpt_manager.save_checkpoint(
            model, optimizer, scheduler, epoch=5, metrics=metrics, is_best=True
        )
        
        assert Path(ckpt_path).exists()
        assert (Path(temp_dir) / "best_model.pt").exists()
        assert (Path(temp_dir) / "latest_checkpoint.pt").exists()
    
    def test_resume_from_checkpoint(self, temp_dir):
        """Test resuming from checkpoint."""
        ckpt_manager = CheckpointManager(temp_dir, "test_run")
        
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # Save checkpoint
        ckpt_manager.save_checkpoint(
            model, optimizer, scheduler, epoch=10, metrics={"loss": 0.5}
        )
        
        # Create new model and resume
        model2 = nn.Linear(10, 2)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)
        
        start_epoch = ckpt_manager.resume_from_latest(model2, optimizer2, scheduler2)
        assert start_epoch == 11
    
    def test_stage_checkpoints(self, temp_dir):
        """Test stage checkpoint saving and loading."""
        ckpt_manager = CheckpointManager(temp_dir, "test_run")
        
        # Save stage checkpoint
        stage_data = {"data": [1, 2, 3], "metadata": {"key": "value"}}
        ckpt_path = ckpt_manager.save_stage_checkpoint("load_data", stage_data)
        
        assert Path(ckpt_path).exists()
        
        # Load stage checkpoint
        loaded_data = ckpt_manager.load_stage_checkpoint("load_data")
        assert loaded_data is not None
        assert loaded_data["data"] == [1, 2, 3]
    
    def test_multiple_checkpoints(self, temp_dir):
        """Test saving multiple checkpoints."""
        ckpt_manager = CheckpointManager(temp_dir, "test_run")
        
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Save multiple checkpoints
        for epoch in range(1, 11):
            ckpt_manager.save_checkpoint(
                model, optimizer, None, epoch=epoch, metrics={"loss": 1.0 / epoch}
            )
        
        # Verify latest checkpoint
        latest_path = Path(temp_dir) / "latest_checkpoint.pt"
        assert latest_path.exists()
        
        checkpoint = torch.load(latest_path, map_location='cpu')
        assert checkpoint["epoch"] == 10
    
    def test_checkpoint_without_scheduler(self, temp_dir):
        """Test checkpointing without scheduler."""
        ckpt_manager = CheckpointManager(temp_dir, "test_run")
        
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        ckpt_path = ckpt_manager.save_checkpoint(
            model, optimizer, None, epoch=1, metrics={"loss": 0.5}
        )
        
        assert Path(ckpt_path).exists()
        
        # Load without scheduler
        model2 = nn.Linear(10, 2)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        
        ckpt_manager.load_checkpoint(ckpt_path, model2, optimizer2, None)
        # Should not raise


class TestDataVersionManager:
    """Stress tests for DataVersionManager."""
    
    def test_basic_versioning(self, temp_dir):
        """Test basic data versioning."""
        version_manager = DataVersionManager(temp_dir)
        
        # Register data version using register_split
        version_manager.register_split(
            split_name="train_split",
            config_hash="test_hash",
            file_path="data/train.csv",
            metadata={"size": 1000}
        )
        
        assert "train_split" in version_manager.versions
        assert len(version_manager.versions["train_split"]) == 1
    
    def test_multiple_versions(self, temp_dir):
        """Test managing multiple data versions."""
        version_manager = DataVersionManager(temp_dir)
        
        # Register multiple versions using register_split
        for i in range(10):
            version_manager.register_split(
                split_name=f"split_{i}",
                config_hash=f"hash_{i}",
                file_path=f"data/split_{i}.csv",
                metadata={"index": i}
            )
        
        assert len(version_manager.versions) == 10
    
    def test_version_metadata(self, temp_dir):
        """Test version metadata storage."""
        version_manager = DataVersionManager(temp_dir)
        
        metadata = {
            "size": 1000,
            "classes": [0, 1],
            "augmented": True,
        }
        version_manager.register_split(
            split_name="test",
            config_hash="test_hash",
            file_path="data/test.csv",
            metadata=metadata
        )
        
        assert "test" in version_manager.versions
        version_entry = version_manager.versions["test"][0]
        assert version_entry["metadata"]["size"] == 1000


class TestCreateRunDirectory:
    """Stress tests for create_run_directory."""
    
    def test_basic_creation(self, temp_dir):
        """Test basic run directory creation."""
        run_dir, run_id = create_run_directory(
            base_dir=temp_dir,
            experiment_name="test",
            run_id="test_123",
        )
        
        assert Path(run_dir).exists()
        assert run_id == "test_123"
        assert "test_123" in run_dir
    
    def test_nested_directories(self, temp_dir):
        """Test creating nested run directories."""
        run_dir, run_id = create_run_directory(
            base_dir=temp_dir,
            experiment_name="test/experiment",
            run_id="test_123",
        )
        
        assert Path(run_dir).exists()
        assert "test" in run_dir
        assert "experiment" in run_dir
        assert run_id == "test_123"

