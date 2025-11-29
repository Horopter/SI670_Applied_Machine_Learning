"""
MLOps infrastructure module.

Provides:
- Experiment tracking and configuration
- Checkpoint management
- Data versioning
- Pipeline orchestration (single-split, k-fold, multi-model)
- Cleanup utilities
"""

# Core components
from .config import (
    RunConfig,
    ExperimentTracker,
    CheckpointManager,
    DataVersionManager,
    create_run_directory,
)

# Pipelines
from .pipeline import (
    PipelineStage,
    MLOpsPipeline,
    build_mlops_pipeline,
    fit_with_tracking,
)
from .kfold import build_kfold_pipeline
from .multimodel import build_multimodel_pipeline

# Cleanup
from .cleanup import (
    cleanup_runs_and_logs,
    cleanup_intermediate_files,
)

__all__ = [
    # Core
    "RunConfig",
    "ExperimentTracker",
    "CheckpointManager",
    "DataVersionManager",
    "create_run_directory",
    # Pipelines
    "PipelineStage",
    "MLOpsPipeline",
    "build_mlops_pipeline",
    "fit_with_tracking",
    "build_kfold_pipeline",
    "build_multimodel_pipeline",
    # Cleanup
    "cleanup_runs_and_logs",
    "cleanup_intermediate_files",
]

