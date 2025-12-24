"""
Model training pipeline.

Trains models using scaled videos and extracted features.
Supports multiple model types and k-fold cross-validation.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.data import stratified_kfold
# Lazy import to avoid circular dependency issues
# VideoConfig and VideoDataset will be imported when needed
from lib.mlops.config import ExperimentTracker, CheckpointManager
from lib.mlops.mlflow_tracker import create_mlflow_tracker, MLFLOW_AVAILABLE
from lib.training.trainer import OptimConfig, TrainConfig, fit
from lib.training.model_factory import create_model, is_pytorch_model, is_xgboost_model, get_model_config
from lib.training.metrics_utils import compute_classification_metrics
from lib.training.cleanup_utils import cleanup_model_and_memory
from lib.training.visualization import plot_learning_curves
from lib.utils.memory import aggressive_gc

# Optional integrations
try:
    from lib.utils.duckdb_analytics import DuckDBAnalytics, DUCKDB_AVAILABLE
except ImportError:
    DUCKDB_AVAILABLE = False
    DuckDBAnalytics = None

logger = logging.getLogger(__name__)


def _flush_logs():
    """Flush all logging handlers and stdout/stderr."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for handler in logging.root.handlers:
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
                handler.stream.flush()
            elif hasattr(handler, 'flush'):
                handler.flush()
    except (OSError, AttributeError, RuntimeError) as e:
        logger.debug(f"Error flushing logs: {e}")


BASELINE_MODELS = {
    "logistic_regression",
    "logistic_regression_stage2",
    "logistic_regression_stage2_stage4",
    "svm",
    "svm_stage2",
    "svm_stage2_stage4"
}

STAGE4_MODELS = {
    "logistic_regression_stage2_stage4",
    "svm_stage2_stage4"
}

MEMORY_INTENSIVE_MODELS_BATCH_LIMITS = {
    "x3d": 1,  # Very memory intensive
    "naive_cnn": 1,  # Processes 1000 frames at full resolution - must use batch_size=1
    "variable_ar_cnn": 2,  # Processes variable-length videos with many frames
    "pretrained_inception": 2,  # Large pretrained model processing many frames
}

MODEL_FILE_EXTENSIONS = ["*.pt", "*.joblib", "*.json"]

# Mapping from model_type to MLflow experiment name
# This mapping ensures MLflow tags match what the notebooks expect
MLFLOW_MODEL_TYPE_MAPPING = {
    "xgboost_pretrained_inception": "pretrained_inception",
    "xgboost_i3d": "x3d",
    "xgboost_r2plus1d": "slowfast",
    # For models that use the same name in both
    "logistic_regression": "logistic_regression",
    "svm": "svm",
    "sklearn_logreg": "sklearn_logreg",
    "gradient_boosting/xgboost": "gradient_boosting/xgboost"
}


def _copy_model_files(source_dir: Path, dest_dir: Path, model_name: str = "") -> None:
    """
    Copy model files from source directory to destination directory.
    
    Args:
        source_dir: Source directory containing model files
        dest_dir: Destination directory to copy files to
        model_name: Optional model name for logging
    
    Raises:
        OSError: If copying fails
    """
    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return
    
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create destination directory {dest_dir}: {e}")
        raise OSError(f"Cannot create destination directory: {dest_dir}") from e
    
    copied_count = 0
    for ext in MODEL_FILE_EXTENSIONS:
        for model_file in source_dir.glob(ext):
            try:
                shutil.copy2(model_file, dest_dir / model_file.name)
                copied_count += 1
            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Failed to copy {model_file} to {dest_dir}: {e}")
    
    if copied_count > 0:
        log_msg = f"Copied {copied_count} model file(s) from {source_dir.name} to {dest_dir.name}"
        if model_name:
            log_msg = f"Saved best model from {model_name}: {log_msg}"
        logger.info(log_msg)
    else:
        logger.warning(f"No model files found in {source_dir} to copy")


def _ensure_lib_models_exists(project_root_path: Path) -> None:
    """Create minimal stub files for lib/models if missing."""
    models_dir = project_root_path / 'lib' / 'models'
    models_init = models_dir / '__init__.py'
    video_py = models_dir / 'video.py'
    
    if models_dir.exists() and models_init.exists() and video_py.exists():
        return
    
    models_dir.mkdir(parents=True, exist_ok=True)
    if not models_init.exists():
        try:
            models_init.write_text('''"""
Video models and datasets module (minimal stub).

This is a minimal stub created automatically.
For full functionality, ensure lib/models is properly synced to the server.
"""

from .video import VideoConfig, VideoDataset

__all__ = ["VideoConfig", "VideoDataset"]
''')
            logger.info(f"Created minimal lib/models/__init__.py at {models_init}")
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"Failed to create minimal __init__.py at {models_init}: {e}")
            raise
    
    if not video_py.exists():
        try:
            video_py.write_text('''"""
Video configuration and dataset (minimal stub).

This is a minimal stub created automatically.
For full functionality, ensure lib/models/video.py is properly synced to the server.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import torch
from torch.utils.data import Dataset
import polars as pl


@dataclass
class VideoConfig:
    """Configuration for video sampling and preprocessing (minimal stub)."""
    num_frames: int = 16
    fixed_size: Optional[int] = None
    max_size: Optional[int] = None
    img_size: Optional[int] = None
    rolling_window: bool = False
    window_size: Optional[int] = None
    window_stride: Optional[int] = None
    augmentation_config: Optional[dict] = None
    temporal_augmentation_config: Optional[dict] = None
    use_scaled_videos: bool = False


class VideoDataset(Dataset):
    """Dataset over videos (minimal stub - will fail at runtime if used without full implementation)."""
    
    def __init__(
        self,
        df: Union[pl.DataFrame, Any],
        project_root: str,
        config: VideoConfig,
        train: bool = True,
        max_videos: Optional[int] = None,
    ) -> None:
        raise RuntimeError(
            "VideoDataset stub cannot be used. "
            "Please ensure lib/models/video.py is properly synced to the server. "
            f"Expected at: {project_root}/lib/models/video.py"
        )
    
    def __len__(self) -> int:
        raise RuntimeError("VideoDataset stub cannot be used")
    
    def __getitem__(self, idx: int):
        raise RuntimeError("VideoDataset stub cannot be used")
''')
            logger.info(f"Created minimal lib/models/video.py at {video_py}")
            logger.warning(
                "Created minimal lib/models stub. "
                "For PyTorch models to work, ensure the full lib/models directory is synced to the server."
            )
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"Failed to create minimal video.py at {video_py}: {e}")
            raise


def _validate_stage5_prerequisites(
    project_root: Path,
    scaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: str,
    model_types: List[str]
) -> Dict[str, Any]:
    """
    Validate that all required data exists before starting Stage 5 training.
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to Stage 3 scaled metadata
        features_stage2_path: Path to Stage 2 features metadata
        features_stage4_path: Path to Stage 4 features metadata
        model_types: List of model types to train
    
    Returns:
        Dictionary with validation results:
        - stage3_available: bool
        - stage2_available: bool
        - stage4_available: bool
        - stage3_count: int (number of videos)
        - stage2_count: int (number of feature rows)
        - stage4_count: int (number of feature rows)
        - runnable_models: List[str] (models that can be run)
        - missing_models: Dict[str, List[str]] (models that cannot run and why)
    """
    from lib.utils.paths import load_metadata_flexible
    
    results = {
        "stage3_available": False,
        "stage2_available": False,
        "stage4_available": False,
        "stage3_count": 0,
        "stage2_count": 0,
        "stage4_count": 0,
        "runnable_models": [],
        "missing_models": {}  # model_type -> [list of missing requirements]
    }
    
    logger.info("=" * 80)
    logger.info("STAGE 5 PREREQUISITE VALIDATION")
    logger.info("=" * 80)
    
    logger.info("\n[1/3] Checking Stage 3 (scaled videos) - REQUIRED for all models...")
    scaled_df = load_metadata_flexible(scaled_metadata_path)
    if scaled_df is None or scaled_df.height == 0:
        logger.error(f"✗ Stage 3 metadata not found or empty: {scaled_metadata_path}")
        logger.error("  Stage 3 is REQUIRED for all models. Please run Stage 3 first.")
        results["stage3_available"] = False
    else:
        results["stage3_available"] = True
        results["stage3_count"] = scaled_df.height
        logger.info(f"✓ Stage 3 metadata found: {scaled_df.height} scaled videos")
        logger.info(f"  Path: {scaled_metadata_path}")
    
    logger.info("\n[2/3] Checking Stage 2 (features) - REQUIRED for *_stage2 models...")
    features2_df = load_metadata_flexible(features_stage2_path)
    if features2_df is None or features2_df.height == 0:
        logger.warning(f"✗ Stage 2 metadata not found or empty: {features_stage2_path}")
        results["stage2_available"] = False
    else:
        results["stage2_available"] = True
        results["stage2_count"] = features2_df.height
        logger.info(f"✓ Stage 2 metadata found: {features2_df.height} feature rows")
        logger.info(f"  Path: {features_stage2_path}")
    
    logger.info("\n[3/3] Checking Stage 4 (scaled features) - REQUIRED for *_stage2_stage4 models...")
    features4_df = load_metadata_flexible(features_stage4_path)
    if features4_df is None or features4_df.height == 0:
        logger.warning(f"✗ Stage 4 metadata not found or empty: {features_stage4_path}")
        results["stage4_available"] = False
    else:
        results["stage4_available"] = True
        results["stage4_count"] = features4_df.height
        logger.info(f"✓ Stage 4 metadata found: {features4_df.height} feature rows")
        logger.info(f"  Path: {features_stage4_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL REQUIREMENTS CHECK")
    logger.info("=" * 80)
    
    for model_type in model_types:
        missing = []
        
        if not results["stage3_available"]:
            missing.append("Stage 3 (scaled videos)")
        
        if model_type in BASELINE_MODELS and not results["stage2_available"]:
            missing.append("Stage 2 (features)")
        
        if model_type in STAGE4_MODELS and not results["stage4_available"]:
            missing.append("Stage 4 (scaled features)")
        
        if missing:
            results["missing_models"][model_type] = missing
            logger.error(f"✗ {model_type}: CANNOT RUN - Missing: {', '.join(missing)}")
        else:
            results["runnable_models"].append(model_type)
            logger.info(f"✓ {model_type}: CAN RUN")
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Stage 3 (scaled videos): {'✓ Available' if results['stage3_available'] else '✗ MISSING'}")
    logger.info(f"  Count: {results['stage3_count']} videos")
    logger.info(f"Stage 2 (features): {'✓ Available' if results['stage2_available'] else '✗ MISSING'}")
    logger.info(f"  Count: {results['stage2_count']} feature rows")
    logger.info(f"Stage 4 (scaled features): {'✓ Available' if results['stage4_available'] else '✗ MISSING'}")
    logger.info(f"  Count: {results['stage4_count']} feature rows")
    logger.info(f"\nRunnable models: {len(results['runnable_models'])}/{len(model_types)}")
    logger.info(f"  {results['runnable_models']}")
    
    if results["missing_models"]:
        logger.error(f"\nCannot run models: {len(results['missing_models'])}/{len(model_types)}")
        for model_type, missing in results["missing_models"].items():
            logger.error(f"  {model_type}: Missing {', '.join(missing)}")
    
    logger.info("=" * 80)
    
    failure_report_path = project_root / "logs" / "stage5_prerequisite_failures.txt"
    failure_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    failures_detected = False
    failure_lines = []
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    failure_lines.append("=" * 80)
    failure_lines.append("STAGE 5 PREREQUISITE FAILURE REPORT")
    failure_lines.append(f"Generated: {timestamp}")
    failure_lines.append("=" * 80)
    failure_lines.append("")
    
    stage2_required_models = [m for m in model_types if m in BASELINE_MODELS]
    if stage2_required_models and not results["stage2_available"]:
        failures_detected = True
        failure_lines.append("STAGE 2 FAILURE DETECTED")
        failure_lines.append("-" * 80)
        failure_lines.append(f"Expected: Stage 2 features metadata at: {features_stage2_path}")
        failure_lines.append(f"Status: NOT FOUND or EMPTY")
        failure_lines.append(f"Required for ALL baseline models: {', '.join(stage2_required_models)}")
        failure_lines.append("Note: All baseline models (svm, logistic_regression and variants) require Stage 2 features.")
        failure_lines.append("")
        failure_lines.append("ACTION REQUIRED:")
        failure_lines.append("  - Run Stage 2 feature extraction:")
        failure_lines.append("    sbatch scripts/slurm_jobs/slurm_stage2_features.sh")
        failure_lines.append("  - Or check if Stage 2 output exists at a different location")
        failure_lines.append("")
    
    stage4_required_models = [m for m in model_types if m in STAGE4_MODELS]
    if stage4_required_models and not results["stage4_available"]:
        failures_detected = True
        failure_lines.append("STAGE 4 FAILURE DETECTED")
        failure_lines.append("-" * 80)
        failure_lines.append(f"Expected: Stage 4 features metadata at: {features_stage4_path}")
        failure_lines.append(f"Status: NOT FOUND or EMPTY")
        failure_lines.append(f"Required for models: {', '.join(stage4_required_models)}")
        failure_lines.append("")
        failure_lines.append("ACTION REQUIRED:")
        failure_lines.append("  - Run Stage 4 scaled feature extraction:")
        failure_lines.append("    sbatch scripts/slurm_jobs/slurm_stage4_scaled_features.sh")
        failure_lines.append("  - Or check if Stage 4 output exists at a different location")
        failure_lines.append("")
    
    if not results["stage3_available"]:
        failures_detected = True
        failure_lines.append("STAGE 3 FAILURE DETECTED (CRITICAL)")
        failure_lines.append("-" * 80)
        failure_lines.append(f"Expected: Stage 3 scaled videos metadata at: {scaled_metadata_path}")
        failure_lines.append(f"Status: NOT FOUND or EMPTY")
        failure_lines.append(f"Required for ALL models: {', '.join(model_types)}")
        failure_lines.append("")
        failure_lines.append("ACTION REQUIRED:")
        failure_lines.append("  - Run Stage 3 video scaling:")
        failure_lines.append("    sbatch scripts/slurm_jobs/slurm_stage3_scaling.sh")
        failure_lines.append("  - Or check if Stage 3 output exists at a different location")
        failure_lines.append("")
    
    if failures_detected:
        failure_lines.append("=" * 80)
        failure_lines.append("SUMMARY")
        failure_lines.append("=" * 80)
        failure_lines.append(f"Total models requested: {len(model_types)}")
        failure_lines.append(f"Runnable models: {len(results['runnable_models'])}")
        failure_lines.append(f"Cannot run models: {len(results['missing_models'])}")
        failure_lines.append("")
        if results["runnable_models"]:
            failure_lines.append("Models that CAN run:")
            for model in results["runnable_models"]:
                failure_lines.append(f"  ✓ {model}")
            failure_lines.append("")
        if results["missing_models"]:
            failure_lines.append("Models that CANNOT run:")
            for model, missing in results["missing_models"].items():
                failure_lines.append(f"  ✗ {model}: Missing {', '.join(missing)}")
        failure_lines.append("")
        failure_lines.append("=" * 80)
        
        try:
            with open(failure_report_path, 'w') as f:
                f.write('\n'.join(failure_lines))
            logger.info(f"\n⚠ Failure report written to: {failure_report_path}")
        except Exception as e:
            logger.warning(f"Failed to write failure report: {e}")
    
    return results


def _save_metrics_to_duckdb(
    metrics: Dict[str, Any],
    model_type: str,
    fold_idx: int,
    project_root_str: str
) -> None:
    """
    Save metrics to DuckDB database for analytics.
    
    Args:
        metrics: Dictionary of metrics to save
        model_type: Model type identifier
        fold_idx: Fold index (0-based, or -1 for aggregated metrics)
        project_root_str: Project root directory as string
    """
    if not DUCKDB_AVAILABLE:
        logger.debug("DuckDB not available, skipping metrics save")
        return
    
    try:
        from lib.utils.duckdb_analytics import DuckDBAnalytics
        import polars as pl
        from datetime import datetime
        
        # Create DuckDB database path
        db_path = Path(project_root_str) / "data" / "stage5_metrics.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB analytics
        analytics = DuckDBAnalytics(str(db_path))
        
        # Prepare metrics DataFrame
        metrics_dict = {
            "model_type": [model_type],
            "fold_idx": [fold_idx],
            "timestamp": [datetime.now().isoformat()],
        }
        
        # Add metric values
        metric_keys = [
            "val_loss", "val_acc", "val_f1", "val_precision", "val_recall",
            "val_f1_class0", "val_precision_class0", "val_recall_class0",
            "val_f1_class1", "val_precision_class1", "val_recall_class1",
            "avg_val_loss", "avg_val_acc", "avg_val_f1", "avg_val_precision", "avg_val_recall",
            "std_val_loss", "std_val_acc", "std_val_f1", "std_val_precision", "std_val_recall"
        ]
        
        for key in metric_keys:
            if key in metrics:
                value = metrics[key]
                # Handle NaN values
                if isinstance(value, float) and (value != value):  # NaN check
                    metrics_dict[key] = [None]
                else:
                    metrics_dict[key] = [value]
        
        # Create DataFrame
        df = pl.DataFrame(metrics_dict)
        
        # Register DataFrame with DuckDB
        table_name = "training_metrics"
        analytics.register_dataframe("metrics_temp", df)
        
        # Create table if it doesn't exist, or insert into existing table
        try:
            # Check if table exists
            result = analytics.conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()
            
            if result and result[0] > 0:
                # Table exists, insert
                analytics.conn.execute(
                    f"INSERT INTO {table_name} SELECT * FROM metrics_temp"
                )
            else:
                # Table doesn't exist, create it
                analytics.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM metrics_temp"
                )
        except Exception as e:
            # Fallback: try to create table directly
            try:
                analytics.conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM metrics_temp"
                )
            except Exception:
                pass
        
        analytics.close()
        logger.debug(f"Saved metrics to DuckDB for {model_type} fold {fold_idx}")
    except Exception as e:
        logger.debug(f"Failed to save metrics to DuckDB: {e}")
        # Don't raise - DuckDB is optional


def _check_airflow_status(
    model_type: str,
    project_root_str: str
) -> Optional[Dict[str, Any]]:
    """
    Check Airflow DAG run status for the model.
    
    Args:
        model_type: Model type identifier
        project_root_str: Project root directory as string
    
    Returns:
        Dictionary with Airflow status information, or None if unavailable
    """
    try:
        # Check if Airflow is configured
        airflow_dag_id = os.environ.get("AIRFLOW_DAG_ID")
        if not airflow_dag_id:
            return None
        
        # Try to query Airflow API or check status file
        status_file = Path(project_root_str) / "logs" / "airflow_status.json"
        if status_file.exists():
            import json
            with open(status_file, 'r') as f:
                status_data = json.load(f)
                return status_data.get(model_type)
        
        return None
    except Exception as e:
        logger.debug(f"Failed to check Airflow status: {e}")
        return None


def _train_xgboost_model_fold(
    model_type: str,
    model_config: Dict[str, Any],
    train_df: Any,
    val_df: Any,
    project_root_str: str,
    fold_idx: int,
    model_output_dir: Path,
    hyperparams: Optional[Dict[str, Any]] = None,
    is_grid_search: bool = False,
    param_fold_results: Optional[List[Dict[str, Any]]] = None,
    fold_results: Optional[List[Dict[str, Any]]] = None,
    use_mlflow: bool = True,
    param_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Train an XGBoost model on a single fold.
    
    Args:
        model_type: Type of model to train
        model_config: Base model configuration
        train_df: Training dataframe
        val_df: Validation dataframe
        project_root_str: Project root as string
        fold_idx: Fold index (0-based)
        model_output_dir: Output directory for models
        hyperparams: Optional hyperparameters to apply (for grid search or final training)
        is_grid_search: Whether this is grid search (affects error handling and result storage)
        param_fold_results: Optional list to append grid search results to
        fold_results: Optional list to append results to
        use_mlflow: Whether to use MLflow tracking
        param_idx: Optional parameter combination index for grid search
    
    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Training XGBoost model {model_type} on fold {fold_idx + 1}...")
    _flush_logs()
    
    fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    result = None
    model = None
    mlflow_tracker = None
    
    # Initialize MLflow tracker if enabled
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            import mlflow
            if mlflow.active_run() is not None:
                mlflow.end_run()
                logger.debug("Ended existing MLflow run before creating new one")
        except (RuntimeError, AttributeError, ValueError) as mlflow_error:
            logger.debug(f"Error ending MLflow run (non-critical): {mlflow_error}")
        
        try:
            # Map model_type to MLflow experiment name using the same mapping as notebooks
            mlflow_model_type = MLFLOW_MODEL_TYPE_MAPPING.get(model_type, model_type)
            mlflow_tracker = create_mlflow_tracker(experiment_name=f"{mlflow_model_type}", use_mlflow=True)
            if mlflow_tracker:
                # CRITICAL: Log the run_id so we can map logs to MLflow runs
                logger.info(
                    f"MLflow run started: run_id={mlflow_tracker.run_id}, "
                    f"experiment={mlflow_model_type}, model={model_type}, "
                    f"fold={fold_idx + 1}, param_combo={param_idx + 1 if param_idx is not None else 'final'}"
                )
                mlflow_tracker.log_config(model_config)
                mlflow_tracker.set_tag("fold", str(fold_idx + 1))
                mlflow_tracker.set_tag("model_type", mlflow_model_type)
                if param_idx is not None:
                    mlflow_tracker.set_tag("param_combination", str(param_idx + 1))
                if hyperparams:
                    # Log hyperparameters as MLflow parameters
                    import mlflow
                    for key, value in hyperparams.items():
                        if isinstance(value, (str, int, float, bool)):
                            mlflow.log_param(key, value)
            else:
                logger.warning(f"MLflow tracker creation returned None for {model_type}")
        except (RuntimeError, ValueError, AttributeError, ImportError) as e:
            logger.warning(f"Failed to create MLflow tracker: {e}")
    
    try:
        xgb_config = model_config.copy()
        if hyperparams:
            xgb_config.update(hyperparams)
        model = create_model(model_type, xgb_config)
        
        model.fit(train_df, project_root=project_root_str)
        
        # Capture XGBoost training history for epoch-wise curves
        try:
            from lib.mlops.config import ExperimentTracker
            # Check if model has evals_result (XGBoost training history)
            if hasattr(model, 'model') and hasattr(model.model, 'evals_result_'):
                evals_result = model.model.evals_result_
                if evals_result:
                    tracker = ExperimentTracker(fold_output_dir)
                    
                    # Extract metrics per boosting round
                    # XGBoost stores metrics in evals_result_ as:
                    # {'validation_0': {'logloss': [...]}, 'validation_1': {'logloss': [...]}}
                    train_losses = evals_result.get('validation_0', {}).get('logloss', [])
                    val_losses = evals_result.get('validation_1', {}).get('logloss', [])
                    
                    # If we have both train and val losses, log them
                    if train_losses and val_losses:
                        for round_idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                            tracker.log_epoch_metrics(round_idx + 1, {"loss": float(train_loss)}, phase="train")
                            tracker.log_epoch_metrics(round_idx + 1, {"loss": float(val_loss)}, phase="val")
                        logger.info(f"Logged {len(train_losses)} boosting rounds to metrics.jsonl")
                    elif val_losses:
                        # Only validation losses available
                        for round_idx, val_loss in enumerate(val_losses):
                            tracker.log_epoch_metrics(round_idx + 1, {"loss": float(val_loss)}, phase="val")
                        logger.info(f"Logged {len(val_losses)} validation boosting rounds to metrics.jsonl")
        except Exception as e:
            logger.debug(f"Could not capture XGBoost training history: {e}")
        
        val_probs = model.predict(val_df, project_root=project_root_str)
        val_preds = np.argmax(val_probs, axis=1)
        val_labels = val_df["label"].to_list()
        label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
        val_y = np.array([label_map[label] for label in val_labels])
        
        metrics = compute_classification_metrics(
            y_true=val_y,
            y_pred=val_preds,
            y_probs=val_probs
        )
        
        # Store results
        result = {
            "fold": fold_idx + 1,
            "val_loss": metrics["val_loss"],
            "val_acc": metrics["val_acc"],
            "val_f1": metrics["val_f1"],
            "val_precision": metrics["val_precision"],
            "val_recall": metrics["val_recall"],
            "val_f1_class0": metrics["val_f1_class0"],
            "val_precision_class0": metrics["val_precision_class0"],
            "val_recall_class0": metrics["val_recall_class0"],
            "val_f1_class1": metrics["val_f1_class1"],
            "val_precision_class1": metrics["val_precision_class1"],
            "val_recall_class1": metrics["val_recall_class1"],
        }
        if hyperparams:
            result.update(hyperparams)
        
        if is_grid_search and param_fold_results is not None:
            param_fold_results.append(result)
        if fold_results is not None:
            fold_results.append(result)
        
        logger.info(
            f"Fold {fold_idx + 1} - Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.4f}, "
            f"Val F1: {metrics['val_f1']:.4f}, Val Precision: {metrics['val_precision']:.4f}, Val Recall: {metrics['val_recall']:.4f}"
        )
        if is_grid_search:
            logger.info(
                f"  Class 0 - Precision: {metrics['val_precision_class0']:.4f}, "
                f"Recall: {metrics['val_recall_class0']:.4f}, F1: {metrics['val_f1_class0']:.4f}"
            )
            logger.info(
                f"  Class 1 - Precision: {metrics['val_precision_class1']:.4f}, "
                f"Recall: {metrics['val_recall_class1']:.4f}, F1: {metrics['val_f1_class1']:.4f}"
            )
        
        # Save model
        model.save(str(fold_output_dir))
        logger.info(f"Saved XGBoost model to {fold_output_dir}")
        
        # Log metrics to MLflow
        if mlflow_tracker is not None:
            try:
                mlflow_metrics = {
                    "val_loss": metrics["val_loss"],
                    "val_acc": metrics["val_acc"],
                    "val_f1": metrics["val_f1"],
                    "val_precision": metrics["val_precision"],
                    "val_recall": metrics["val_recall"],
                    "val_f1_class0": metrics["val_f1_class0"],
                    "val_precision_class0": metrics["val_precision_class0"],
                    "val_recall_class0": metrics["val_recall_class0"],
                    "val_f1_class1": metrics["val_f1_class1"],
                    "val_precision_class1": metrics["val_precision_class1"],
                    "val_recall_class1": metrics["val_recall_class1"],
                }
                mlflow_tracker.log_metrics(mlflow_metrics, step=fold_idx + 1)
                
                # Log model artifact - find the saved model file
                model_files = list(fold_output_dir.glob("xgboost_model.json"))
                if not model_files:
                    model_files = list(fold_output_dir.glob("*.json"))
                if model_files:
                    model_path = model_files[0]
                    if model_path.exists() and model_path.stat().st_size > 0:
                        mlflow_tracker.log_artifact(str(model_path), artifact_path="models")
                        logger.info(f"Logged model artifact to MLflow: {model_path.name}")
                else:
                    logger.debug(f"No model file found to log to MLflow in {fold_output_dir}")
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.error(f"Failed to log to MLflow: {e}", exc_info=True)
        
        # Save metrics to DuckDB
        _save_metrics_to_duckdb(result, model_type, fold_idx, project_root_str)
        
        # Check Airflow status
        airflow_status = _check_airflow_status(model_type, project_root_str)
        if airflow_status:
            logger.debug(f"Airflow status: {airflow_status}")
        
        # Generate plots for this fold
        try:
            from .visualization import plot_fold_metrics
            # Extract positive class probabilities for plotting
            if val_probs.ndim == 2 and val_probs.shape[1] == 2:
                y_proba_pos = val_probs[:, 1]
            else:
                y_proba_pos = val_probs.flatten() if val_probs.ndim > 1 else val_probs
            
            plot_fold_metrics(
                y_true=val_y,
                y_pred=val_preds,
                y_probs=val_probs,
                fold_output_dir=fold_output_dir,
                fold_num=fold_idx + 1,
                model_type=model_type
            )
            
            # Generate comprehensive additional plots
            from .visualization import (
                plot_calibration_curve, plot_per_class_metrics, 
                plot_error_analysis, plot_feature_importance
            )
            
            # Calibration curve
            try:
                plot_calibration_curve(
                    y_true=val_y,
                    y_proba=val_probs,
                    save_path=fold_output_dir / "calibration_curve.png",
                    title=f"{model_type} - Fold {fold_idx + 1} - Calibration Curve"
                )
            except Exception as e:
                logger.debug(f"Failed to generate calibration curve: {e}")
            
            # Per-class metrics breakdown
            try:
                metrics_per_class = {
                    "Class 0": {
                        "precision": metrics.get("val_precision_class0", 0),
                        "recall": metrics.get("val_recall_class0", 0),
                        "f1": metrics.get("val_f1_class0", 0)
                    },
                    "Class 1": {
                        "precision": metrics.get("val_precision_class1", 0),
                        "recall": metrics.get("val_recall_class1", 0),
                        "f1": metrics.get("val_f1_class1", 0)
                    }
                }
                plot_per_class_metrics(
                    metrics_per_class=metrics_per_class,
                    save_path=fold_output_dir / "per_class_metrics.png",
                    title=f"{model_type} - Fold {fold_idx + 1} - Per-Class Metrics"
                )
            except Exception as e:
                logger.debug(f"Failed to generate per-class metrics plot: {e}")
            
            # Error analysis
            try:
                plot_error_analysis(
                    y_true=val_y,
                    y_pred=val_preds,
                    y_proba=val_probs,
                    save_path=fold_output_dir / "error_analysis.png",
                    title=f"{model_type} - Fold {fold_idx + 1} - Error Analysis"
                )
            except Exception as e:
                logger.debug(f"Failed to generate error analysis plot: {e}")
            
            # Feature importance (for XGBoost models)
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                    feature_names = getattr(model, 'feature_names', None)
                    if feature_names is None:
                        # Try to get from model
                        try:
                            feature_names = [f"feature_{i}" for i in range(len(model.model.feature_importances_))]
                        except:
                            feature_names = None
                    
                    if feature_names:
                        feature_importance_dict = dict(zip(feature_names, model.model.feature_importances_))
                        plot_feature_importance(
                            feature_importance=feature_importance_dict,
                            save_path=fold_output_dir / "feature_importance.png",
                            title=f"{model_type} - Fold {fold_idx + 1} - Feature Importance"
                        )
            except Exception as e:
                logger.debug(f"Failed to generate feature importance plot: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to generate plots for XGBoost fold {fold_idx + 1}: {e}", exc_info=True)
    
    except Exception as e:
        logger.error(f"Error training XGBoost fold {fold_idx + 1}: {e}", exc_info=True)
        result = {
            "fold": fold_idx + 1,
            "val_loss": float('nan'),
            "val_acc": float('nan'),
            "val_f1": float('nan'),
            "val_precision": float('nan'),
            "val_recall": float('nan'),
            "val_f1_class0": float('nan'),
            "val_precision_class0": float('nan'),
            "val_recall_class0": float('nan'),
            "val_f1_class1": float('nan'),
            "val_precision_class1": float('nan'),
            "val_recall_class1": float('nan'),
        }
        if hyperparams:
            result.update(hyperparams)
        if is_grid_search and param_fold_results is not None:
            param_fold_results.append(result)
        if fold_results is not None:
            fold_results.append(result)
    
    finally:
        # End MLflow run if active
        if mlflow_tracker is not None:
            try:
                run_id = mlflow_tracker.run_id
                mlflow_tracker.end_run()
                logger.info(f"MLflow run ended: run_id={run_id}, model={model_type}, fold={fold_idx + 1}")
            except (RuntimeError, AttributeError, ValueError) as cleanup_error:
                logger.warning(f"Error ending MLflow run: {cleanup_error}")
        
        cleanup_model_and_memory(model=model if model is not None else None, clear_cuda=is_grid_search)
        aggressive_gc(clear_cuda=is_grid_search)
    
    return result if result is not None else {}


def _train_pytorch_model_fold(
    model_type: str,
    model_config: Dict[str, Any],
    train_df: Any,
    val_df: Any,
    project_root_str: str,
    fold_idx: int,
    model_output_dir: Path,
    video_config: Any,
    hyperparams: Optional[Dict[str, Any]] = None,
    is_grid_search: bool = False,
    param_fold_results: Optional[List[Dict[str, Any]]] = None,
    fold_results: Optional[List[Dict[str, Any]]] = None,
    use_tracking: bool = True,
    use_mlflow: bool = True,
    param_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train a PyTorch model on a single fold.
    
    Args:
        model_type: Type of model to train
        model_config: Base model configuration
        train_df: Training dataframe
        val_df: Validation dataframe
        project_root_str: Project root as string
        fold_idx: Fold index (0-based)
        model_output_dir: Output directory for models
        video_config: VideoConfig instance
        hyperparams: Optional hyperparameters to apply
        is_grid_search: Whether this is grid search
        param_fold_results: Optional list to append grid search results to
        fold_results: Optional list to append results to
        use_tracking: Whether to use experiment tracking
        use_mlflow: Whether to use MLflow
        param_idx: Parameter combination index (for grid search)
    
    Returns:
        Dictionary with validation metrics
    """
    from lib.models import VideoDataset
    from lib.models.video import variable_ar_collate
    from lib.training.trainer import OptimConfig, TrainConfig, fit, evaluate
    from lib.mlops.config import ExperimentTracker, CheckpointManager
    from lib.mlops.mlflow_tracker import create_mlflow_tracker, MLFLOW_AVAILABLE
    
    logger.info(f"Training PyTorch model {model_type} on fold {fold_idx + 1}...")
    _flush_logs()
    
    fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = VideoDataset(train_df, project_root=project_root_str, config=video_config)
    val_dataset = VideoDataset(val_df, project_root=project_root_str, config=video_config)
    
    use_cuda = torch.cuda.is_available()
    current_config = model_config.copy()
    if hyperparams:
        current_config.update(hyperparams)
    
    num_workers = current_config.get("num_workers", model_config.get("num_workers", 0))
    batch_size = current_config.get("batch_size", model_config.get("batch_size", 8))
    gradient_accumulation_steps = current_config.get("gradient_accumulation_steps", model_config.get("gradient_accumulation_steps", 1))
    
    if model_type in MEMORY_INTENSIVE_MODELS_BATCH_LIMITS:
        max_batch_size = MEMORY_INTENSIVE_MODELS_BATCH_LIMITS[model_type]
        if batch_size > max_batch_size:
            effective_batch_size = batch_size * gradient_accumulation_steps
            logger.warning(
                f"{model_type} model requires batch_size<={max_batch_size} to prevent OOM. "
                f"Overriding batch_size from {batch_size} to {max_batch_size}. "
                f"Adjusting gradient_accumulation_steps to maintain effective batch size of {effective_batch_size}."
            )
            gradient_accumulation_steps = (effective_batch_size + max_batch_size - 1) // max_batch_size
            batch_size = max_batch_size
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    logger.info(
        f"Training configuration - Batch size: {batch_size}, "
        f"Gradient accumulation steps: {gradient_accumulation_steps}, "
        f"Effective batch size: {effective_batch_size}"
    )
    
    is_memory_intensive = model_type in ["x3d", "slowfast"]
    use_pin_memory = use_cuda and not is_memory_intensive
    prefetch_factor = 1 if (is_memory_intensive and num_workers > 0) else (2 if num_workers > 0 else None)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
        collate_fn=variable_ar_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
        collate_fn=variable_ar_collate,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if model_type in ["x3d", "slowfast"]:
            torch.backends.cudnn.enabled = True  # Keep enabled but optimize
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        aggressive_gc(clear_cuda=True)
        logger.info("Applied PyTorch memory optimizations: expandable_segments, cudnn.benchmark=False")
    
    try:
        model = create_model(model_type, model_config)
    except (TypeError, ValueError) as e:
        logger.error(f"Error creating model {model_type}: {e}")
        raise
    model = model.to(device)
    
    # Create optimizer and scheduler configs
    optim_cfg = OptimConfig(
        lr=current_config.get("learning_rate", model_config.get("learning_rate", 1e-4)),
        weight_decay=current_config.get("weight_decay", model_config.get("weight_decay", 1e-4)),
        max_grad_norm=current_config.get("max_grad_norm", model_config.get("max_grad_norm", 1.0)),
        backbone_lr=current_config.get("backbone_lr", model_config.get("backbone_lr", None)),
        head_lr=current_config.get("head_lr", model_config.get("head_lr", None)),
    )
    train_cfg = TrainConfig(
        num_epochs=current_config.get("num_epochs", model_config.get("num_epochs", 20)),
        device=str(device),
        log_interval=model_config.get("log_interval", 10),
        use_class_weights=model_config.get("use_class_weights", True),
        use_amp=model_config.get("use_amp", True),
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=model_config.get("early_stopping_patience", 5),
        scheduler_type=model_config.get("scheduler_type", "cosine"),
        warmup_epochs=model_config.get("warmup_epochs", 2),
        warmup_factor=model_config.get("warmup_factor", 0.1),
        log_grad_norm=model_config.get("log_grad_norm", False),
        hyper_aggressive_gc=model_type in ["x3d", "slowfast"],
    )
    
    use_differential_lr = model_type in [
        "i3d", "r2plus1d", "slowfast", "x3d", "pretrained_inception",
        "vit_gru", "vit_transformer"
    ]
    
    tracker = None
    ckpt_manager = None
    mlflow_tracker = None
    
    if use_tracking:
        tracker = ExperimentTracker(str(fold_output_dir))
    
    run_id = f"{model_type}_fold{fold_idx + 1}"
    if param_idx is not None:
        run_id += f"_param{param_idx + 1}"
    ckpt_manager = CheckpointManager(str(fold_output_dir), run_id=run_id)
    
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            import mlflow
            if mlflow.active_run() is not None:
                mlflow.end_run()
                logger.debug("Ended existing MLflow run before creating new one")
        except (RuntimeError, AttributeError, ValueError) as mlflow_error:
            logger.debug(f"Error ending MLflow run (non-critical): {mlflow_error}")
        
        try:
            mlflow_tracker = create_mlflow_tracker(experiment_name=f"{model_type}", use_mlflow=True)
            if mlflow_tracker:
                # CRITICAL: Log the run_id so we can map logs to MLflow runs
                logger.info(
                    f"MLflow run started: run_id={mlflow_tracker.run_id}, "
                    f"experiment={model_type}, model={model_type}, "
                    f"fold={fold_idx + 1}, param_combo={param_idx + 1 if param_idx is not None else 'final'}"
                )
                mlflow_tracker.log_config(model_config)
                mlflow_tracker.set_tag("fold", str(fold_idx + 1))
                mlflow_tracker.set_tag("model_type", model_type)
                if param_idx is not None:
                    mlflow_tracker.set_tag("param_combination", str(param_idx + 1))
            else:
                logger.warning(f"MLflow tracker creation returned None for {model_type}")
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to create MLflow tracker: {e}")
    
    is_hyper_aggressive = model_type in ["x3d", "slowfast"]
    if device.type == "cuda" and is_hyper_aggressive:
        logger.info(f"Performing hyper-aggressive GC before training {model_type}...")
        for _ in range(10):
            aggressive_gc(clear_cuda=True)
        for _ in range(10):
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        aggressive_gc(clear_cuda=True)
        torch.cuda.synchronize()
        logger.info(f"Hyper-aggressive GC complete. GPU memory cleared.")
    
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty for fold {fold_idx + 1}")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty for fold {fold_idx + 1}")
    
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=variable_ar_collate,
            )
            
            sample_batch = next(iter(test_loader))
            sample_clips, sample_labels = sample_batch
            
            if model_type in ["x3d", "slowfast"]:
                if sample_clips.dim() == 5:
                    if sample_clips.shape[1] == 3:
                        N, C, T, H, W = sample_clips.shape
                    else:
                        N, T, C, H, W = sample_clips.shape
                    
                    min_spatial_size = 32
                    if H < min_spatial_size or W < min_spatial_size:
                        logger.warning(
                            f"Skipping forward pass test for {model_type}: input spatial dimensions "
                            f"({H}x{W}) are too small (minimum {min_spatial_size}x{min_spatial_size} required). "
                            f"Temporal dimension: {T}. Continuing with training..."
                        )
                        _flush_logs()
                        del sample_batch, sample_clips, sample_labels, test_loader
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    else:
                        from lib.training.trainer import _convert_and_normalize_clips
                        sample_clips = _convert_and_normalize_clips(sample_clips)
                        sample_clips = sample_clips.to(device, non_blocking=False)
                        
                        if sample_clips.dtype != torch.float32:
                            sample_clips = sample_clips.float()
                        
                        try:
                            sample_output = model(sample_clips)
                            logger.info(f"Model forward pass test successful. Output shape: {sample_output.shape}")
                            _flush_logs()
                        except RuntimeError as oom_error:
                            error_msg = str(oom_error)
                            if "out of memory" in error_msg.lower():
                                logger.warning(
                                    f"OOM during forward pass test. Model: {model_type}, Batch size: 1. "
                                    f"This may indicate the model is too large for available GPU memory."
                                )
                                logger.warning("Attempting to continue with training (may fail if model is too large)...")
                            elif "smaller than kernel size" in error_msg.lower() or "input image" in error_msg.lower():
                                logger.warning(
                                    f"Input dimension mismatch during forward pass test for {model_type}: {oom_error}. "
                                    f"Input shape: {sample_clips.shape}. Training will handle this via error handling. Continuing..."
                                )
                            elif "sizes of tensors must match" in error_msg.lower() and model_type == "slowfast":
                                logger.warning(
                                    f"SlowFast temporal dimension mismatch during forward pass test: {oom_error}. "
                                    f"Input shape: {sample_clips.shape}. Training will handle this via error handling. Continuing..."
                                )
                            else:
                                raise
                        
                        del sample_batch, sample_clips, sample_labels
                        if 'sample_output' in locals():
                            del sample_output
                        del test_loader
                        
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            logger.error(f"CUDA OOM during model forward pass test: {e}. Model: {model_type}, Batch size: 1.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.warning("Continuing with training despite OOM in forward pass test...")
        else:
            logger.error(f"Model forward pass test failed: {e}", exc_info=True)
            raise ValueError(f"Model initialization failed: {e}") from e
    except (ValueError, RuntimeError) as e:
        logger.error(f"Model forward pass test failed: {e}", exc_info=True)
        raise ValueError(f"Model initialization failed: {e}") from e
    
    max_oom_retries = 3
    oom_retry_count = 0
    training_successful = False
    val_metrics = None
    
    while oom_retry_count <= max_oom_retries and not training_successful:
        try:
            if oom_retry_count > 0:
                new_batch_size = max(1, batch_size // (2 ** oom_retry_count))
                if new_batch_size < batch_size:
                    logger.warning(f"OOM retry {oom_retry_count}: Reducing batch size from {batch_size} to {new_batch_size}")
                    batch_size = new_batch_size
                    gradient_accumulation_steps = effective_batch_size // batch_size
                    if gradient_accumulation_steps < 1:
                        gradient_accumulation_steps = 1
                    
                    train_cfg.gradient_accumulation_steps = gradient_accumulation_steps
                    
                    train_loader = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=use_cuda,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                        collate_fn=variable_ar_collate,
                    )
                    val_loader = DataLoader(
                        val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=use_cuda,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                        collate_fn=variable_ar_collate,
                    )
                    
                    logger.info(
                        f"Retrying with batch_size={batch_size}, "
                        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
                        f"effective_batch_size={batch_size * gradient_accumulation_steps}"
                    )
                    
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            hyper_aggressive_gc = model_type in ["x3d", "slowfast"]
            model = fit(
                model,
                train_loader,
                val_loader,
                optim_cfg,
                train_cfg,
                use_differential_lr=use_differential_lr,
                hyper_aggressive_gc=hyper_aggressive_gc,
                tracker=tracker,  # Pass tracker to log metrics during training
            )
            
            if device.type == "cuda":
                if is_hyper_aggressive:
                    for _ in range(10):
                        aggressive_gc(clear_cuda=True)
                    for _ in range(10):
                        torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    torch.cuda.empty_cache()
            
            # Get metrics with probabilities for plotting
            val_metrics = evaluate(
                model, val_loader, device=str(device), 
                hyper_aggressive_gc=is_hyper_aggressive,
                return_probs=True  # Return probabilities for plotting
            )
            training_successful = True
            
            if device.type == "cuda" and is_hyper_aggressive:
                for _ in range(10):
                    aggressive_gc(clear_cuda=True)
                for _ in range(10):
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                aggressive_gc(clear_cuda=True)
                torch.cuda.synchronize()
            else:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                aggressive_gc(clear_cuda=device.type == "cuda")
        
        except RuntimeError as e:
            error_msg = str(e)
            if ("out of memory" in error_msg.lower() or "cuda" in error_msg.lower()) and oom_retry_count < max_oom_retries:
                logger.warning(
                    f"CUDA OOM during training (attempt {oom_retry_count + 1}/{max_oom_retries + 1}): {e}. "
                    f"Model: {model_type}, Fold: {fold_idx + 1}, Current batch size: {batch_size}"
                )
                if device.type == "cuda":
                    if is_hyper_aggressive:
                        for _ in range(10):
                            aggressive_gc(clear_cuda=True)
                        for _ in range(10):
                            torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        aggressive_gc(clear_cuda=True)
                        torch.cuda.synchronize()
                    else:
                        for _ in range(3):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        aggressive_gc(clear_cuda=True)
                oom_retry_count += 1
                continue
            else:
                if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                    logger.error(f"CUDA OOM or runtime error during training (max retries reached): {e}. Model: {model_type}, Fold: {fold_idx + 1}")
                else:
                    logger.error(f"Runtime error during training: {e}. Model: {model_type}, Fold: {fold_idx + 1}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error during training: {e}. Model: {model_type}, Fold: {fold_idx + 1}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise
    
    val_loss = val_metrics["loss"]
    val_acc = val_metrics["accuracy"]
    val_f1 = val_metrics["f1"]
    val_precision = val_metrics["precision"]
    val_recall = val_metrics["recall"]
    per_class = val_metrics["per_class"]
    
    try:
        model.eval()
        if device.type == "cuda" and is_hyper_aggressive:
            for _ in range(10):
                aggressive_gc(clear_cuda=True)
            for _ in range(5):
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        model_path = fold_output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metrics to DuckDB
        _save_metrics_to_duckdb(result, model_type, fold_idx, project_root_str)
        
        # Check Airflow status
        airflow_status = _check_airflow_status(model_type, project_root_str)
        if airflow_status:
            logger.debug(f"Airflow status: {airflow_status}")
        
        if device.type == "cuda" and is_hyper_aggressive:
            for _ in range(10):
                aggressive_gc(clear_cuda=True)
            for _ in range(10):
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            aggressive_gc(clear_cuda=True)
            torch.cuda.synchronize()
    except (OSError, IOError, PermissionError) as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        raise IOError(f"Cannot save model to {model_path}") from e
    
    if mlflow_tracker is not None:
        try:
            mlflow_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
            }
            for class_idx, metrics in per_class.items():
                mlflow_metrics[f"val_precision_class{class_idx}"] = metrics["precision"]
                mlflow_metrics[f"val_recall_class{class_idx}"] = metrics["recall"]
                mlflow_metrics[f"val_f1_class{class_idx}"] = metrics["f1"]
            
            mlflow_tracker.log_metrics(mlflow_metrics, step=fold_idx + 1)
            
            # Log artifact - check if file exists first
            if model_path.exists() and model_path.stat().st_size > 0:
                mlflow_tracker.log_artifact(str(model_path), artifact_path="models")
                logger.info(f"Logged model artifact to MLflow: {model_path.name}")
            else:
                logger.warning(f"Model file not found or empty, skipping artifact logging: {model_path}")
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to log to MLflow: {e}", exc_info=True)
    
    result = {
        "fold": fold_idx + 1,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1_class0": per_class.get("0", {}).get("f1", 0.0),
        "val_precision_class0": per_class.get("0", {}).get("precision", 0.0),
        "val_recall_class0": per_class.get("0", {}).get("recall", 0.0),
        "val_f1_class1": per_class.get("1", {}).get("f1", 0.0),
        "val_precision_class1": per_class.get("1", {}).get("precision", 0.0),
        "val_recall_class1": per_class.get("1", {}).get("recall", 0.0),
    }
    if hyperparams:
        result.update(hyperparams)
    
    logger.info(
        f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
        f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}"
    )
    _flush_logs()
    
    if per_class:
        for class_idx, metrics in per_class.items():
            logger.info(
                f"  Class {class_idx} - Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            )
    
    # Ensure final validation metrics are saved to metrics.jsonl even if training was skipped
    if tracker is not None:
        try:
            # Log final validation metrics (epoch 0 indicates final evaluation, not training epoch)
            tracker.log_epoch_metrics(
                0,  # Use epoch 0 to indicate final metrics
                {
                    "loss": val_loss,
                    "accuracy": val_acc,
                    "f1": val_f1,
                    "precision": val_precision,
                    "recall": val_recall,
                },
                phase="val"
            )
            logger.debug(f"Saved final validation metrics to {tracker.metrics_file}")
            
            # Save training curves plot if metrics.jsonl has epoch data
            try:
                metrics_df = tracker.load_metrics()
                if not metrics_df.is_empty():
                    # Extract train/val losses and accuracies per epoch
                    train_losses = []
                    val_losses = []
                    train_accs = []
                    val_accs = []
                    
                    # Get unique epochs (excluding epoch 0 which is final eval)
                    epochs = sorted([
                        int(e) for e in metrics_df["epoch"].unique().to_list()
                        if e > 0
                    ])
                    
                    for epoch in epochs:
                        epoch_data = metrics_df.filter(
                            (pl.col("epoch") == epoch)
                        )
                        
                        # Train metrics
                        train_data = epoch_data.filter(pl.col("phase") == "train")
                        if not train_data.is_empty():
                            train_loss = train_data.filter(
                                pl.col("metric") == "loss"
                            )["value"].to_list()
                            train_acc = train_data.filter(
                                pl.col("metric") == "accuracy"
                            )["value"].to_list()
                            if train_loss:
                                train_losses.append(train_loss[0])
                            if train_acc:
                                train_accs.append(train_acc[0])
                        
                        # Val metrics
                        val_data = epoch_data.filter(pl.col("phase") == "val")
                        if not val_data.is_empty():
                            val_loss_list = val_data.filter(
                                pl.col("metric") == "loss"
                            )["value"].to_list()
                            val_acc_list = val_data.filter(
                                pl.col("metric") == "accuracy"
                            )["value"].to_list()
                            if val_loss_list:
                                val_losses.append(val_loss_list[0])
                            if val_acc_list:
                                val_accs.append(val_acc_list[0])
                    
                    # Only plot if we have data
                    if train_losses or val_losses:
                        curve_path = fold_output_dir / "training_curves.png"
                        plot_learning_curves(
                            train_losses=train_losses if train_losses else [0.0] * len(val_losses),
                            val_losses=val_losses if val_losses else [0.0] * len(train_losses),
                            save_path=curve_path,
                            train_accs=train_accs if train_accs else None,
                            val_accs=val_accs if val_accs else None,
                            title=f"{model_type} - Fold {fold_idx + 1} Training Curves"
                        )
                        logger.info(f"Saved training curves to {curve_path}")
            except Exception as plot_error:
                logger.debug(f"Failed to save training curves: {plot_error}")
        except Exception as e:
            logger.debug(f"Failed to save final validation metrics: {e}")
    
    if mlflow_tracker is not None:
        try:
            run_id = mlflow_tracker.run_id
            mlflow_tracker.end_run()
            logger.info(f"MLflow run ended: run_id={run_id}, model={model_type}, fold={fold_idx + 1}")
        except (RuntimeError, AttributeError, ValueError) as cleanup_error:
            logger.warning(f"Error ending MLflow run: {cleanup_error}")
    
    # Save metrics to DuckDB
    _save_metrics_to_duckdb(result, model_type, fold_idx, project_root_str)
    
    # Check Airflow status
    airflow_status = _check_airflow_status(model_type, project_root_str)
    if airflow_status:
        logger.debug(f"Airflow status: {airflow_status}")
    
    # Generate plots for this fold
    try:
        from .visualization import plot_fold_metrics
        if "probs" in val_metrics and "preds" in val_metrics and "labels" in val_metrics:
            # Extract predictions and probabilities from val_metrics (already concatenated)
            val_probs = val_metrics["probs"]
            val_preds = val_metrics["preds"]
            val_y = val_metrics["labels"]
            
            plot_fold_metrics(
                y_true=val_y,
                y_pred=val_preds,
                y_probs=val_probs,
                fold_output_dir=fold_output_dir,
                fold_num=fold_idx + 1,
                model_type=model_type
            )
    except Exception as e:
        logger.warning(f"Failed to generate plots for PyTorch fold {fold_idx + 1}: {e}", exc_info=True)
    
    cleanup_model_and_memory(model=model, device=device, clear_cuda=device.type == "cuda")
    aggressive_gc(clear_cuda=device.type == "cuda")
    
    return result


def _train_baseline_model_fold(
    model_type: str,
    model_config: Dict[str, Any],
    train_df: Any,
    val_df: Any,
    project_root_str: str,
    fold_idx: int,
    model_output_dir: Path,
    features_stage2_path: str,
    features_stage4_path: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    is_grid_search: bool = False,
    param_fold_results: Optional[List[Dict[str, Any]]] = None,
    fold_results: Optional[List[Dict[str, Any]]] = None,
    use_tracking: bool = True,
    use_mlflow: bool = True,
    param_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Train a baseline (sklearn) model on a single fold.
    
    Args:
        model_type: Type of model to train
        model_config: Base model configuration
        train_df: Training dataframe
        val_df: Validation dataframe
        project_root_str: Project root as string
        fold_idx: Fold index (0-based)
        model_output_dir: Output directory for models
        features_stage2_path: Path to Stage 2 features
        features_stage4_path: Path to Stage 4 features
        hyperparams: Optional hyperparameters to apply (for grid search or final training)
        is_grid_search: Whether this is grid search (affects error handling and result storage)
        param_fold_results: Optional list to append grid search results to
        fold_results: Optional list to append results to
        use_tracking: Whether to use experiment tracking
        use_mlflow: Whether to use MLflow tracking
        param_idx: Optional parameter combination index for grid search
    
    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Training baseline model {model_type} on fold {fold_idx + 1}...")
    _flush_logs()
    
    fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tracker for metrics logging
    tracker = None
    if use_tracking:
        from lib.mlops.config import ExperimentTracker
        tracker = ExperimentTracker(str(fold_output_dir))
    
    result = None
    model = None
    mlflow_tracker = None
    
    # Initialize MLflow tracker if enabled
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            import mlflow
            if mlflow.active_run() is not None:
                mlflow.end_run()
                logger.debug("Ended existing MLflow run before creating new one")
        except (RuntimeError, AttributeError, ValueError) as mlflow_error:
            logger.debug(f"Error ending MLflow run (non-critical): {mlflow_error}")
        
        try:
            # Map model_type to MLflow experiment name using the same mapping as notebooks
            mlflow_model_type = MLFLOW_MODEL_TYPE_MAPPING.get(model_type, model_type)
            mlflow_tracker = create_mlflow_tracker(experiment_name=f"{mlflow_model_type}", use_mlflow=True)
            if mlflow_tracker:
                # CRITICAL: Log the run_id so we can map logs to MLflow runs
                logger.info(
                    f"MLflow run started: run_id={mlflow_tracker.run_id}, "
                    f"experiment={mlflow_model_type}, model={model_type}, "
                    f"fold={fold_idx + 1}, param_combo={param_idx + 1 if param_idx is not None else 'final'}"
                )
                mlflow_tracker.log_config(model_config)
                mlflow_tracker.set_tag("fold", str(fold_idx + 1))
                mlflow_tracker.set_tag("model_type", mlflow_model_type)
                if param_idx is not None:
                    mlflow_tracker.set_tag("param_combination", str(param_idx + 1))
                if hyperparams:
                    # Log hyperparameters as MLflow parameters
                    import mlflow
                    for key, value in hyperparams.items():
                        if isinstance(value, (str, int, float, bool)):
                            mlflow.log_param(key, value)
            else:
                logger.warning(f"MLflow tracker creation returned None for {model_type}")
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to create MLflow tracker: {e}")
    
    try:
        baseline_config = model_config.copy()
        if hyperparams:
            baseline_config.update(hyperparams)
        
        from lib.utils.paths import load_metadata_flexible
        
        if model_type in BASELINE_MODELS:
            stage2_df = load_metadata_flexible(features_stage2_path)
            if stage2_df is not None and stage2_df.height > 0:
                if "model_specific_config" not in baseline_config:
                    baseline_config["model_specific_config"] = {}
                baseline_config["model_specific_config"]["features_stage2_path"] = features_stage2_path
                baseline_config["features_stage2_path"] = features_stage2_path
                logger.debug(f"Passing Stage 2 features path to {model_type}: {features_stage2_path}")
            else:
                raise ValueError(
                    f"Stage 2 features are REQUIRED for {model_type}. "
                    f"Features must be pre-extracted in Stage 2. "
                    f"Stage 2 metadata not found or empty at: {features_stage2_path}. "
                    f"Please run Stage 2 feature extraction first."
                )
            
            if model_type in STAGE4_MODELS:
                stage4_df = load_metadata_flexible(features_stage4_path)
                if stage4_df is not None and stage4_df.height > 0:
                    if "model_specific_config" not in baseline_config:
                        baseline_config["model_specific_config"] = {}
                    baseline_config["model_specific_config"]["features_stage4_path"] = features_stage4_path
                    baseline_config["features_stage4_path"] = features_stage4_path
                    logger.debug(f"Passing Stage 4 features path to {model_type}: {features_stage4_path}")
                else:
                    raise ValueError(
                        f"Stage 4 features are REQUIRED for {model_type}. "
                        f"Features must be pre-extracted in Stage 4. "
                        f"Stage 4 metadata not found or empty at: {features_stage4_path}. "
                        f"Please run Stage 4 scaled feature extraction first."
                    )
            else:
                if "model_specific_config" not in baseline_config:
                    baseline_config["model_specific_config"] = {}
                baseline_config["model_specific_config"]["features_stage4_path"] = None
                baseline_config["features_stage4_path"] = None
        
        model = create_model(model_type, baseline_config)
        
        # Create tracker for epoch-wise training metrics (if use_tracking is enabled)
        tracker = None
        if use_tracking:
            try:
                tracker = ExperimentTracker(fold_output_dir)
                # Set tracker on model if it supports it (for epoch-wise training)
                if hasattr(model, 'tracker'):
                    model.tracker = tracker
                    logger.info(f"Enabled epoch-wise training tracking for {model_type} fold {fold_idx + 1}")
            except Exception as e:
                logger.debug(f"Could not create tracker for {model_type} fold {fold_idx + 1}: {e}")
        
        logger.info(f"Starting model.fit() for {model_type} fold {fold_idx + 1}...")
        logger.info(f"Training data: {train_df.height} rows")
        _flush_logs()
        
        try:
            # Pass output_dir to fit() for epoch-wise training metrics
            model.fit(train_df, project_root=project_root_str, output_dir=str(fold_output_dir))
            logger.info(f"Model.fit() completed successfully for fold {fold_idx + 1}")
            _flush_logs()
        except MemoryError as e:
            logger.error(f"Memory error during model.fit() for fold {fold_idx + 1}: {e}")
            raise
        except Exception as e:
            error_msg = str(e)
            if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                logger.critical(f"CRITICAL: Possible crash during model.fit() for fold {fold_idx + 1}: {e}")
                logger.critical("This may indicate a memory issue, corrupted data, or library incompatibility")
                logger.critical("Check log file for more details")
            raise
        
        # Evaluate on validation set
        logger.info(f"Starting model.predict() for {model_type} fold {fold_idx + 1}...")
        logger.info(f"Validation data: {val_df.height} rows")
        _flush_logs()
        
        try:
            val_probs = model.predict(val_df, project_root=project_root_str)
            logger.info(f"Model.predict() completed successfully for fold {fold_idx + 1}")
            _flush_logs()
        except MemoryError as e:
            logger.error(f"Memory error during model.predict() for fold {fold_idx + 1}: {e}")
            raise
        except Exception as e:
            error_msg = str(e)
            if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                logger.critical(f"CRITICAL: Possible crash during model.predict() for fold {fold_idx + 1}: {e}")
                logger.critical("This may indicate a memory issue, corrupted data, or library incompatibility")
                logger.critical("Check log file for more details")
            raise
        
        val_preds = np.argmax(val_probs, axis=1)
        val_labels = val_df["label"].to_list()
        label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
        val_y = np.array([label_map[label] for label in val_labels])
        
        metrics = compute_classification_metrics(
            y_true=val_y,
            y_pred=val_preds,
            y_probs=val_probs
        )
        
        # Store results
        result = {
            "fold": fold_idx + 1,
            "val_loss": metrics["val_loss"],
            "val_acc": metrics["val_acc"],
            "val_f1": metrics["val_f1"],
            "val_precision": metrics["val_precision"],
            "val_recall": metrics["val_recall"],
            "val_f1_class0": metrics["val_f1_class0"],
            "val_precision_class0": metrics["val_precision_class0"],
            "val_recall_class0": metrics["val_recall_class0"],
            "val_f1_class1": metrics["val_f1_class1"],
            "val_precision_class1": metrics["val_precision_class1"],
            "val_recall_class1": metrics["val_recall_class1"],
        }
        if hyperparams:
            result.update(hyperparams)
        
        if is_grid_search and param_fold_results is not None:
            param_fold_results.append(result)
        if fold_results is not None:
            fold_results.append(result)
        
        logger.info(
            f"Fold {fold_idx + 1} - Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.4f}, "
            f"Val F1: {metrics['val_f1']:.4f}, Val Precision: {metrics['val_precision']:.4f}, Val Recall: {metrics['val_recall']:.4f}"
        )
        if is_grid_search:
            logger.info(
                f"  Class 0 - Precision: {metrics['val_precision_class0']:.4f}, "
                f"Recall: {metrics['val_recall_class0']:.4f}, F1: {metrics['val_f1_class0']:.4f}"
            )
            logger.info(
                f"  Class 1 - Precision: {metrics['val_precision_class1']:.4f}, "
                f"Recall: {metrics['val_recall_class1']:.4f}, F1: {metrics['val_f1_class1']:.4f}"
            )
        
        # Save model
        model.save(str(fold_output_dir))
        logger.info(f"Saved baseline model to {fold_output_dir}")
        
        # Log metrics to MLflow
        if mlflow_tracker is not None:
            try:
                mlflow_metrics = {
                    "val_loss": metrics["val_loss"],
                    "val_acc": metrics["val_acc"],
                    "val_f1": metrics["val_f1"],
                    "val_precision": metrics["val_precision"],
                    "val_recall": metrics["val_recall"],
                    "val_f1_class0": metrics["val_f1_class0"],
                    "val_precision_class0": metrics["val_precision_class0"],
                    "val_recall_class0": metrics["val_recall_class0"],
                    "val_f1_class1": metrics["val_f1_class1"],
                    "val_precision_class1": metrics["val_precision_class1"],
                    "val_recall_class1": metrics["val_recall_class1"],
                }
                mlflow_tracker.log_metrics(mlflow_metrics, step=fold_idx + 1)
                
                # Log model artifact - find the saved model file
                model_files = list(fold_output_dir.glob("*.joblib"))
                if not model_files:
                    model_files = list(fold_output_dir.glob("*.pkl"))
                if model_files:
                    model_path = model_files[0]
                    if model_path.exists() and model_path.stat().st_size > 0:
                        mlflow_tracker.log_artifact(str(model_path), artifact_path="models")
                        logger.info(f"Logged model artifact to MLflow: {model_path.name}")
                else:
                    logger.debug(f"No model file found to log to MLflow in {fold_output_dir}")
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.error(f"Failed to log to MLflow: {e}", exc_info=True)
        
        # Save metrics to DuckDB
        _save_metrics_to_duckdb(result, model_type, fold_idx, project_root_str)
        
        # Check Airflow status
        airflow_status = _check_airflow_status(model_type, project_root_str)
        if airflow_status:
            logger.debug(f"Airflow status: {airflow_status}")
        
        # Generate plots for this fold
        try:
            from .visualization import plot_fold_metrics
            # Extract positive class probabilities for plotting
            if val_probs.ndim == 2 and val_probs.shape[1] == 2:
                y_proba_pos = val_probs[:, 1]
            else:
                y_proba_pos = val_probs.flatten() if val_probs.ndim > 1 else val_probs
            
            plot_fold_metrics(
                y_true=val_y,
                y_pred=val_preds,
                y_probs=val_probs,
                fold_output_dir=fold_output_dir,
                fold_num=fold_idx + 1,
                model_type=model_type
            )
            
            # Generate comprehensive additional plots
            from .visualization import (
                plot_calibration_curve, plot_per_class_metrics, 
                plot_error_analysis
            )
            
            # Calibration curve
            try:
                plot_calibration_curve(
                    y_true=val_y,
                    y_proba=val_probs,
                    save_path=fold_output_dir / "calibration_curve.png",
                    title=f"{model_type} - Fold {fold_idx + 1} - Calibration Curve"
                )
            except Exception as e:
                logger.debug(f"Failed to generate calibration curve: {e}")
            
            # Per-class metrics breakdown
            try:
                metrics_per_class = {
                    "Class 0": {
                        "precision": metrics.get("val_precision_class0", 0),
                        "recall": metrics.get("val_recall_class0", 0),
                        "f1": metrics.get("val_f1_class0", 0)
                    },
                    "Class 1": {
                        "precision": metrics.get("val_precision_class1", 0),
                        "recall": metrics.get("val_recall_class1", 0),
                        "f1": metrics.get("val_f1_class1", 0)
                    }
                }
                plot_per_class_metrics(
                    metrics_per_class=metrics_per_class,
                    save_path=fold_output_dir / "per_class_metrics.png",
                    title=f"{model_type} - Fold {fold_idx + 1} - Per-Class Metrics"
                )
            except Exception as e:
                logger.debug(f"Failed to generate per-class metrics plot: {e}")
            
            # Error analysis
            try:
                plot_error_analysis(
                    y_true=val_y,
                    y_pred=val_preds,
                    y_proba=val_probs,
                    save_path=fold_output_dir / "error_analysis.png",
                    title=f"{model_type} - Fold {fold_idx + 1} - Error Analysis"
                )
            except Exception as e:
                logger.debug(f"Failed to generate error analysis plot: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to generate plots for baseline fold {fold_idx + 1}: {e}", exc_info=True)
        
        # Log final validation metrics to metrics.jsonl
        if tracker is not None:
            try:
                tracker.log_epoch_metrics(
                    0,  # Use epoch 0 to indicate final metrics (baseline models don't have epochs)
                    {
                        "loss": metrics["val_loss"],
                        "accuracy": metrics["val_acc"],
                        "f1": metrics["val_f1"],
                        "precision": metrics["val_precision"],
                        "recall": metrics["val_recall"],
                    },
                    phase="val"
                )
                logger.debug(f"Saved final validation metrics to {tracker.metrics_file}")
            except Exception as e:
                logger.debug(f"Failed to save final validation metrics: {e}")
    
    except Exception as e:
        logger.error(
            f"Error training baseline fold {fold_idx + 1}: {e}",
            exc_info=True
        )
        result = {
            "fold": fold_idx + 1,
            "val_loss": float('nan'),
            "val_acc": float('nan'),
            "val_f1": float('nan'),
            "val_precision": float('nan'),
            "val_recall": float('nan'),
            "val_f1_class0": float('nan'),
            "val_precision_class0": float('nan'),
            "val_recall_class0": float('nan'),
            "val_f1_class1": float('nan'),
            "val_precision_class1": float('nan'),
            "val_recall_class1": float('nan'),
        }
        if hyperparams:
            result.update(hyperparams)
        if is_grid_search and param_fold_results is not None:
            param_fold_results.append(result)
        if fold_results is not None:
            fold_results.append(result)
    
    finally:
        # End MLflow run if active
        if mlflow_tracker is not None:
            try:
                run_id = mlflow_tracker.run_id
                mlflow_tracker.end_run()
                logger.info(f"MLflow run ended: run_id={run_id}, model={model_type}, fold={fold_idx + 1}")
            except (RuntimeError, AttributeError, ValueError) as cleanup_error:
                logger.warning(f"Error ending MLflow run: {cleanup_error}")
        
        cleanup_model_and_memory(model=model if model is not None else None, clear_cuda=False)
        aggressive_gc(clear_cuda=False)
    
    return result if result is not None else {}


def stage5_train_models(
    project_root: str,
    scaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: str,
    model_types: List[str],
    n_splits: int = 5,
    num_frames: int = 1000,
    output_dir: str = "data/stage5",
    use_tracking: bool = True,
    use_mlflow: bool = True,
    train_ensemble: bool = False,
    ensemble_method: str = "meta_learner",
    delete_existing: bool = False,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Stage 5: Train models using scaled videos and features.
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to scaled metadata (from Stage 3)
        features_stage2_path: Path to Stage 2 features metadata
        features_stage4_path: Path to Stage 4 features metadata
        model_types: List of model types to train
        n_splits: Number of k-fold splits
        num_frames: Number of frames per video
        output_dir: Directory to save training results
        use_tracking: Whether to use experiment tracking
        train_ensemble: Whether to train ensemble model after individual models (default: False)
        ensemble_method: Ensemble method - "meta_learner" or "weighted_average" (default: "meta_learner")
        delete_existing: If True, delete existing model checkpoints/results before regenerating (clean mode)
        resume: If True, skip training folds that already have saved models (resume mode, default: True)
    
    Returns:
        Dictionary of training results
    """
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True at pipeline start")
    
    # Additional PyTorch memory optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Applied global PyTorch memory optimizations at pipeline start")
    
    # Immediate logging to show function has started
    logger.info("=" * 80)
    logger.info("Stage 5: Model Training Pipeline Started")
    logger.info("=" * 80)
    logger.info("Model types: %s", model_types)
    logger.info("K-fold splits: %d", n_splits)
    logger.info("Frames per video: %d", num_frames)
    logger.info("Output directory: %s", output_dir)
    logger.info("Initializing pipeline...")
    _flush_logs()
    
    if not project_root or not isinstance(project_root, str):
        raise ValueError(f"project_root must be a non-empty string, got: {type(project_root)}")
    if not scaled_metadata_path or not isinstance(scaled_metadata_path, str):
        raise ValueError(f"scaled_metadata_path must be a non-empty string, got: {type(scaled_metadata_path)}")
    if not features_stage2_path or not isinstance(features_stage2_path, str):
        raise ValueError(f"features_stage2_path must be a non-empty string, got: {type(features_stage2_path)}")
    if not features_stage4_path or not isinstance(features_stage4_path, str):
        raise ValueError(f"features_stage4_path must be a non-empty string, got: {type(features_stage4_path)}")
    if not model_types or not isinstance(model_types, list) or len(model_types) == 0:
        raise ValueError(f"model_types must be a non-empty list, got: {type(model_types)}")
    if n_splits <= 0 or not isinstance(n_splits, int):
        raise ValueError(f"n_splits must be a positive integer, got: {n_splits}")
    if num_frames <= 0 or not isinstance(num_frames, int):
        raise ValueError(f"num_frames must be a positive integer, got: {num_frames}")
    if not isinstance(output_dir, str):
        raise ValueError(f"output_dir must be a string, got: {type(output_dir)}")
    
    # Convert project_root to Path and resolve it once (avoid variable shadowing)
    try:
        project_root_path = Path(project_root).resolve()
        if not project_root_path.exists():
            raise FileNotFoundError(f"Project root directory does not exist: {project_root_path}")
        if not project_root_path.is_dir():
            raise NotADirectoryError(f"Project root is not a directory: {project_root_path}")
    except (OSError, ValueError) as e:
        logger.error(f"Invalid project_root path: {project_root} - {e}")
        raise ValueError(f"Invalid project_root path: {project_root}") from e
    
    project_root_str = str(project_root_path)
    
    try:
        _ensure_lib_models_exists(project_root_path)
    except Exception as e:
        logger.error(f"Failed to ensure lib/models exists: {e}")
        raise
    
    try:
        output_dir_path = project_root_path / output_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise ValueError(f"Cannot create output directory: {output_dir}") from e
    
    output_dir = output_dir_path
    
    # CRITICAL: Validate all prerequisites before starting any training
    validation_results = _validate_stage5_prerequisites(
        project_root_path,
        scaled_metadata_path,
        features_stage2_path,
        features_stage4_path,
        model_types
    )
    
    if not validation_results["runnable_models"]:
        error_msg = (
            "✗ ERROR: None of the requested models can be run with the available data.\n"
            "Please check the validation summary above and re-run the required stages.\n\n"
            "Required stages:\n"
        )
        if not validation_results["stage3_available"]:
            error_msg += "  - Stage 3 (scaled videos): REQUIRED for all models\n"
        if any(m in BASELINE_MODELS for m in model_types) and not validation_results["stage2_available"]:
            error_msg += "  - Stage 2 (features): REQUIRED for all baseline models (svm, logistic_regression and variants)\n"
        if any("stage2_stage4" in m for m in model_types) and not validation_results["stage4_available"]:
            error_msg += "  - Stage 4 (scaled features): REQUIRED for *_stage2_stage4 models\n"
        raise FileNotFoundError(error_msg)
    
    # Filter to only runnable models
    if len(validation_results["runnable_models"]) < len(model_types):
        skipped = set(model_types) - set(validation_results["runnable_models"])
        logger.warning(
            f"\n⚠ WARNING: Skipping {len(skipped)} model(s) due to missing prerequisites: {skipped}\n"
            f"Will train {len(validation_results['runnable_models'])} model(s): {validation_results['runnable_models']}"
        )
        model_types = validation_results["runnable_models"]
    
    # Load metadata (support both CSV and Arrow/Parquet)
    logger.info("\nStage 5: Loading metadata...")
    
    from lib.utils.paths import load_metadata_flexible
    from lib.utils.data_integrity import DataIntegrityChecker
    from lib.utils.guardrails import ResourceMonitor, HealthCheckStatus, ResourceExhaustedError, DataIntegrityError
    
    metadata_path_obj = Path(scaled_metadata_path)
    is_valid, error_msg, scaled_df = DataIntegrityChecker.validate_metadata_file(
        metadata_path_obj,
        required_columns={'video_path', 'label'},
        min_rows=3000,
        allow_empty=False
    )
    
    if not is_valid:
        raise DataIntegrityError(f"Metadata validation failed: {error_msg}")
    
    if scaled_df is None:
        scaled_df = load_metadata_flexible(scaled_metadata_path)
        if scaled_df is None:
            raise FileNotFoundError(f"Scaled metadata not found: {scaled_metadata_path}")
    
    # CRITICAL: Verify dataframe has more than 3000 rows (double-check)
    num_rows = scaled_df.height
    logger.info(f"Loaded metadata: {num_rows} rows")
    if num_rows <= 3000:
        error_msg = (
            f"✗ ERROR: Insufficient data for training. "
            f"Expected more than 3000 rows, but got {num_rows} rows.\n"
            f"Please ensure Stage 3 completed successfully and generated enough scaled videos.\n"
            f"Metadata path: {scaled_metadata_path}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"✓ Data validation passed: {num_rows} rows (> 3000 required)")
    _flush_logs()
    
    monitor = ResourceMonitor()
    health = monitor.full_health_check(project_root_path)
    if health.status.value >= HealthCheckStatus.UNHEALTHY.value:
        logger.warning(f"System health check: {health.status.value} - {health.message}")
        if health.status == HealthCheckStatus.CRITICAL:
            raise ResourceExhaustedError(f"Critical system state: {health.message}")
    
    # Lazy loading: Only load features/videos if needed by any model
    from lib.training.video_training_pipeline import is_feature_based, is_video_based
    
    # Determine what to load based on model types
    needs_features = any(is_feature_based(m) or "xgboost" in m for m in model_types)
    needs_videos = any(is_video_based(m) for m in model_types)
    
    # Lazy load features only if any model requires them
    features2_df = None
    features4_df = None
    if needs_features:
        logger.info("Loading Stage 2 and Stage 4 features (required for feature-based models)...")
        features2_df = load_metadata_flexible(features_stage2_path)
        features4_df = load_metadata_flexible(features_stage4_path)
        if features2_df is None:
            logger.debug("Stage 2 features metadata not found (optional for some models)")
        if features4_df is None:
            logger.debug("Stage 4 features metadata not found (optional for some models)")
    else:
        logger.debug("Skipping feature loading - no feature-based models in training list")
    
    logger.info(f"Stage 5: Found {scaled_df.height} scaled videos")
    _flush_logs()
    
    logger.info("=" * 80)
    logger.info("STAGE 5: VALIDATING VIDEOS (checking for corruption and empty videos)")
    logger.info("=" * 80)
    from lib.data import filter_existing_videos
    try:
        scaled_df = filter_existing_videos(
            scaled_df, 
            project_root=project_root_str,
            check_frames=True,  # Check for videos with no frames
            check_corruption=True  # Check for corrupted videos (moov atom errors, etc.)
        )
        logger.info(f"✓ Video validation complete: {scaled_df.height} valid videos ready for training")
        logger.info("=" * 80)
        _flush_logs()
    except ValueError as e:
        error_msg = (
            f"✗ ERROR: Video validation failed. "
            f"Please check that scaled videos exist and are not corrupted.\n"
            f"Error: {str(e)}\n\n"
            f"Common issues:\n"
            f"  - Corrupted videos (moov atom not found): Re-run Stage 3 scaling\n"
            f"  - Videos with no frames: Check Stage 3 output\n"
            f"  - Missing video files: Verify data/scaled_videos/ directory"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    try:
        from lib.models import VideoConfig
    except ImportError as e:
        raise ImportError(
            f"CRITICAL: Cannot import VideoConfig from lib.models. "
            f"lib/models must be available for Stage 5. Error: {e}"
        ) from e
    
    # Create video config (will be used only for PyTorch models)
    # Always use scaled videos - augmentation done in Stage 1, scaling in Stage 3
    # Handle both old and new VideoConfig versions (some servers may not have use_scaled_videos yet)
    # For memory-intensive models (5c+), use chunked frame loading with adaptive sizing to prevent OOM
    # Adaptive chunk size starts at lower values for very memory-intensive models, reduces on OOM (multiplicative decrease), increases on success (additive increase)
    # Memory-intensive models that need chunked frame loading (5c+ scripts)
    # Models 5c-5l need very small initial chunk size (10) due to persistent OOM - consistent with forward pass chunk size
    MEMORY_INTENSIVE_MODELS_SMALL_CHUNK = [
        "naive_cnn",           # 5c - processes 1000 frames at full resolution, very memory intensive
        "pretrained_inception", # 5d - large pretrained model
        "variable_ar_cnn",     # 5e - variable-length videos with many frames
        "vit_gru",             # 5k - Vision Transformer with GRU
        "vit_transformer",     # 5l - Vision Transformer
    ]
    
    MEMORY_INTENSIVE_MODELS_ULTRA_SMALL_CHUNK = [
        "x3d",                 # 5q - extremely memory intensive, needs smallest chunk size
        "slowfast",            # 5r - dual pathway architecture, very memory intensive
    ]
    
    XGBOOST_PRETRAINED_MODELS = [
        "xgboost_pretrained_inception",  # 5f
        "xgboost_i3d",                    # 5g
        "xgboost_r2plus1d",               # 5h
        "xgboost_vit_gru",                # 5i
        "xgboost_vit_transformer",        # 5j
    ]
    
    MEMORY_INTENSIVE_MODELS = [
        "naive_cnn",           # 5c - processes 1000 frames at full resolution
        "pretrained_inception", # 5d - large pretrained model
        "variable_ar_cnn",     # 5e - variable-length videos with many frames
        "i3d",                 # 5o - 3D CNN
        "r2plus1d",            # 5p - 3D CNN
        "x3d",                 # 5q - very memory intensive
        "slowfast",            # 5r - dual-pathway architecture
        "vit_gru",             # 5k - Vision Transformer with GRU
        "vit_transformer",     # 5l - Vision Transformer
    ]
    use_chunked_loading = False
    chunk_size = None
    for model_type in model_types:
        if model_type in MEMORY_INTENSIVE_MODELS:
            use_chunked_loading = True
            # Use ultra-small chunk size (4) for X3D and SlowFast due to extreme memory requirements
            if model_type in MEMORY_INTENSIVE_MODELS_ULTRA_SMALL_CHUNK:
                chunk_size = 4  # Ultra-small chunk size for maximum OOM resistance (5q, 5r)
            # Use very small chunk size (10) for models 5c-5l that have persistent OOM issues
            elif model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK:
                chunk_size = 10  # Initial chunk size for very memory-intensive models (5c-5l) - consistent with forward pass, capped at 28
            else:
                chunk_size = 30  # Initial chunk size for other memory-intensive models (5o-5p)
            logger.info(
                f"Enabling adaptive chunked frame loading for {model_type}: "
                f"initial_chunk_size={chunk_size}, num_frames={num_frames}. "
                f"Chunk size will adapt automatically based on OOM events (AIMD algorithm)."
            )
            break
    
    use_frame_cache = os.environ.get("FVC_USE_FRAME_CACHE", "1") == "1"
    frame_cache_dir = os.environ.get("FVC_FRAME_CACHE_DIR", "data/.frame_cache")
    
    if use_frame_cache:
        logger.info(
            f"Frame caching enabled (default): cache_dir={frame_cache_dir}. "
            f"This will cache processed frames to disk to speed up training. "
            f"First epoch will be slower (building cache), subsequent epochs will be faster. "
            f"To disable, set FVC_USE_FRAME_CACHE=0"
        )
    else:
        logger.info(
            "Frame caching disabled (FVC_USE_FRAME_CACHE=0). "
            "Training will decode videos every epoch (slower but uses less disk space)."
        )
    
    try:
        if use_chunked_loading and chunk_size is not None:
            video_config = VideoConfig(
                num_frames=num_frames,
                fixed_size=256,
                use_scaled_videos=True,
                chunk_size=chunk_size,
                use_frame_cache=use_frame_cache,
                frame_cache_dir=frame_cache_dir if use_frame_cache else None
            )
        else:
            video_config = VideoConfig(
                num_frames=num_frames,
                fixed_size=256,
                use_scaled_videos=True,
                use_frame_cache=use_frame_cache,
                frame_cache_dir=frame_cache_dir if use_frame_cache else None
            )
    except TypeError:
        # Fallback: server version doesn't support these parameters
        logger.warning(
            "VideoConfig on server doesn't support 'use_scaled_videos', 'fixed_size', 'chunk_size', or frame_cache parameters. "
            "Using default VideoConfig and setting use_scaled_videos=True, fixed_size=256, and frame_cache manually (videos should already be scaled from Stage 3)."
        )
        video_config = VideoConfig(num_frames=num_frames)
        # CRITICAL: Set fixed_size=256 for consistent input dimensions
        video_config.fixed_size = 256
        # CRITICAL: Set use_scaled_videos=True even if constructor doesn't support it
        # Stage 5 ALWAYS uses scaled videos from Stage 3
        video_config.use_scaled_videos = True
        logger.info("Manually set fixed_size=256, use_scaled_videos=True on VideoConfig (server version fallback)")
        # CRITICAL: Set frame_cache parameters even if constructor doesn't support them
        # Frame caching is enabled by default to speed up training
        if use_frame_cache:
            video_config.use_frame_cache = True
            video_config.frame_cache_dir = frame_cache_dir
            logger.info(f"Manually set use_frame_cache=True, frame_cache_dir={frame_cache_dir} on VideoConfig (server version fallback)")
    
    if not getattr(video_config, 'use_scaled_videos', False):
        logger.warning(
            "CRITICAL: use_scaled_videos is False in VideoConfig for Stage 5! "
            "This should NEVER happen - Stage 5 always uses scaled videos from Stage 3. "
            "Forcing use_scaled_videos=True."
        )
        video_config.use_scaled_videos = True
        logger.info("Forced use_scaled_videos=True on VideoConfig (Stage 5 requirement)")
    
    # CRITICAL: Verify fixed_size is set to 256 for consistent input dimensions
    # This ensures all models get consistent input size, even when using scaled videos
    if getattr(video_config, 'fixed_size', None) != 256:
        logger.warning(
            f"fixed_size is not 256 in VideoConfig (got {getattr(video_config, 'fixed_size', None)}). "
            f"Setting fixed_size=256."
        )
        video_config.fixed_size = 256
        logger.info("Forced fixed_size=256 on VideoConfig")
    
    for model_type in model_types:
        if model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK or model_type in XGBOOST_PRETRAINED_MODELS:
            target_frames = 500 if model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK else 400
            video_config.num_frames = target_frames
            logger.info(
                f"Overriding num_frames to {target_frames} for {model_type}. Original num_frames was {num_frames}."
            )
            break
    
    results = {}
    
    def _is_fold_complete(fold_dir: Path, model_type: str) -> bool:
        """Check if a fold directory contains a complete trained model."""
        if not fold_dir.exists():
            return False
        
        # PyTorch models: check for model.pt
        if is_pytorch_model(model_type):
            model_file = fold_dir / "model.pt"
            if model_file.exists() and model_file.stat().st_size > 0:
                return True
        
        model_files = list(fold_dir.glob("model.*"))
        if model_files:
            for model_file in model_files:
                if model_file.stat().st_size > 0:
                    return True
        
        return False
    
    # Delete existing model results if clean mode
    if delete_existing:
        logger.info("Stage 5: Deleting existing model results (clean mode)...")
        deleted_count = 0
        for model_type in model_types:
            model_output_dir = output_dir / model_type
            if model_output_dir.exists():
                try:
                    shutil.rmtree(model_output_dir)
                    deleted_count += 1
                    logger.info(f"Deleted existing results for {model_type}")
                except (OSError, PermissionError, FileNotFoundError) as e:
                    logger.warning(f"Could not delete {model_output_dir}: {e}")
        logger.info(f"Stage 5: Deleted {deleted_count} existing model directories")
        _flush_logs()
    
    for model_type in model_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage 5: Training model: {model_type}")
        logger.info(f"{'='*80}")
        _flush_logs()
        
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        model_config = get_model_config(model_type)
        
        if model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK:
            model_config["num_frames"] = 500
            logger.info(
                f"Overriding model_config num_frames to 500 for {model_type} "
                f"(small-chunk model) to match video_config."
            )
        elif model_type in XGBOOST_PRETRAINED_MODELS:
            # ARCHITECTURAL IMPROVEMENT: Use 400 frames for enhanced feature extraction
            # Enhanced multi-layer + temporal pooling allows more frames while staying within memory
            model_config["num_frames"] = 400  # Increased from 250 to 400 for better temporal coverage
            logger.info(
                f"Overriding model_config num_frames to 400 for {model_type} "
                f"(XGBoost pretrained model with enhanced feature extraction) to improve feature quality."
            )
        
        # K-fold cross-validation
        fold_results = []
        
        # CRITICAL: Enforce 5-fold stratified cross-validation
        if n_splits != 5:
            logger.warning(f"n_splits={n_splits} specified, but enforcing 5-fold CV as required")
            n_splits = 5
        
        from .grid_search import get_hyperparameter_grid, generate_parameter_combinations, select_best_hyperparameters
        
        param_grid = get_hyperparameter_grid(model_type)
        param_combinations = generate_parameter_combinations(param_grid)
        
        logger.info(f"Grid search: {len(param_combinations)} hyperparameter combinations to try")
        _flush_logs()
        
        from sklearn.model_selection import StratifiedShuffleSplit
        
        grid_search_sample_size = float(os.environ.get("FVC_GRID_SEARCH_SAMPLE_SIZE", "0.1"))
        grid_search_sample_size = max(0.05, min(0.5, grid_search_sample_size))
        
        logger.info("=" * 80)
        logger.info(f"HYPERPARAMETER SEARCH: Using {grid_search_sample_size*100:.1f}% stratified sample for efficiency")
        logger.info("=" * 80)
        
        labels = scaled_df["label"].to_list()
        test_size = 1.0 - grid_search_sample_size
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        sample_indices, _ = next(sss.split(scaled_df, labels))
        
        # Create sample DataFrame
        sample_df = scaled_df[sample_indices]
        logger.info(f"Hyperparameter search sample: {sample_df.height} rows ({100.0 * sample_df.height / scaled_df.height:.1f}% of {scaled_df.height} total)")
        logger.info(f"To change sample size, set FVC_GRID_SEARCH_SAMPLE_SIZE environment variable (current: {grid_search_sample_size})")
        _flush_logs()
        
        grid_search_folds = stratified_kfold(
            sample_df,
            n_splits=n_splits,
            random_state=42
        )
        
        if len(grid_search_folds) != n_splits:
            raise ValueError(f"Expected {n_splits} folds for grid search, got {len(grid_search_folds)}")
        
        logger.info(f"Using {n_splits}-fold stratified cross-validation on {grid_search_sample_size*100:.1f}% sample for hyperparameter search")
        _flush_logs()
        
        all_grid_results = []
        
        if not param_combinations:
            # No grid search, use default config
            param_combinations = [{}]
            logger.info("No hyperparameter grid defined, using default configuration")
        
        for param_idx, params in enumerate(param_combinations):
            logger.info(f"\n{'='*80}")
            logger.info(f"Grid Search: Hyperparameter combination {param_idx + 1}/{len(param_combinations)}")
            logger.info(f"Parameters: {params}")
            logger.info(f"{'='*80}")
            _flush_logs()
            
            current_config = model_config.copy()
            params_to_apply = params.copy()
            if model_type in MEMORY_INTENSIVE_MODELS_BATCH_LIMITS and "batch_size" in params_to_apply:
                max_batch = MEMORY_INTENSIVE_MODELS_BATCH_LIMITS[model_type]
                if params_to_apply["batch_size"] > max_batch:
                    logger.debug(
                        f"Removing batch_size={params_to_apply['batch_size']} from grid search params for {model_type} "
                        f"(will be capped at {max_batch})"
                    )
                    del params_to_apply["batch_size"]
            current_config.update(params_to_apply)
            
            param_fold_results = []
            
            for fold_idx in range(n_splits):
                logger.info(f"\nHyperparameter Search - {model_type} - Fold {fold_idx + 1}/{n_splits} ({grid_search_sample_size*100:.1f}% sample)")
                _flush_logs()
                
                # Delete existing fold if delete_existing is True (do this BEFORE checking if complete)
                fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                if delete_existing and fold_output_dir.exists():
                    try:
                        shutil.rmtree(fold_output_dir)
                        logger.info(f"Deleted existing hyperparameter search fold {fold_idx + 1} directory (clean mode)")
                        _flush_logs()
                    except (OSError, PermissionError, FileNotFoundError) as e:
                        logger.warning(f"Could not delete {fold_output_dir}: {e}")
                        _flush_logs()
                
                if resume and not delete_existing and _is_fold_complete(fold_output_dir, model_type):
                    logger.info(f"Fold {fold_idx + 1} already trained (found existing model). Skipping.")
                    logger.info(f"To retrain this fold, use --delete-existing flag")
                    # Try to load existing results and log metrics if tracking is enabled
                    if use_tracking:
                        try:
                            from lib.mlops.config import ExperimentTracker
                            tracker = ExperimentTracker(str(fold_output_dir))
                            # Try to load existing metrics from metadata.json if available
                            metadata_file = fold_output_dir / "metadata.json"
                            if metadata_file.exists():
                                import json
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                # Log final validation metrics (epoch 0 indicates final metrics)
                                tracker.log_epoch_metrics(
                                    0,
                                    {
                                        "loss": metadata.get("val_loss", 0.0),
                                        "accuracy": metadata.get("val_acc", 0.0),
                                        "f1": metadata.get("val_f1", 0.0),
                                        "precision": metadata.get("val_precision", 0.0),
                                        "recall": metadata.get("val_recall", 0.0),
                                    },
                                    phase="val"
                                )
                                logger.debug(f"Saved existing validation metrics to {tracker.metrics_file}")
                        except Exception as e:
                            logger.debug(f"Failed to load and log existing metrics: {e}")
                    continue
                
                train_df, val_df = grid_search_folds[fold_idx]
            
                if "dup_group" in scaled_df.columns:
                    train_groups = set(train_df["dup_group"].unique().to_list())
                    val_groups = set(val_df["dup_group"].unique().to_list())
                    overlap = train_groups & val_groups
                    if overlap:
                        logger.error(
                            f"CRITICAL: Data leakage detected in fold {fold_idx + 1}! "
                            f"{len(overlap)} duplicate groups appear in both train and val: {list(overlap)[:5]}"
                        )
                        raise ValueError(f"Data leakage: duplicate groups in both train and val sets")
                    logger.info(f"Fold {fold_idx + 1}: No data leakage (checked {len(train_groups)} train groups, {len(val_groups)} val groups)")
                
                # Train model
                try:
                    if is_pytorch_model(model_type):
                        # Use extracted helper function for PyTorch model training
                        result = _train_pytorch_model_fold(
                            model_type=model_type,
                            model_config=model_config,
                            train_df=train_df,
                            val_df=val_df,
                            project_root_str=project_root_str,
                            fold_idx=fold_idx,
                            model_output_dir=model_output_dir,
                            video_config=video_config,
                            hyperparams=params,
                            is_grid_search=True,
                            param_fold_results=param_fold_results,
                            fold_results=fold_results,
                            use_tracking=use_tracking,
                            use_mlflow=use_mlflow,
                            param_idx=param_idx,
                        )
                    
                    elif is_xgboost_model(model_type):
                        # XGBoost model training (uses pretrained models for feature extraction)
                        _train_xgboost_model_fold(
                            model_type=model_type,
                            model_config=model_config,
                            train_df=train_df,
                            val_df=val_df,
                            project_root_str=project_root_str,
                            fold_idx=fold_idx,
                            model_output_dir=model_output_dir,
                            hyperparams=params,
                            is_grid_search=True,
                            param_fold_results=param_fold_results,
                            fold_results=fold_results,
                            use_mlflow=use_mlflow,
                            param_idx=param_idx
                        )
                    
                    else:
                        # Baseline model training (sklearn)
                        _train_baseline_model_fold(
                            model_type=model_type,
                            model_config=model_config,
                            train_df=train_df,
                            val_df=val_df,
                            project_root_str=project_root_str,
                            fold_idx=fold_idx,
                            model_output_dir=model_output_dir,
                            features_stage2_path=features_stage2_path,
                            features_stage4_path=features_stage4_path,
                            hyperparams=params,
                            is_grid_search=True,
                            param_fold_results=param_fold_results,
                            fold_results=fold_results,
                            use_tracking=use_tracking,
                            use_mlflow=use_mlflow,
                            param_idx=param_idx
                        )
                    
                except Exception as e:
                    logger.error(f"Error training fold {fold_idx + 1}: {e}", exc_info=True)
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": float('nan'),
                        "val_acc": float('nan'),
                        "val_f1": float('nan'),
                        "val_precision": float('nan'),
                        "val_recall": float('nan'),
                        "val_f1_class0": float('nan'),
                        "val_precision_class0": float('nan'),
                        "val_recall_class0": float('nan'),
                        "val_f1_class1": float('nan'),
                        "val_precision_class1": float('nan'),
                        "val_recall_class1": float('nan'),
                    })
                finally:
                    # Always cleanup resources, even on error
                    # End MLflow run if active
                    if 'mlflow_tracker' in locals() and mlflow_tracker is not None:
                        try:
                            mlflow_tracker.end_run()
                        except Exception as e:
                            logger.debug(f"Error ending MLflow run: {e}")
                    
                    # Clear model and aggressively free memory
                    device_obj = device if 'device' in locals() else None
                    cleanup_model_and_memory(
                        model=model if 'model' in locals() else None,
                        device=device_obj,
                        clear_cuda=device_obj.type == "cuda" if device_obj and device_obj.type == "cuda" else False
                    )
                    aggressive_gc(clear_cuda=device_obj.type == "cuda" if device_obj and device_obj.type == "cuda" else False)
            
            # After all folds for this parameter combination, aggregate results
            if param_fold_results:
                mean_f1 = np.mean([r.get("val_f1", 0) for r in param_fold_results if not np.isnan(r.get("val_f1", 0))])
                mean_acc = np.mean([r.get("val_acc", 0) for r in param_fold_results if not np.isnan(r.get("val_acc", 0))])
                grid_result = {
                    "mean_f1": mean_f1,
                    "mean_acc": mean_acc,
                    "fold_results": param_fold_results
                }
                grid_result.update(params)  # Include hyperparameters
                all_grid_results.append(grid_result)
                logger.info(f"Parameter combination {param_idx + 1} - Mean F1: {mean_f1:.4f}, Mean Acc: {mean_acc:.4f}")
                _flush_logs()
        
        # Select best hyperparameters from grid search (on sample)
        best_params = None
        if param_combinations and all_grid_results and len(all_grid_results) > 1:
            best_params = select_best_hyperparameters(model_type, all_grid_results)
            logger.info(f"Best hyperparameters selected from {grid_search_sample_size*100:.1f}% sample: {best_params}")
            _flush_logs()
        elif param_combinations and len(param_combinations) == 1:
            # Single parameter combination - use it
            best_params = param_combinations[0]
            logger.info(f"Using single parameter combination: {best_params}")
        
        # FINAL TRAINING: Train on full dataset with best hyperparameters
        logger.info("=" * 80)
        logger.info("FINAL TRAINING: Using full dataset with best hyperparameters")
        logger.info("=" * 80)
        _flush_logs()
        
        # For baseline models, create test set split BEFORE final training
        # to avoid data leakage (test set should be held out)
        test_df = None
        trainval_df = scaled_df
        if model_type in BASELINE_MODELS:
            try:
                from lib.data.loading import train_val_test_split, SplitConfig
                split_config = SplitConfig(val_size=0.0, test_size=0.2, random_state=42)
                splits = train_val_test_split(scaled_df, split_config)
                trainval_df = splits["train"]  # 80% for train+val (CV)
                test_df = splits["test"]  # 20% for test (held out)
                logger.info(f"Test set created: {test_df.height} rows (held out for final evaluation)")
                logger.info(f"Train+Val set: {trainval_df.height} rows (for CV)")
                _flush_logs()
            except Exception as e:
                logger.warning(f"Failed to create test set split: {e}. Using full dataset for CV.")
                test_df = None
                trainval_df = scaled_df
        
        # Create folds from train+val dataset for final training
        all_folds = stratified_kfold(
            trainval_df,
            n_splits=n_splits,
            random_state=42
        )
        
        if len(all_folds) != n_splits:
            raise ValueError(f"Expected {n_splits} folds for final training, got {len(all_folds)}")
        
        logger.info(f"Final training: Using {n_splits}-fold stratified cross-validation on train+val dataset ({trainval_df.height} rows)")
        _flush_logs()
        
        # Train on full dataset with best hyperparameters
        fold_results = []
        final_config = model_config.copy()
        if best_params:
            final_config.update(best_params)
            logger.info(f"Final training using best hyperparameters: {best_params}")
        else:
            logger.info("Final training using default hyperparameters (no grid search)")
        
        # Train all folds on full dataset with best hyperparameters
        for fold_idx in range(n_splits):
            logger.info(f"\nFinal Training - {model_type} - Fold {fold_idx + 1}/{n_splits} (full dataset)")
            _flush_logs()
            
            # Delete existing fold if delete_existing is True (do this BEFORE checking if complete)
            fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
            if delete_existing and fold_output_dir.exists():
                try:
                    shutil.rmtree(fold_output_dir)
                    logger.info(f"Deleted existing final training fold {fold_idx + 1} directory (clean mode)")
                    _flush_logs()
                except (OSError, PermissionError, FileNotFoundError) as e:
                    logger.warning(f"Could not delete {fold_output_dir}: {e}")
                    _flush_logs()
            
            # Check if fold is already complete (resume mode) - only if not deleting
            if resume and not delete_existing and _is_fold_complete(fold_output_dir, model_type):
                logger.info(f"Final training fold {fold_idx + 1} already trained (found existing model). Skipping.")
                logger.info(f"To retrain this fold, use --delete-existing flag")
                # Try to load existing results and log metrics if tracking is enabled
                if use_tracking:
                    try:
                        from lib.mlops.config import ExperimentTracker
                        tracker = ExperimentTracker(str(fold_output_dir))
                        # Try to load existing metrics from metadata.json if available
                        metadata_file = fold_output_dir / "metadata.json"
                        if metadata_file.exists():
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            # Log final validation metrics (epoch 0 indicates final metrics)
                            tracker.log_epoch_metrics(
                                0,
                                {
                                    "loss": metadata.get("val_loss", 0.0),
                                    "accuracy": metadata.get("val_acc", 0.0),
                                    "f1": metadata.get("val_f1", 0.0),
                                    "precision": metadata.get("val_precision", 0.0),
                                    "recall": metadata.get("val_recall", 0.0),
                                },
                                phase="val"
                            )
                            logger.debug(f"Saved existing validation metrics to {tracker.metrics_file}")
                    except Exception as e:
                        logger.debug(f"Failed to load and log existing metrics: {e}")
                continue
            
            # Get the specific fold from full dataset
            train_df, val_df = all_folds[fold_idx]
            
            # Validate no data leakage (check dup_group if present)
            if "dup_group" in scaled_df.columns:
                train_groups = set(train_df["dup_group"].unique().to_list())
                val_groups = set(val_df["dup_group"].unique().to_list())
                overlap = train_groups & val_groups
                if overlap:
                    logger.error(
                        f"CRITICAL: Data leakage detected in fold {fold_idx + 1}! "
                        f"{len(overlap)} duplicate groups appear in both train and val: {list(overlap)[:5]}"
                    )
                    raise ValueError(f"Data leakage: duplicate groups in both train and val sets")
                logger.info(f"Fold {fold_idx + 1}: No data leakage (checked {len(train_groups)} train groups, {len(val_groups)} val groups)")
            
            # Train model with best hyperparameters (reuse same training code as grid search)
            try:
                if is_pytorch_model(model_type):
                    # Use extracted helper function for PyTorch model training
                    result = _train_pytorch_model_fold(
                        model_type=model_type,
                        model_config=model_config,  # Base config
                        train_df=train_df,
                        val_df=val_df,
                        project_root_str=project_root_str,
                        fold_idx=fold_idx,
                        model_output_dir=model_output_dir,
                        video_config=video_config,
                        hyperparams=best_params,  # Best params from grid search
                        is_grid_search=False,
                        param_fold_results=None,
                        fold_results=fold_results,
                        use_tracking=use_tracking,
                        use_mlflow=use_mlflow,
                        param_idx=None,
                    )
                    
                elif is_xgboost_model(model_type):
                    # XGBoost model training with best hyperparameters
                    _train_xgboost_model_fold(
                        model_type=model_type,
                        model_config=model_config,
                        train_df=train_df,
                        val_df=val_df,
                        project_root_str=project_root_str,
                        fold_idx=fold_idx,
                        model_output_dir=model_output_dir,
                        hyperparams=best_params,
                        is_grid_search=False,
                        fold_results=fold_results,
                        use_mlflow=use_mlflow,
                        param_idx=None
                    )
                    
                else:
                    # Baseline model training with best hyperparameters
                    _train_baseline_model_fold(
                        model_type=model_type,
                        model_config=model_config,
                        train_df=train_df,
                        val_df=val_df,
                        project_root_str=project_root_str,
                        fold_idx=fold_idx,
                        model_output_dir=model_output_dir,
                        features_stage2_path=features_stage2_path,
                        features_stage4_path=features_stage4_path,
                        hyperparams=best_params,
                        is_grid_search=False,
                        fold_results=fold_results,
                        use_tracking=use_tracking,
                        use_mlflow=use_mlflow,
                        param_idx=None
                    )
                    
            except Exception as e:
                logger.error(f"Error training final fold {fold_idx + 1}: {e}", exc_info=True)
                fold_results.append({
                    "fold": fold_idx + 1,
                    "val_loss": float('nan'),
                    "val_acc": float('nan'),
                    "val_f1": float('nan'),
                    "val_precision": float('nan'),
                    "val_recall": float('nan'),
                    "val_f1_class0": float('nan'),
                    "val_precision_class0": float('nan'),
                    "val_recall_class0": float('nan'),
                    "val_f1_class1": float('nan'),
                    "val_precision_class1": float('nan'),
                    "val_recall_class1": float('nan'),
                })
        
        # Save best model from final training
        if fold_results:
            best_fold = max(fold_results, key=lambda x: x.get("val_f1", 0) if isinstance(x.get("val_f1"), (int, float)) and not np.isnan(x.get("val_f1", 0)) else -1)
            best_fold_idx = best_fold.get("fold", 1)
            
            best_model_dir = model_output_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            
            best_fold_dir = model_output_dir / f"fold_{best_fold_idx}"
            if best_fold_dir.exists():
                try:
                    _copy_model_files(best_fold_dir, best_model_dir, f"fold {best_fold_idx}")
                except Exception as e:
                    logger.error(f"Failed to copy best model files: {e}")
        
        # Aggregate results (filter out NaN values) - use fold_results from final training
        if fold_results:
            def get_valid_values(key: str):
                """Extract valid (non-NaN) values for a metric key."""
                return [
                    r.get(key) for r in fold_results
                    if key in r and isinstance(r.get(key), (int, float))
                    and not (isinstance(r.get(key), float)
                             and r.get(key) != r.get(key))  # Check for NaN
                ]
            
            def compute_mean_std(values):
                """Compute mean and std for a list of values."""
                if not values:
                    return float('nan'), float('nan')
                mean_val = sum(values) / len(values)
                if len(values) > 1:
                    std_val = np.std(values)
                else:
                    std_val = 0.0
                return mean_val, std_val
            
            # Aggregate all metrics
            valid_losses = get_valid_values("val_loss")
            valid_accs = get_valid_values("val_acc")
            valid_f1s = get_valid_values("val_f1")
            valid_precisions = get_valid_values("val_precision")
            valid_recalls = get_valid_values("val_recall")
            valid_f1_class0 = get_valid_values("val_f1_class0")
            valid_precision_class0 = get_valid_values("val_precision_class0")
            valid_recall_class0 = get_valid_values("val_recall_class0")
            valid_f1_class1 = get_valid_values("val_f1_class1")
            valid_precision_class1 = get_valid_values("val_precision_class1")
            valid_recall_class1 = get_valid_values("val_recall_class1")
            
            avg_val_loss, std_val_loss = compute_mean_std(valid_losses)
            avg_val_acc, std_val_acc = compute_mean_std(valid_accs)
            avg_val_f1, std_val_f1 = compute_mean_std(valid_f1s)
            avg_val_precision, std_val_precision = compute_mean_std(valid_precisions)
            avg_val_recall, std_val_recall = compute_mean_std(valid_recalls)
            avg_val_f1_class0, std_val_f1_class0 = compute_mean_std(valid_f1_class0)
            avg_val_precision_class0, std_val_precision_class0 = compute_mean_std(valid_precision_class0)
            avg_val_recall_class0, std_val_recall_class0 = compute_mean_std(valid_recall_class0)
            avg_val_f1_class1, std_val_f1_class1 = compute_mean_std(valid_f1_class1)
            avg_val_precision_class1, std_val_precision_class1 = compute_mean_std(valid_precision_class1)
            avg_val_recall_class1, std_val_recall_class1 = compute_mean_std(valid_recall_class1)
            
            results[model_type] = {
                "fold_results": fold_results,
                # Aggregated overall metrics
                "avg_val_loss": float(avg_val_loss),
                "std_val_loss": float(std_val_loss),
                "avg_val_acc": float(avg_val_acc),
                "std_val_acc": float(std_val_acc),
                "avg_val_f1": float(avg_val_f1),
                "std_val_f1": float(std_val_f1),
                "avg_val_precision": float(avg_val_precision),
                "std_val_precision": float(std_val_precision),
                "avg_val_recall": float(avg_val_recall),
                "std_val_recall": float(std_val_recall),
                # Aggregated per-class metrics
                "avg_val_f1_class0": float(avg_val_f1_class0),
                "std_val_f1_class0": float(std_val_f1_class0),
                "avg_val_precision_class0": float(avg_val_precision_class0),
                "std_val_precision_class0": float(std_val_precision_class0),
                "avg_val_recall_class0": float(avg_val_recall_class0),
                "std_val_recall_class0": float(std_val_recall_class0),
                "avg_val_f1_class1": float(avg_val_f1_class1),
                "std_val_f1_class1": float(std_val_f1_class1),
                "avg_val_precision_class1": float(avg_val_precision_class1),
                "std_val_precision_class1": float(std_val_precision_class1),
                "avg_val_recall_class1": float(avg_val_recall_class1),
                "std_val_recall_class1": float(std_val_recall_class1),
                "best_hyperparameters": best_params,
            }
            
            logger.info(
                "\n%s - Avg Val Loss: %.4f ± %.4f, Avg Val Acc: %.4f ± %.4f, Avg Val F1: %.4f ± %.4f",
                model_type, avg_val_loss, std_val_loss, avg_val_acc, std_val_acc, avg_val_f1, std_val_f1
            )
            logger.info(
                "  Avg Val Precision: %.4f ± %.4f, Avg Val Recall: %.4f ± %.4f",
                avg_val_precision, std_val_precision, avg_val_recall, std_val_recall
            )
            logger.info(
                "  Class 0 - F1: %.4f ± %.4f, Precision: %.4f ± %.4f, Recall: %.4f ± %.4f",
                avg_val_f1_class0, std_val_f1_class0, avg_val_precision_class0, std_val_precision_class0,
                avg_val_recall_class0, std_val_recall_class0
            )
            logger.info(
                "  Class 1 - F1: %.4f ± %.4f, Precision: %.4f ± %.4f, Recall: %.4f ± %.4f",
                avg_val_f1_class1, std_val_f1_class1, avg_val_precision_class1, std_val_precision_class1,
                avg_val_recall_class1, std_val_recall_class1
            )
            _flush_logs()
            
            # Save aggregated metrics to metrics.json
            # CRITICAL: Ensure all required metrics are present before saving
            # This is especially important for XGBoost models (5f-5j) and PyTorch models (5k-5r)
            try:
                import json
                metrics_file = model_output_dir / "metrics.json"
                
                # Validate that all expected metrics are present
                expected_metrics = [
                    "avg_val_loss", "std_val_loss",
                    "avg_val_acc", "std_val_acc",
                    "avg_val_f1", "std_val_f1",
                    "avg_val_precision", "std_val_precision",
                    "avg_val_recall", "std_val_recall",
                    "avg_val_f1_class0", "std_val_f1_class0",
                    "avg_val_precision_class0", "std_val_precision_class0",
                    "avg_val_recall_class0", "std_val_recall_class0",
                    "avg_val_f1_class1", "std_val_f1_class1",
                    "avg_val_precision_class1", "std_val_precision_class1",
                    "avg_val_recall_class1", "std_val_recall_class1",
                ]
                
                missing_metrics = [m for m in expected_metrics if m not in results[model_type]]
                if missing_metrics:
                    logger.warning(
                        f"Missing metrics for {model_type}: {missing_metrics}. "
                        f"This may indicate an issue with metric computation."
                    )
                
                with open(metrics_file, "w") as f:
                    json.dump(results[model_type], f, indent=2, default=str)
                logger.info(f"Saved aggregated metrics to {metrics_file} (model: {model_type})")
                
                # Save aggregated metrics to DuckDB
                try:
                    _save_metrics_to_duckdb(
                        results[model_type],
                        model_type,
                        -1,  # -1 indicates aggregated metrics
                        project_root_str
                    )
                except Exception as e:
                    logger.debug(f"Failed to save aggregated metrics to DuckDB: {e}")
                
                # Check Airflow status
                airflow_status = _check_airflow_status(model_type, project_root_str)
                if airflow_status:
                    logger.debug(f"Airflow status for {model_type}: {airflow_status}")
            except Exception as e:
                logger.error(f"Failed to save metrics.json for {model_type}: {e}", exc_info=True)
                # Don't raise - continue with other models
            
            # Generate visualization plots
            try:
                plots_dir = model_output_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate CV fold comparison
                from .visualization import plot_cv_fold_comparison, plot_hyperparameter_search
                
                plot_cv_fold_comparison(
                    fold_results,
                    plots_dir / "cv_fold_comparison.png",
                    title=f"{model_type} - Cross-Validation Results (Full Dataset)"
                )
                
                # Plot hyperparameter search results if grid search was performed
                if all_grid_results and len(all_grid_results) > 1:
                    plot_hyperparameter_search(
                        all_grid_results,
                        plots_dir / "hyperparameter_search.png",
                        title=f"{model_type} - Hyperparameter Search Results"
                    )
                
                logger.info(f"Generated plots for {model_type} in {plots_dir}")
                _flush_logs()
            except Exception as e:
                logger.warning(f"Failed to generate plots for {model_type}: {e}", exc_info=True)
            
            # For baseline models, evaluate on test set and save results.json
            if model_type in BASELINE_MODELS and fold_results and test_df is not None:
                try:
                    logger.info(f"Evaluating {model_type} on held-out test set...")
                    _flush_logs()
                    
                    logger.info(f"Test set size: {test_df.height} rows")
                    _flush_logs()
                    
                    # Load best model
                    best_model_dir = model_output_dir / "best_model"
                    if not best_model_dir.exists():
                        # Fallback: use fold_1 if best_model doesn't exist
                        best_model_dir = model_output_dir / "fold_1"
                        logger.warning(f"best_model directory not found, using {best_model_dir}")
                    
                    # Load model
                    from lib.training.model_factory import create_model
                    best_model_config = model_config.copy()
                    if best_params:
                        best_model_config.update(best_params)
                    
                    if model_type in BASELINE_MODELS:
                        from lib.utils.paths import load_metadata_flexible
                        stage2_df = load_metadata_flexible(features_stage2_path)
                        if stage2_df is not None and stage2_df.height > 0:
                            if "model_specific_config" not in best_model_config:
                                best_model_config["model_specific_config"] = {}
                            best_model_config["model_specific_config"]["features_stage2_path"] = features_stage2_path
                            best_model_config["features_stage2_path"] = features_stage2_path
                        
                        if model_type in STAGE4_MODELS:
                            stage4_df = load_metadata_flexible(features_stage4_path)
                            if stage4_df is not None and stage4_df.height > 0:
                                if "model_specific_config" not in best_model_config:
                                    best_model_config["model_specific_config"] = {}
                                best_model_config["model_specific_config"]["features_stage4_path"] = features_stage4_path
                                best_model_config["features_stage4_path"] = features_stage4_path
                    
                    test_model = create_model(model_type, best_model_config)
                    test_model.load(str(best_model_dir))
                    
                    # Evaluate on test set
                    test_probs = test_model.predict(test_df, project_root=project_root_str)
                    test_preds = np.argmax(test_probs, axis=1)
                    test_labels = test_df["label"].to_list()
                    label_map = {label: idx for idx, label in enumerate(sorted(set(test_labels)))}
                    test_y = np.array([label_map[label] for label in test_labels])
                    
                    # Compute test metrics
                    from lib.training.metrics_utils import compute_classification_metrics
                    test_metrics = compute_classification_metrics(
                        y_true=test_y,
                        y_pred=test_preds,
                        y_probs=test_probs
                    )
                    
                    # Compute additional metrics
                    from sklearn.metrics import (
                        roc_auc_score, average_precision_score, 
                        precision_recall_fscore_support, confusion_matrix
                    )
                    
                    if test_probs.ndim == 2 and test_probs.shape[1] == 2:
                        test_probs_pos = test_probs[:, 1]
                    else:
                        test_probs_pos = test_probs.flatten() if test_probs.ndim > 1 else test_probs
                    
                    test_auc = roc_auc_score(test_y, test_probs_pos)
                    test_ap = average_precision_score(test_y, test_probs_pos)
                    test_precision_per_class, test_recall_per_class, test_f1_per_class, _ = precision_recall_fscore_support(
                        test_y, test_preds, average=None, zero_division=0
                    )
                    test_cm = confusion_matrix(test_y, test_preds).tolist()
                    
                    # Prepare test results
                    test_results = {
                        "test_f1": float(test_metrics["val_f1"]),
                        "test_auc": float(test_auc),
                        "test_ap": float(test_ap),
                        "test_acc": float(test_metrics["val_acc"]),
                        "test_precision": float(test_metrics["val_precision"]),
                        "test_recall": float(test_metrics["val_recall"]),
                        "test_confusion_matrix": test_cm,
                        "test_precision_class0": float(test_precision_per_class[0]) if len(test_precision_per_class) > 0 else 0.0,
                        "test_recall_class0": float(test_recall_per_class[0]) if len(test_recall_per_class) > 0 else 0.0,
                        "test_f1_class0": float(test_f1_per_class[0]) if len(test_f1_per_class) > 0 else 0.0,
                        "test_precision_class1": float(test_precision_per_class[1]) if len(test_precision_per_class) > 1 else 0.0,
                        "test_recall_class1": float(test_recall_per_class[1]) if len(test_recall_per_class) > 1 else 0.0,
                        "test_f1_class1": float(test_f1_per_class[1]) if len(test_f1_per_class) > 1 else 0.0,
                    }
                    
                    # Combine with existing results
                    results_json = results[model_type].copy()
                    results_json.update(test_results)
                    if best_params:
                        results_json["best_params"] = best_params
                    
                    # Save results.json
                    results_file = model_output_dir / "results.json"
                    import json
                    with open(results_file, "w") as f:
                        json.dump(results_json, f, indent=2, default=str)
                    
                    logger.info(f"Test F1: {test_results['test_f1']:.4f}, Test AUC: {test_results['test_auc']:.4f}")
                    logger.info(f"Saved test results to {results_file}")
                    _flush_logs()
                    
                    # Cleanup
                    cleanup_model_and_memory(model=test_model, clear_cuda=False)
                    del test_model
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate {model_type} on test set: {e}", exc_info=True)
                    logger.warning("Continuing without test set evaluation")
                    _flush_logs()
        
        # Aggressive GC after all folds for this model type
        aggressive_gc(clear_cuda=False)
    
    # Train ensemble if requested
    if train_ensemble:
        logger.info("\n" + "="*80)
        logger.info("Training Ensemble Model")
        logger.info("="*80)
        _flush_logs()
        
        try:
            from .ensemble import train_ensemble_model
            
            ensemble_results = train_ensemble_model(
                project_root=project_root_str,
                scaled_metadata_path=scaled_metadata_path,
                base_model_types=model_types,
                base_models_dir=str(output_dir),
                n_splits=n_splits,
                num_frames=num_frames,
                output_dir=str(output_dir),
                ensemble_method=ensemble_method
            )
            
            results["ensemble"] = ensemble_results
            logger.info("✓ Ensemble training completed")
            _flush_logs()
        except Exception as e:
            logger.error(f"Error training ensemble: {e}", exc_info=True)
            logger.warning("Continuing without ensemble results")
            _flush_logs()
    
    logger.info("=" * 80)
    logger.info("Stage 5: Model Training Pipeline Completed")
    logger.info("=" * 80)
    _flush_logs()
    
    return results

