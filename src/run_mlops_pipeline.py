#!/usr/bin/env python3
"""
MLOps Pipeline Runner: Execute the optimized MLOps workflow.

This script demonstrates the new MLOps pipeline with:
- Experiment tracking
- Configuration versioning
- Checkpoint management with resume capability
- Data versioning
- Structured metrics logging
"""

import os
import sys
import logging
from pathlib import Path

# Setup extensive logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for extensive logs
    format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mlops_runner")

# Set specific loggers to appropriate levels
logging.getLogger("lib").setLevel(logging.DEBUG)
logging.getLogger("lib.mlops").setLevel(logging.DEBUG)
logging.getLogger("lib.training").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.augmentation").setLevel(logging.DEBUG)
logging.getLogger("lib.features").setLevel(logging.DEBUG)
logging.getLogger("lib.models").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.mlops import (
    RunConfig, ExperimentTracker, create_run_directory,
    build_mlops_pipeline, build_kfold_pipeline, build_multimodel_pipeline,
    cleanup_runs_and_logs
)
from lib.training import list_available_models


def main():
    """Run the MLOps pipeline."""
    # Detect project root
    if "SLURM_SUBMIT_DIR" in os.environ:
        project_root = os.environ["SLURM_SUBMIT_DIR"]
    else:
        project_root = str(Path(__file__).parent.parent)
    
    project_root = os.path.abspath(project_root)
    data_csv = os.path.join(project_root, "data", "video_index_input.csv")
    
    # Clean up previous runs, logs, models, and intermediate_data for fresh run
    logger.info("Cleaning up previous runs, logs, models, and intermediate_data for fresh start...")
    cleanup_runs_and_logs(project_root, keep_models=False, keep_intermediate_data=False)
    
    # Create run directory
    output_base = os.path.join(project_root, "runs")
    run_dir, run_id = create_run_directory(output_base, "fvc_binary_classifier")
    
    # Set up per-run file logging so we always have a pipeline log, even if SLURM output is missing
    run_log_dir = Path(run_dir) / "logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_log_dir / "pipeline.log"
    file_handler = logging.FileHandler(run_log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("MLOps Pipeline Run")
    logger.info("=" * 80)
    logger.info("Run ID: %s", run_id)
    logger.info("Run Directory: %s", run_dir)
    logger.info("Project Root: %s", project_root)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.debug("Environment variables:")
    for key in ["CUDA_VISIBLE_DEVICES", "FVC_FIXED_SIZE", "USE_MULTIMODEL", "MODELS_TO_TRAIN", "SKIP_MODELS"]:
        if key in os.environ:
            logger.debug("  %s=%s", key, os.environ[key])
    
    # Create experiment tracker
    tracker = ExperimentTracker(run_dir, run_id)
    
    # Create run configuration
    config = RunConfig(
        run_id=run_id,
        experiment_name="fvc_binary_classifier",
        description="FVC binary video classification with comprehensive augmentations",
        tags=["video_classification", "binary", "augmentations"],
        
        # Data config
        data_csv=data_csv,
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
        random_seed=42,
        
        # Video config (optimized for 1 GPU, 4 CPUs, 80GB RAM)
        num_frames=6,  # Reduced from 8 to 6 for 80GB RAM constraint
        fixed_size=int(os.environ.get("FVC_FIXED_SIZE", 112)),  # Conservative: 112x112 to minimize memory
        augmentation_config={
            'rotation_degrees': 15.0,
            'rotation_p': 0.5,
            'affine_p': 0.3,
            'gaussian_noise_std': 0.1,
            'gaussian_noise_p': 0.3,
            'gaussian_blur_p': 0.3,
            'cutout_p': 0.5,
            'cutout_max_size': 32,
            'elastic_transform_p': 0.2,
            'color_jitter_brightness': 0.3,
            'color_jitter_contrast': 0.3,
            'color_jitter_saturation': 0.3,
            'color_jitter_hue': 0.1,
        },
        temporal_augmentation_config={
            'frame_drop_prob': 0.1,
            'frame_dup_prob': 0.1,
            'reverse_prob': 0.1,
        },
        num_augmentations_per_video=1,  # Conservative: 1 augmentation to minimize memory
        
        # Training config (optimized for 80GB RAM)
        batch_size=4,  # Reduced from 8 to 4 for 80GB RAM constraint
        num_epochs=20,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size of 16
        early_stopping_patience=5,
        
        # System config (optimized for 4 CPUs, 80GB RAM)
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        num_workers=0,  # Always 0 to avoid multiprocessing memory overhead with 4 CPUs
        use_amp=True,
        
        # Paths
        project_root=project_root,
        output_dir=run_dir,
    )
    
    # Log system metadata with extensive details
    import torch
    import platform
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        logger.warning("psutil not available, skipping detailed system info")
    import gc
    
    system_metadata = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        system_metadata.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.device_count() > 0 else None,
        })
        logger.info("CUDA Device: %s", system_metadata.get("gpu_name", "Unknown"))
        logger.info("GPU Memory: %.2f GB", system_metadata.get("gpu_memory_total_gb", 0))
    
    if PSUTIL_AVAILABLE:
        try:
            system_metadata.update({
                "cpu_count": psutil.cpu_count(),
                "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "memory_total_gb": psutil.virtual_memory().total / 1e9,
                "memory_available_gb": psutil.virtual_memory().available / 1e9,
            })
            logger.info("CPU Count: %d", system_metadata.get("cpu_count", 0))
            logger.info("Total Memory: %.2f GB", system_metadata.get("memory_total_gb", 0))
            logger.info("Available Memory: %.2f GB", system_metadata.get("memory_available_gb", 0))
        except Exception as e:
            logger.warning("Could not get system info: %s", e)
    else:
        logger.debug("psutil not available, skipping detailed system info")
    
    tracker.log_metadata(system_metadata)
    logger.debug("System metadata logged: %s", system_metadata)
    
    # Build and run pipeline
    try:
        # Configuration: Use multi-model pipeline or single model
        use_multimodel = os.environ.get("USE_MULTIMODEL", "true").lower() == "true"
        use_kfold = True
        n_splits = 5
        
        # Create checkpoint manager for pipeline-level checkpointing
        from lib.mlops import CheckpointManager
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        ckpt_manager = CheckpointManager(checkpoint_dir, run_id)
        
        if use_multimodel:
            # Multi-model pipeline: train all models sequentially
            logger.info("=" * 80)
            logger.info("MULTI-MODEL TRAINING MODE")
            logger.info("=" * 80)
            
            # Define models to train (all models from proposal)
            all_available = list_available_models()
            models_to_train = [
                "logistic_regression",
                "svm",
                "naive_cnn",
                "vit_gru",
                "vit_transformer",
                "slowfast",
                "x3d",
            ]
            
            # Filter to only available models
            models_to_train = [m for m in models_to_train if m in all_available]
            
            # Option 1: SKIP_MODELS - exclude specific models (comma-separated)
            # Example: SKIP_MODELS="naive_cnn,slowfast"
            if "SKIP_MODELS" in os.environ:
                skip_list = [m.strip() for m in os.environ["SKIP_MODELS"].split(",")]
                models_to_train = [m for m in models_to_train if m not in skip_list]
                logger.info("Skipping models: %s", skip_list)
                logger.info("Remaining models to train: %s", models_to_train)
            
            # Option 2: MODELS_TO_TRAIN - explicitly specify which models (takes precedence over SKIP_MODELS)
            # Example: MODELS_TO_TRAIN="logistic_regression,svm,vit_gru"
            if "MODELS_TO_TRAIN" in os.environ:
                env_models = os.environ["MODELS_TO_TRAIN"].split(",")
                models_to_train = [m.strip() for m in env_models if m.strip() in all_available]
                logger.info("Using models from MODELS_TO_TRAIN: %s", models_to_train)
            
            logger.info("Models to train: %s", models_to_train)
            logger.info("Total models: %d", len(models_to_train))
            logger.info("Using %d-fold stratified cross-validation", n_splits)
            logger.info("Models will be trained sequentially with shared data pipeline")
            logger.info("Each model has its own checkpoint directory and can be resumed independently")
            logger.debug("Model training order:")
            for i, model in enumerate(models_to_train, 1):
                logger.debug("  %d. %s", i, model)
            
            # Build multi-model pipeline
            pipeline = build_multimodel_pipeline(
                config, models_to_train, tracker, n_splits=n_splits, ckpt_manager=ckpt_manager
            )
            
        elif use_kfold:
            # Single model with k-fold
            logger.info("Using %d-fold stratified cross-validation (single model: %s)", 
                       n_splits, config.model_type)
            pipeline = build_kfold_pipeline(config, tracker, n_splits=n_splits, ckpt_manager=ckpt_manager)
        else:
            # Single model, single split
            logger.info("Using single train/val split (model: %s)", config.model_type)
            pipeline = build_mlops_pipeline(config, tracker)
            pipeline.ckpt_manager = ckpt_manager
        
        logger.info("=" * 80)
        logger.info("Starting pipeline execution...")
        logger.info("=" * 80)
        logger.debug("Pipeline type: %s", "multi-model" if use_multimodel else ("k-fold" if use_kfold else "single-split"))
        logger.debug("Checkpoint directory: %s", checkpoint_dir)
        import time
        pipeline_start = time.time()
        
        artifacts = pipeline.run_pipeline(ckpt_manager=ckpt_manager)
        
        pipeline_duration = time.time() - pipeline_start
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("Run ID: %s", run_id)
        logger.info("Results saved to: %s", run_dir)
        logger.info("Total execution time: %.2f seconds (%.2f minutes)", 
                   pipeline_duration, pipeline_duration / 60)
        logger.info("=" * 80)
        
        # Print summary with extensive details
        logger.debug("Retrieving metrics from tracker...")
        metrics_df = tracker.get_metrics()
        logger.debug("Total metrics logged: %d", metrics_df.height if metrics_df.height > 0 else 0)
        
        if metrics_df.height > 0:
            logger.info("Metrics Summary:")
            best_val = tracker.get_best_metric("accuracy", phase="val", maximize=True)
            if best_val:
                logger.info("  Best validation accuracy: %.4f (epoch %d)", 
                           best_val['value'], best_val['epoch'])
            
            # Log all available metrics
            logger.debug("Available metrics:")
            for col in metrics_df.columns:
                logger.debug("  - %s", col)
        
        logger.debug("Pipeline artifacts: %s", list(artifacts.keys()) if artifacts else "None")
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        logger.info("Checkpoints saved. Pipeline can be resumed.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", str(e))
        logger.error("Exception type: %s", type(e).__name__)
        logger.error("Full traceback:", exc_info=True)
        logger.error("Run ID: %s", run_id)
        logger.error("Run directory: %s", run_dir)
        logger.error("Checkpoints may be available for resume")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

