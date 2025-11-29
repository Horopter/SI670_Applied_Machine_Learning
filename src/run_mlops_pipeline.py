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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlops_runner")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.mlops_core import RunConfig, ExperimentTracker, create_run_directory
from lib.mlops_pipeline import build_mlops_pipeline
from lib.mlops_pipeline_kfold import build_kfold_pipeline


def main():
    """Run the MLOps pipeline."""
    # Detect project root
    if "SLURM_SUBMIT_DIR" in os.environ:
        project_root = os.environ["SLURM_SUBMIT_DIR"]
    else:
        project_root = str(Path(__file__).parent.parent)
    
    project_root = os.path.abspath(project_root)
    data_csv = os.path.join(project_root, "data", "video_index_input.csv")
    
    # Create run directory
    output_base = os.path.join(project_root, "runs")
    run_dir, run_id = create_run_directory(output_base, "fvc_binary_classifier")
    
    logger.info("=" * 80)
    logger.info("MLOps Pipeline Run")
    logger.info("=" * 80)
    logger.info("Run ID: %s", run_id)
    logger.info("Run Directory: %s", run_dir)
    logger.info("Project Root: %s", project_root)
    
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
        
        # Video config
        num_frames=16,
        fixed_size=224,
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
        num_augmentations_per_video=3,
        
        # Training config
        batch_size=32,
        num_epochs=20,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_accumulation_steps=1,
        early_stopping_patience=5,
        
        # System config
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        num_workers=4,
        use_amp=True,
        
        # Paths
        project_root=project_root,
        output_dir=run_dir,
    )
    
    # Log system metadata
    import torch
    import platform
    tracker.log_metadata({
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
    })
    
    # Build and run pipeline with K-fold cross-validation
    try:
        # Use K-fold cross-validation to prevent overfitting/underfitting
        use_kfold = True
        n_splits = 5
        
        # Create checkpoint manager for pipeline-level checkpointing
        from lib.mlops_core import CheckpointManager
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        ckpt_manager = CheckpointManager(checkpoint_dir, run_id)
        
        if use_kfold:
            logger.info("Using %d-fold stratified cross-validation", n_splits)
            pipeline = build_kfold_pipeline(config, tracker, n_splits=n_splits, ckpt_manager=ckpt_manager)
        else:
            logger.info("Using single train/val split")
            pipeline = build_mlops_pipeline(config, tracker)
            pipeline.ckpt_manager = ckpt_manager
        
        artifacts = pipeline.run_pipeline(ckpt_manager=ckpt_manager)
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("Run ID: %s", run_id)
        logger.info("Results saved to: %s", run_dir)
        logger.info("=" * 80)
        
        # Print summary
        metrics_df = tracker.get_metrics()
        if metrics_df.height > 0:
            best_val = tracker.get_best_metric("accuracy", phase="val", maximize=True)
            if best_val:
                logger.info("Best validation accuracy: %.4f (epoch %d)", 
                           best_val['value'], best_val['epoch'])
        
        return 0
    
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

