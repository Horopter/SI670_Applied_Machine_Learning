#!/usr/bin/env python3
"""
Stage 5: Model Training Script

Trains models using scaled videos and extracted features.

Usage:
    python src/scripts/run_stage5_training.py
    python src/scripts/run_stage5_training.py --model-types logistic_regression svm
    python src/scripts/run_stage5_training.py --n-splits 5
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.training.pipeline import stage5_train_models
from lib.training.video_training_pipeline import FEATURE_BASED_MODELS, VIDEO_BASED_MODELS
from lib.utils.memory import log_memory_stats

# Setup logging (INFO level for production, DEBUG is too verbose)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Run Stage 5: Model Training."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Train models using scaled videos and features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: train logistic_regression and svm with 5-fold CV
  python src/scripts/run_stage5_training.py
  
  # Train specific models
  python src/scripts/run_stage5_training.py --model-types logistic_regression svm naive_cnn
  
  # Custom k-fold splits
  python src/scripts/run_stage5_training.py --n-splits 10
  
  # Train all available models
  python src/scripts/run_stage5_training.py --model-types all
        """
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--scaled-metadata",
        type=str,
        default="data/scaled_videos/scaled_metadata.arrow",
        help="Path to scaled metadata from Stage 3 (default: data/scaled_videos/scaled_metadata.arrow). "
             "Also supports .parquet and .csv formats."
    )
    parser.add_argument(
        "--features-stage2",
        type=str,
        default="data/features_stage2/features_metadata.arrow",
        help="Path to Stage 2 features metadata (default: data/features_stage2/features_metadata.arrow). "
             "Also supports .parquet and .csv formats."
    )
    parser.add_argument(
        "--features-stage4",
        type=str,
        default="data/features_stage4/features_scaled_metadata.arrow",
        help="Path to Stage 4 features metadata (default: data/features_stage4/features_scaled_metadata.arrow). "
             "Also supports .parquet and .csv formats."
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=["logistic_regression", "svm"],
        help="Model types to train (default: logistic_regression svm). Use 'all' for all available models."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of k-fold splits (default: 5)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames per video (default: 8, optimized for 256GB RAM)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stage5",
        help="Output directory for training results (default: data/training_results)"
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable experiment tracking"
    )
    parser.add_argument(
        "--train-ensemble",
        action="store_true",
        default=False,
        help="Train ensemble model after individual models (default: False)"
    )
    parser.add_argument(
        "--ensemble-method",
        type=str,
        choices=["meta_learner", "weighted_average"],
        default="meta_learner",
        help="Ensemble method: 'meta_learner' (train MLP) or 'weighted_average' (simple average) (default: meta_learner)"
    )
    parser.add_argument(
        "--model-idx",
        type=int,
        default=None,
        help="Model index for multi-node training (0-based). If specified, trains only this model from the list. "
             "For multi-node: each node trains one model."
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing model checkpoints/results before regenerating (clean mode, default: False, preserves existing)"
    )
    
    args = parser.parse_args()
    
    # Input validation
    if not args.project_root or not isinstance(args.project_root, str):
        logger.error(f"project_root must be a non-empty string, got: {type(args.project_root)}")
        sys.exit(1)
    if not args.scaled_metadata or not isinstance(args.scaled_metadata, str):
        logger.error(f"scaled_metadata must be a non-empty string, got: {type(args.scaled_metadata)}")
        sys.exit(1)
    if not args.features_stage2 or not isinstance(args.features_stage2, str):
        logger.error(f"features_stage2 must be a non-empty string, got: {type(args.features_stage2)}")
        sys.exit(1)
    if not args.features_stage4 or not isinstance(args.features_stage4, str):
        logger.error(f"features_stage4 must be a non-empty string, got: {type(args.features_stage4)}")
        sys.exit(1)
    if not args.model_types or not isinstance(args.model_types, list) or len(args.model_types) == 0:
        logger.error(f"model_types must be a non-empty list, got: {type(args.model_types)}")
        sys.exit(1)
    if not isinstance(args.n_splits, int) or args.n_splits <= 0:
        logger.error(f"n_splits must be a positive integer, got: {args.n_splits}")
        sys.exit(1)
    if not isinstance(args.num_frames, int) or args.num_frames <= 0:
        logger.error(f"num_frames must be a positive integer, got: {args.num_frames}")
        sys.exit(1)
    
    # Convert to Path objects with validation
    try:
        project_root = Path(args.project_root).resolve()
        if not project_root.exists():
            logger.error(f"Project root directory does not exist: {project_root}")
            sys.exit(1)
        if not project_root.is_dir():
            logger.error(f"Project root is not a directory: {project_root}")
            sys.exit(1)
    except (OSError, ValueError) as e:
        logger.error(f"Invalid project_root path: {args.project_root} - {e}")
        sys.exit(1)
    
    scaled_metadata_path = project_root / args.scaled_metadata
    features_stage2_path = project_root / args.features_stage2
    features_stage4_path = project_root / args.features_stage4
    
    # Validate paths exist (warn but don't fail - they might be created later)
    if not scaled_metadata_path.exists():
        logger.warning(f"Scaled metadata path does not exist: {scaled_metadata_path}")
    if not features_stage2_path.exists():
        logger.warning(f"Stage 2 features path does not exist: {features_stage2_path}")
    if not features_stage4_path.exists():
        logger.warning(f"Stage 4 features path does not exist: {features_stage4_path}")
    
    try:
        output_dir = project_root / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create output directory {args.output_dir}: {e}")
        sys.exit(1)
    
    # Handle "all" model types
    if "all" in args.model_types:
        # Get all models (feature + video based)
        all_model_types = list(FEATURE_BASED_MODELS | VIDEO_BASED_MODELS)
        logger.info("Training all available models: %s", all_model_types)
    else:
        all_model_types = args.model_types
    
    # Handle model-idx for multi-node training
    if args.model_idx is not None:
        if args.model_idx < 0 or args.model_idx >= len(all_model_types):
            logger.error("Invalid model-idx %d. Must be between 0 and %d", args.model_idx, len(all_model_types) - 1)
            return 1
        model_types = [all_model_types[args.model_idx]]
        logger.info("Multi-node mode: Training model %d/%d: %s", args.model_idx + 1, len(all_model_types), model_types[0])
    else:
        model_types = all_model_types
    
    # Logging setup - also log to file
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage5_training_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Start logging
    logger.info("=" * 80)
    logger.info("STAGE 5: MODEL TRAINING")
    logger.info("=" * 80)
    logger.info("Project root: %s", project_root)
    logger.info("Scaled metadata: %s", scaled_metadata_path)
    logger.info("Features Stage 2: %s", features_stage2_path)
    logger.info("Features Stage 4: %s", features_stage4_path)
    logger.info("Model types: %s", model_types)
    logger.info("K-fold splits: %d", args.n_splits)
    logger.info("Number of frames: %d", args.num_frames)
    logger.info("Output directory: %s", output_dir)
    logger.info("Experiment tracking: %s", "Disabled" if args.no_tracking else "Enabled")
    if args.model_idx is not None:
        logger.info("Model index: %d (multi-node mode)", args.model_idx)
    logger.info("Delete existing: %s", args.delete_existing)
    logger.info("Log file: %s", log_file)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.debug("Command line arguments: %s", sys.argv)
    
    # Check prerequisites
    logger.info("=" * 80)
    logger.info("Checking prerequisites...")
    logger.info("=" * 80)
    
    # Basic file existence check (detailed validation done in pipeline)
    if not scaled_metadata_path.exists():
        logger.error("Scaled metadata file not found: %s", scaled_metadata_path)
        logger.error("Please run Stage 3 first: python src/scripts/run_stage3_scaling.py")
        return 1
    logger.info("✓ Scaled metadata file found: %s", scaled_metadata_path)
    
    # Validate model types
    available_models = list(FEATURE_BASED_MODELS | VIDEO_BASED_MODELS)
    invalid_models = [m for m in model_types if m not in available_models]
    if invalid_models:
        logger.error("Invalid model types: %s", invalid_models)
        logger.error("Available models: %s", available_models)
        return 1
    
    logger.info("✓ All model types are valid")
    logger.debug("Available models: %s", available_models)
    logger.debug("Feature-based: %s", FEATURE_BASED_MODELS)
    logger.debug("Video-based: %s", VIDEO_BASED_MODELS)
    
    # Log initial memory stats
    logger.info("=" * 80)
    logger.info("Initial memory statistics:")
    logger.info("=" * 80)
    log_memory_stats("Stage 5: before training", detailed=True)
    
    # Run Stage 5
    logger.info("=" * 80)
    logger.info("Starting Stage 5: Model Training")
    logger.info("=" * 80)
    logger.info("Training %d model(s) with %d-fold cross-validation", len(model_types), args.n_splits)
    logger.info("This may take a while depending on dataset size and model complexity...")
    logger.info("Progress will be logged in real-time")
    logger.info("=" * 80)
    
    stage_start = time.time()
    
    try:
        # Use pipeline defaults for batch_size/epochs (from model config)
        # Don't hardcode - let pipeline use model-specific defaults
        results = stage5_train_models(
            project_root=str(project_root),
            scaled_metadata_path=str(scaled_metadata_path),
            features_stage2_path=str(features_stage2_path),
            features_stage4_path=str(features_stage4_path),
            model_types=model_types,
            n_splits=args.n_splits,
            num_frames=args.num_frames,
            output_dir=args.output_dir,
            use_tracking=not args.no_tracking,
            use_mlflow=not args.no_tracking,
            train_ensemble=args.train_ensemble,
            ensemble_method=args.ensemble_method,
            delete_existing=args.delete_existing
        )
        
        stage_duration = time.time() - stage_start
        
        logger.info("=" * 80)
        logger.info("STAGE 5 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Execution time: %.2f seconds (%.2f minutes)", 
                   stage_duration, stage_duration / 60)
        logger.info("Output directory: %s", output_dir)
        logger.info("Models trained: %s", model_types)
        logger.info("K-fold splits: %d", args.n_splits)
        
        if results:
            logger.debug("Training results: %s", results)
        
        # Log final memory stats
        logger.info("=" * 80)
        logger.info("Final memory statistics:")
        logger.info("=" * 80)
        log_memory_stats("Stage 5: after training", detailed=True)
        
        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info("Results saved to: %s", output_dir)
        logger.info("=" * 80)
        
        # Ensure all logs are flushed before exit
        sys.stdout.flush()
        sys.stderr.flush()
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("TRAINING INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning("Partial results may be available in: %s", output_dir)
        logger.warning("You can resume by running the script again")
        return 130
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("STAGE 5 FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", str(e))
        logger.error("Exception type: %s", type(e).__name__)
        logger.error("Full traceback:", exc_info=True)
        logger.error("Output directory: %s", output_dir)
        logger.error("Partial results may be available")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    # Ensure all output is flushed before exit
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exit_code)

