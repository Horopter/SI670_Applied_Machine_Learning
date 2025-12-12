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
import signal
import traceback
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup signal handlers to catch crashes and log diagnostics
def setup_crash_handlers(logger):
    """Setup signal handlers to catch crashes and log diagnostics."""
    def crash_handler(signum, frame):
        """Handle crash signals and log diagnostics before exit."""
        logger.critical("=" * 80)
        logger.critical("CRITICAL: Process received signal %d (likely crash/segfault)", signum)
        logger.critical("=" * 80)
        logger.critical("Signal name: %s", signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum))
        logger.critical("Attempting to log diagnostics before crash...")
        
        try:
            import psutil
            process = psutil.Process()
            logger.critical("Memory usage: RSS=%.2f GB, VMS=%.2f GB", 
                          process.memory_info().rss / 1024**3,
                          process.memory_info().vms / 1024**3)
        except Exception:
            pass
        
        try:
            logger.critical("Stack trace at crash:")
            for line in traceback.format_stack(frame):
                logger.critical(line.rstrip())
        except Exception:
            pass
        
        logger.critical("=" * 80)
        sys.stdout.flush()
        sys.stderr.flush()
        # Re-raise to get core dump
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    # Register handlers for common crash signals
    if hasattr(signal, 'SIGSEGV'):
        signal.signal(signal.SIGSEGV, crash_handler)
    if hasattr(signal, 'SIGABRT'):
        signal.signal(signal.SIGABRT, crash_handler)
    if hasattr(signal, 'SIGBUS'):
        signal.signal(signal.SIGBUS, crash_handler)
    if hasattr(signal, 'SIGFPE'):
        signal.signal(signal.SIGFPE, crash_handler)

from lib.training.pipeline import stage5_train_models
from lib.training.video_training_pipeline import FEATURE_BASED_MODELS, VIDEO_BASED_MODELS
from lib.utils.memory import log_memory_stats

# Setup logging (INFO level for production, DEBUG is too verbose)
# Configure for immediate output (unbuffered)
class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after each log message."""
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,  # Override any existing configuration
    handlers=[FlushingStreamHandler(sys.stdout)]  # Use flushing handler
)
logger = logging.getLogger(__name__)

# Force immediate flushing for all handlers
for handler in logging.root.handlers:
    if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
        handler.stream.flush()

# Setup crash handlers after logger is created
setup_crash_handlers(logger)


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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training by skipping folds that already have saved models (default: True). Use --no-resume to disable."
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode - train all folds even if they already exist"
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
    
    # Force immediate flushing for file handler
    if hasattr(file_handler, 'stream') and hasattr(file_handler.stream, 'flush'):
        file_handler.stream.flush()
    
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
    logger.info("Resume mode: %s", args.resume)
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
    
    # Immediate logging before calling pipeline function
    logger.info("Calling Stage 5 training pipeline...")
    logger.info("This may take a while - progress will be logged in real-time")
    sys.stdout.flush()  # Ensure immediate output
    
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
            delete_existing=args.delete_existing,
            resume=args.resume
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
    try:
        exit_code = 0
        try:
            exit_code = main()
            # Ensure all output is flushed before exit
            sys.stdout.flush()
            sys.stderr.flush()
        except SystemExit as e:
            # Capture exit code from SystemExit (from sys.exit() calls)
            exit_code = e.code if e.code is not None else 0
            sys.stdout.flush()
            sys.stderr.flush()
        except KeyboardInterrupt:
            logger.critical("Process interrupted by user")
            sys.stdout.flush()
            sys.stderr.flush()
            exit_code = 130
        except Exception as e:
            # Catch any unhandled exceptions that might lead to crashes
            logger.critical("=" * 80)
            logger.critical("UNHANDLED EXCEPTION - This may cause a crash")
            logger.critical("=" * 80)
            logger.critical(f"Exception type: {type(e).__name__}")
            logger.critical(f"Exception message: {str(e)}")
            logger.critical("Full traceback:", exc_info=True)
            logger.critical("=" * 80)
            sys.stdout.flush()
            sys.stderr.flush()
            exit_code = 1
        
        # Explicit cleanup before exit
        import gc
        gc.collect()
        
        # Use os._exit to bypass Python cleanup that might cause crashes
        os._exit(exit_code)
    except SystemExit:
        # Re-raise system exits (normal termination)
        raise
    except Exception:
        # Last resort: force exit
        import os
        os._exit(1)

