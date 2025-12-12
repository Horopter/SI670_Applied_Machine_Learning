#!/usr/bin/env python3
"""
Stage 4: Scaled Video Feature Extraction Script

Extracts additional features from scaled videos (P features).
Includes binary features: is_upscaled and is_downscaled.

Usage:
    python src/scripts/run_stage4_scaled_features.py
    python src/scripts/run_stage4_scaled_features.py --num-frames 8
    python src/scripts/run_stage4_scaled_features.py --scaled-metadata data/scaled_videos/scaled_metadata.arrow
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.features import stage4_extract_scaled_features
from lib.utils.memory import log_memory_stats

# Setup extensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger("lib").setLevel(logging.DEBUG)
logging.getLogger("lib.features").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)


def main():
    """Run Stage 4: Scaled Video Feature Extraction."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Extract features from scaled videos (includes is_upscaled and is_downscaled features)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: use scaled metadata from Stage 3
  python src/scripts/run_stage4_scaled_features.py
  
  # Custom number of frames
  python src/scripts/run_stage4_scaled_features.py --num-frames 6
  
  # Custom metadata path
  python src/scripts/run_stage4_scaled_features.py --scaled-metadata data/custom/scaled_metadata.arrow
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
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to sample per video (if provided, overrides percentage-based sampling). "
             "If not provided, uses percentage-based adaptive sampling (10% of frames, min=5, max=50)"
    )
    parser.add_argument(
        "--frame-percentage",
        type=float,
        default=0.10,
        help="Percentage of frames to sample per video (default: 0.10 = 10%). "
             "Only used if --num-frames is not provided."
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Minimum frames to sample per video (for percentage-based sampling, default: 5)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum frames to sample per video (for percentage-based sampling, default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features_stage4",
        help="Output directory for features (default: data/features_stage4)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for video range (0-based, inclusive). If not specified, processes all videos from start."
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for video range (0-based, exclusive). If not specified, processes all videos to end."
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing feature files before regenerating (clean mode, default: False, preserves existing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing feature files (skip already processed videos, default: True)"
    )
    parser.add_argument(
        "--execution-order",
        type=str,
        default="forward",
        choices=["forward", "reverse", "0", "1"],
        help="Execution order: 'forward' or '0' (default) processes from start_idx to end_idx, "
             "'reverse' or '1' processes from end_idx-1 down to start_idx"
    )
    
    args = parser.parse_args()
    
    # Input validation
    if not args.project_root or not isinstance(args.project_root, str):
        logger.error(f"project_root must be a non-empty string, got: {type(args.project_root)}")
        sys.exit(1)
    if not args.scaled_metadata or not isinstance(args.scaled_metadata, str):
        logger.error(f"scaled_metadata must be a non-empty string, got: {type(args.scaled_metadata)}")
        sys.exit(1)
    if args.num_frames is not None and (not isinstance(args.num_frames, int) or args.num_frames <= 0):
        logger.error(f"num_frames must be a positive integer, got: {args.num_frames}")
        sys.exit(1)
    if not isinstance(args.frame_percentage, (int, float)) or not (0 < args.frame_percentage <= 1):
        logger.error(f"frame_percentage must be between 0 and 1, got: {args.frame_percentage}")
        sys.exit(1)
    if not isinstance(args.min_frames, int) or args.min_frames <= 0:
        logger.error(f"min_frames must be a positive integer, got: {args.min_frames}")
        sys.exit(1)
    if not isinstance(args.max_frames, int) or args.max_frames <= 0:
        logger.error(f"max_frames must be a positive integer, got: {args.max_frames}")
        sys.exit(1)
    if args.min_frames > args.max_frames:
        logger.error(f"min_frames ({args.min_frames}) must be <= max_frames ({args.max_frames})")
        sys.exit(1)
    if args.start_idx is not None and (not isinstance(args.start_idx, int) or args.start_idx < 0):
        logger.error(f"start_idx must be a non-negative integer, got: {args.start_idx}")
        sys.exit(1)
    if args.end_idx is not None and (not isinstance(args.end_idx, int) or args.end_idx < 0):
        logger.error(f"end_idx must be a non-negative integer, got: {args.end_idx}")
        sys.exit(1)
    
    # Normalize execution_order: "0" or "forward" -> "forward", "1" or "reverse" -> "reverse"
    if args.execution_order in ("0", "forward"):
        execution_order = "forward"
    elif args.execution_order in ("1", "reverse"):
        execution_order = "reverse"
    else:
        execution_order = "forward"  # Default
    
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
    
    # Handle relative paths - if args.scaled_metadata is relative, make it relative to project_root
    if Path(args.scaled_metadata).is_absolute():
        scaled_metadata_path = Path(args.scaled_metadata)
    else:
        scaled_metadata_path = project_root / args.scaled_metadata
    
    try:
        output_dir = project_root / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create output directory {args.output_dir}: {e}")
        sys.exit(1)
    
    # Logging setup - also log to file
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage4_scaled_features_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Start logging
    logger.info("=" * 80)
    logger.info("STAGE 4: SCALED VIDEO FEATURE EXTRACTION")
    logger.info("=" * 80)
    logger.info("Project root: %s", project_root)
    logger.info("Scaled metadata: %s", scaled_metadata_path)
    logger.info("Output directory: %s", output_dir)
    if args.num_frames is not None:
        logger.info("Frame sampling: Fixed %d frames per video", args.num_frames)
    else:
        logger.info("Frame sampling: Adaptive (%.1f%% of frames, min=%d, max=%d)", 
                   args.frame_percentage * 100, args.min_frames, args.max_frames)
    if args.start_idx is not None or args.end_idx is not None:
        logger.info("Video range: [%s, %s)",
                   args.start_idx if args.start_idx is not None else "0",
                   args.end_idx if args.end_idx is not None else "all")
    logger.info("Delete existing: %s", args.delete_existing)
    logger.info("Resume mode: %s", args.resume)
    logger.info("Execution order: %s", execution_order)
    logger.info("Log file: %s", log_file)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.debug("Command line arguments: %s", sys.argv)
    
    # Check prerequisites
    logger.info("=" * 80)
    logger.info("Checking prerequisites...")
    logger.info("=" * 80)
    
    # Try to find metadata file (check for alternative formats)
    from lib.utils.paths import find_metadata_file
    
    if not scaled_metadata_path.exists():
        logger.warning("Specified metadata file not found: %s", scaled_metadata_path)
        logger.info("Checking for alternative metadata file formats...")
        
        # Try to find metadata file with flexible search
        metadata_dir = scaled_metadata_path.parent
        metadata_name = scaled_metadata_path.stem  # e.g., "scaled_metadata"
        
        found_metadata = find_metadata_file(metadata_dir, metadata_name)
        
        if found_metadata and found_metadata.exists():
            logger.info("✓ Found alternative metadata file: %s", found_metadata)
            logger.info("  Using this file instead of: %s", scaled_metadata_path)
            scaled_metadata_path = found_metadata
        else:
            logger.error("Scaled metadata file not found: %s", scaled_metadata_path)
            
            # Check for alternative formats
            logger.info("Checking for alternative metadata files in: %s", metadata_dir)
            
            if metadata_dir.exists():
                # Look for common metadata file patterns
                alt_files = []
                for pattern in ["scaled_metadata.*", "*metadata*.parquet", "*metadata*.arrow", "*metadata*.csv"]:
                    alt_files.extend(list(metadata_dir.glob(pattern)))
                
                if alt_files:
                    logger.info("Found potential metadata files:")
                    for f in sorted(alt_files):
                        logger.info("  - %s (size: %s bytes)", f.name, f.stat().st_size if f.exists() else "N/A")
                    logger.info("")
                    logger.info("Try using one of these files with --scaled-metadata")
                else:
                    logger.warning("No metadata files found in: %s", metadata_dir)
                    logger.info("")
                    logger.info("Available files in directory:")
                    try:
                        dir_files = sorted(metadata_dir.iterdir())
                        for f in dir_files[:20]:  # Show first 20 files
                            logger.info("  - %s", f.name)
                        if len(dir_files) > 20:
                            logger.info("  ... and %d more files", len(dir_files) - 20)
                    except Exception as e:
                        logger.debug("Could not list directory contents: %s", e)
            else:
                logger.error("Metadata directory does not exist: %s", metadata_dir)
            
            logger.error("")
            logger.error("Please run Stage 3 first: python src/scripts/run_stage3_scaling.py")
            logger.error("Or specify the correct metadata file path with --scaled-metadata")
            return 1
    
    logger.info("✓ Scaled metadata file found: %s", scaled_metadata_path)
    logger.info("  File size: %s bytes", scaled_metadata_path.stat().st_size)
    
    # Log system information
    try:
        import psutil
        logger.debug("System information:")
        logger.debug("  CPU count: %d", psutil.cpu_count())
        logger.debug("  Total memory: %.2f GB", psutil.virtual_memory().total / 1e9)
        logger.debug("  Available memory: %.2f GB", psutil.virtual_memory().available / 1e9)
    except ImportError:
        logger.debug("psutil not available, skipping system info")
    
    # Log initial memory stats
    logger.info("=" * 80)
    logger.info("Initial memory statistics:")
    logger.info("=" * 80)
    log_memory_stats("Stage 4: before scaled feature extraction", detailed=True)
    
    # Run Stage 4
    logger.info("=" * 80)
    logger.info("Starting Stage 4: Scaled Video Feature Extraction")
    logger.info("=" * 80)
    logger.info("This may take a while depending on dataset size...")
    logger.info("Progress will be logged in real-time")
    logger.info("=" * 80)
    
    stage_start = time.time()
    
    try:
        result_df = stage4_extract_scaled_features(
            project_root=str(project_root),
            scaled_metadata_path=str(scaled_metadata_path),
            num_frames=args.num_frames,
            output_dir=args.output_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            delete_existing=args.delete_existing,
            resume=args.resume,
            frame_percentage=args.frame_percentage,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            execution_order=execution_order
        )
        
        stage_duration = time.time() - stage_start
        
        logger.info("=" * 80)
        logger.info("STAGE 4 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Execution time: %.2f seconds (%.2f minutes)", 
                   stage_duration, stage_duration / 60)
        logger.info("Output directory: %s", output_dir)
        logger.info("Features metadata: %s", output_dir / "features_scaled_metadata.arrow")
        
        if result_df is not None and hasattr(result_df, 'height'):
            logger.info("Total videos processed: %d", result_df.height)
            logger.debug("Result DataFrame shape: %s", result_df.shape)
        else:
            logger.warning("Result DataFrame is None or invalid")
        
        # Log final memory stats
        logger.info("=" * 80)
        logger.info("Final memory statistics:")
        logger.info("=" * 80)
        log_memory_stats("Stage 4: after scaled feature extraction", detailed=True)
        
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  - Run Stage 5: python src/scripts/run_stage5_training.py")
        logger.info("  - Or continue with full pipeline: python src/run_new_pipeline.py --skip-stage 1,2,3,4")
        logger.info("=" * 80)
        
        # Ensure all logs are flushed before exit
        sys.stdout.flush()
        sys.stderr.flush()
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("FEATURE EXTRACTION INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning("Partial results may be available in: %s", output_dir)
        logger.warning("You can resume by running the script again")
        return 130
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("STAGE 4 FAILED")
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

