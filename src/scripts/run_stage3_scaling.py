#!/usr/bin/env python3
"""
Stage 3: Video Scaling Script

Scales videos to a target max dimension using letterboxing or autoencoder.
Can both downscale and upscale videos to ensure max(width, height) = target_size.

Usage:
    python src/scripts/run_stage3_scaling.py
    python src/scripts/run_stage3_scaling.py --target-size 224
    python src/scripts/run_stage3_scaling.py --method resolution
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import time
from pathlib import Path
import polars as pl

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.scaling import stage3_scale_videos
from lib.utils.memory import log_memory_stats
from lib.data import load_metadata, filter_existing_videos
from lib.utils.paths import resolve_video_path, load_metadata_flexible

# Setup extensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger("lib").setLevel(logging.DEBUG)
logging.getLogger("lib.scaling").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)


def check_videos_needing_scaling(
    project_root: Path,
    augmented_metadata_path: Path,
    output_dir: Path,
    target_size: int,
    start_idx: int = None,
    end_idx: int = None
) -> tuple[list[int], int, int]:
    """
    Pass 1: Check which videos need scaling.
    
    Returns:
        (videos_needing_scaling, videos_complete, videos_incomplete)
        - videos_needing_scaling: List of indices in the dataframe that need scaling
        - videos_complete: Count of videos with scaled versions already
        - videos_incomplete: Count of videos needing scaling
    """
    logger.info("PASS 1: Checking which videos need scaling...")
    
    # Load augmented metadata
    try:
        df = load_metadata_flexible(str(augmented_metadata_path))
    except Exception as e:
        logger.error(f"Failed to load augmented metadata: {e}")
        raise
    
    if df is None or df.height == 0:
        logger.warning("No videos found in augmented metadata")
        return [], 0, 0
    
    df = filter_existing_videos(df, str(project_root))
    
    # Apply range filtering if specified
    total_videos = df.height
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else total_videos
        if start < 0:
            start = 0
        if end > total_videos:
            end = total_videos
        if start >= end:
            raise ValueError(f"Invalid range: start_idx={start}, end_idx={end}")
        df = df.slice(start, end - start)
    
    # Check existing scaled metadata
    metadata_path_arrow = output_dir / "scaled_metadata.arrow"
    metadata_path_parquet = output_dir / "scaled_metadata.parquet"
    metadata_path_csv = output_dir / "scaled_metadata.csv"
    
    metadata_path = None
    if metadata_path_arrow.exists():
        metadata_path = metadata_path_arrow
    elif metadata_path_parquet.exists():
        metadata_path = metadata_path_parquet
    elif metadata_path_csv.exists():
        metadata_path = metadata_path_csv
    
    # Load existing scaled video paths
    existing_scaled_paths = set()
    if metadata_path and metadata_path.exists():
        try:
            if metadata_path.suffix == '.arrow':
                existing_metadata = pl.read_ipc(metadata_path)
            elif metadata_path.suffix == '.parquet':
                existing_metadata = pl.read_parquet(metadata_path)
            else:
                existing_metadata = pl.read_csv(str(metadata_path))
            
            # Extract video IDs from existing scaled videos
            for row in existing_metadata.iter_rows(named=True):
                scaled_path = row.get("video_path", "")
                if scaled_path:
                    # Extract video_id from scaled path (e.g., "video_id_scaled_original.mp4" or "video_id_scaled_aug0.mp4")
                    scaled_filename = Path(scaled_path).stem
                    # Remove "_scaled_original" or "_scaled_augX" suffix to get base video_id
                    if "_scaled_original" in scaled_filename:
                        video_id = scaled_filename.replace("_scaled_original", "")
                    elif "_scaled_aug" in scaled_filename:
                        video_id = scaled_filename.split("_scaled_aug")[0]
                    else:
                        video_id = scaled_filename
                    existing_scaled_paths.add(video_id)
        except Exception as e:
            logger.warning(f"Could not load existing scaled metadata: {e}")
    
    # Pass 1: Identify videos that need scaling
    videos_needing_scaling = []
    videos_complete = 0
    videos_incomplete = 0
    
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            if not Path(video_path).exists():
                videos_needing_scaling.append(idx)
                videos_incomplete += 1
                continue
            
            # Extract video_id (same logic as in stage3_scale_videos)
            video_id = Path(video_path).stem
            video_id = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in video_id)
            
            # Check if scaled video exists (check both original and augmented versions)
            aug_idx = row.get("augmentation_idx", -1)
            is_original = row.get("is_original", False)
            
            if is_original or aug_idx == -1 or aug_idx is None:
                output_filename = f"{video_id}_scaled_original.mp4"
            else:
                try:
                    aug_idx_int = int(aug_idx) if aug_idx is not None else -1
                    output_filename = f"{video_id}_scaled_aug{aug_idx_int}.mp4"
                except (ValueError, TypeError):
                    output_filename = f"{video_id}_scaled_original.mp4"
            
            output_path = output_dir / output_filename
            
            # Check if scaled video exists
            if output_path.exists() or video_id in existing_scaled_paths:
                videos_complete += 1
            else:
                videos_incomplete += 1
                videos_needing_scaling.append(idx)
            
            # Log progress every 100 videos or at the end
            if (idx + 1) % 100 == 0 or (idx + 1) == df.height:
                logger.info(f"Checked {idx + 1}/{df.height} videos... ({videos_complete} complete, {videos_incomplete} need scaling)")
                
        except Exception as e:
            logger.debug(f"Error checking video {video_rel}: {e}")
            videos_needing_scaling.append(idx)
            videos_incomplete += 1
            # Log progress even on errors
            if (idx + 1) % 100 == 0 or (idx + 1) == df.height:
                logger.info(f"Checked {idx + 1}/{df.height} videos... ({videos_complete} complete, {videos_incomplete} need scaling)")
    
    # Final summary log
    logger.info(f"PASS 1 complete: Checked all {df.height} videos ({videos_complete} complete, {videos_incomplete} need scaling)")
    
    return videos_needing_scaling, videos_complete, videos_incomplete


def main():
    """Run Stage 3: Video Scaling."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Scale videos to target max dimension (can downscale or upscale)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 224x224 with resolution method
  python src/scripts/run_stage3_scaling.py
  
  # Custom target size
  python src/scripts/run_stage3_scaling.py --target-size 112
  
  # Custom metadata path
  python src/scripts/run_stage3_scaling.py --augmented-metadata data/custom/augmented_metadata.csv
        """
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--augmented-metadata",
        type=str,
        default="data/augmented_videos/augmented_metadata.arrow",
        help="Path to augmented metadata from Stage 1 (default: data/augmented_videos/augmented_metadata.arrow). "
             "Supports .arrow, .parquet, or .csv formats. Will auto-detect format if extension is omitted."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="autoencoder",
        choices=["resolution", "autoencoder"],
        help="Scaling method (default: autoencoder). Use 'resolution' for letterbox resizing"
    )
    parser.add_argument(
        "--autoencoder-model",
        type=str,
        default=None,
        help="Hugging Face model name for autoencoder (default: stabilityai/sd-vae-ft-mse). "
             "Only used when --method=autoencoder"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Target max dimension. Videos will be scaled so max(width, height) = target_size (default: 256)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Number of frames to process per chunk (default: 400)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/scaled_videos",
        help="Output directory for scaled videos (default: data/scaled_videos)"
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
        help="Delete existing scaled video files before regenerating (clean mode, default: False, preserves existing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing scaled video files (skip already processed videos, default: True)"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode (process all videos, even if already scaled)"
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
    
    # Set resume default to True if neither --resume nor --no-resume was specified
    # argparse doesn't support default=True with store_true, so we handle it manually
    # Check if either flag was explicitly provided
    resume_provided = '--resume' in sys.argv or '--no-resume' in sys.argv
    if not resume_provided:
        args.resume = True
    
    # Input validation
    if not args.project_root or not isinstance(args.project_root, str):
        logger.error(f"project_root must be a non-empty string, got: {type(args.project_root)}")
        sys.exit(1)
    if not args.augmented_metadata or not isinstance(args.augmented_metadata, str):
        logger.error(f"augmented_metadata must be a non-empty string, got: {type(args.augmented_metadata)}")
        sys.exit(1)
    if not isinstance(args.target_size, int) or args.target_size <= 0:
        logger.error(f"target_size must be a positive integer, got: {args.target_size}")
        sys.exit(1)
    if not isinstance(args.chunk_size, int) or args.chunk_size <= 0:
        logger.error(f"chunk_size must be a positive integer, got: {args.chunk_size}")
        sys.exit(1)
    if args.method not in ["letterbox", "autoencoder", "resolution"]:
        logger.error(f"method must be 'letterbox', 'autoencoder', or 'resolution', got: {args.method}")
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
    
    augmented_metadata_path = project_root / args.augmented_metadata
    
    try:
        output_dir = project_root / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create output directory {args.output_dir}: {e}")
        sys.exit(1)
    
    # Logging setup - also log to file
    log_dir = project_root / "logs" / "stage3"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage3_scaling_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Start logging
    logger.info("=" * 80)
    logger.info("STAGE 3: VIDEO SCALING")
    logger.info("=" * 80)
    logger.info("Project root: %s", project_root)
    logger.info("Augmented metadata: %s", augmented_metadata_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Scaling method: %s", args.method)
    logger.info("Target max dimension: %d pixels", args.target_size)
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
    
    if not augmented_metadata_path.exists():
        logger.error("Augmented metadata file not found: %s", augmented_metadata_path)
        logger.error("Please run Stage 1 first: python src/scripts/run_stage1_augmentation.py")
        return 1
    logger.info("✓ Augmented metadata file found: %s", augmented_metadata_path)
    
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
    log_memory_stats("Stage 3: before scaling", detailed=True)
    
    # Run Stage 3
    stage_start = time.time()
    
    try:
        # Two-pass mode when --resume is enabled
        if args.resume:
            logger.info("=" * 80)
            logger.info("RESUME MODE: Two-Pass Scaling")
            logger.info("=" * 80)
            
            # Pass 1: Check which videos need scaling
            pass1_start = time.time()
            
            videos_needing_scaling, videos_complete, videos_incomplete = check_videos_needing_scaling(
                project_root=project_root,
                augmented_metadata_path=augmented_metadata_path,
                output_dir=output_dir,
                target_size=args.target_size,
                start_idx=args.start_idx,
                end_idx=args.end_idx
            )
            pass1_duration = time.time() - pass1_start
            
            logger.info("=" * 80)
            logger.info("PASS 1 COMPLETE: Scaling Status Check")
            logger.info("=" * 80)
            logger.info(f"Check duration: {pass1_duration:.2f} seconds ({pass1_duration / 60:.2f} minutes)")
            logger.info(f"Total videos checked: {videos_complete + videos_incomplete}")
            logger.info(f"Videos with scaled versions: {videos_complete}")
            logger.info(f"Videos needing scaling: {videos_incomplete}")
            logger.info("=" * 80)
            
            if videos_incomplete == 0:
                logger.info("All videos already have scaled versions!")
                logger.info("Running verification pass to ensure metadata completeness...")
                # Still run stage3_scale_videos to ensure metadata is complete
                result_df = stage3_scale_videos(
                    project_root=str(project_root),
                    augmented_metadata_path=str(augmented_metadata_path),
                    output_dir=args.output_dir,
                    method=args.method,
                    target_size=args.target_size,
                    chunk_size=args.chunk_size,
                    autoencoder_model=args.autoencoder_model,
                    start_idx=args.start_idx,
                    end_idx=args.end_idx,
                    delete_existing=args.delete_existing,
                    resume=args.resume,
                    execution_order=execution_order
                )
            else:
                logger.info("=" * 80)
                logger.info("PASS 2: Scaling videos that need processing...")
                logger.info("=" * 80)
                # Pass 2: Scale videos (stage3_scale_videos will automatically skip existing ones in resume mode)
                result_df = stage3_scale_videos(
                    project_root=str(project_root),
                    augmented_metadata_path=str(augmented_metadata_path),
                    output_dir=args.output_dir,
                    method=args.method,
                    target_size=args.target_size,
                    chunk_size=args.chunk_size,
                    autoencoder_model=args.autoencoder_model,
                    start_idx=args.start_idx,
                    end_idx=args.end_idx,
                    delete_existing=args.delete_existing,
                    resume=args.resume,
                    execution_order=execution_order
                )
        else:
            # Non-resume mode: scale all videos
            logger.info("=" * 80)
            logger.info("Starting Stage 3: Video Scaling")
            logger.info("=" * 80)
            logger.info("This may take a while depending on dataset size...")
            logger.info("Progress will be logged in real-time")
            logger.info("=" * 80)
            
            result_df = stage3_scale_videos(
            project_root=str(project_root),
            augmented_metadata_path=str(augmented_metadata_path),
            output_dir=args.output_dir,
            method=args.method,
            target_size=args.target_size,
            chunk_size=args.chunk_size,
            autoencoder_model=args.autoencoder_model,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            delete_existing=args.delete_existing,
            resume=args.resume,
            execution_order=execution_order
        )
        
        stage_duration = time.time() - stage_start
        
        logger.info("=" * 80)
        logger.info("STAGE 3 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Execution time: %.2f seconds (%.2f minutes)", 
                   stage_duration, stage_duration / 60)
        logger.info("Output directory: %s", output_dir)
        logger.info("Scaled metadata: %s", output_dir / "scaled_metadata.arrow")
        
        if result_df is not None and hasattr(result_df, 'height'):
            logger.info("Total videos processed: %d", result_df.height)
            logger.debug("Result DataFrame shape: %s", result_df.shape)
        else:
            logger.warning("Result DataFrame is None or invalid")
        
        # Log final memory stats
        logger.info("=" * 80)
        logger.info("Final memory statistics:")
        logger.info("=" * 80)
        log_memory_stats("Stage 3: after scaling", detailed=True)
        
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  - Run Stage 4: python src/scripts/run_stage4_scaled_features.py")
        logger.info("  - Or continue with full pipeline: python src/run_new_pipeline.py --skip-stage 1,2,3")
        logger.info("=" * 80)
        
        # Ensure all logs are flushed before exit
        sys.stdout.flush()
        sys.stderr.flush()
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("SCALING INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning("Partial results may be available in: %s", output_dir)
        logger.warning("You can resume by running the script again")
        return 130
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("STAGE 3 FAILED")
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

