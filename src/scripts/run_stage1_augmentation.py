#!/usr/bin/env python3
"""
Stage 1: Video Augmentation Script

Generates multiple augmented versions of each video using spatial and temporal
transformations. Creates augmented clips for training data augmentation.

Usage:
    python src/scripts/run_stage1_augmentation.py
    python src/scripts/run_stage1_augmentation.py --num-augmentations 10
    python src/scripts/run_stage1_augmentation.py --project-root /path/to/project
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

from lib.augmentation import stage1_augment_videos
from lib.utils.memory import log_memory_stats
from lib.data import load_metadata, filter_existing_videos
from lib.utils.paths import resolve_video_path, load_metadata_flexible
import polars as pl

# Setup extensive logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level for extensive logs
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger("lib").setLevel(logging.DEBUG)
logging.getLogger("lib.augmentation").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)


def check_videos_needing_augmentations(
    project_root: Path,
    output_dir: Path,
    num_augmentations: int,
    start_idx: int = None,
    end_idx: int = None
) -> tuple[list[int], int, int]:
    """
    Pass 1: Check which videos need augmentations.
    
    Returns:
        (videos_needing_augmentations, videos_complete, videos_incomplete)
        - videos_needing_augmentations: List of indices in the dataframe that need augmentations
        - videos_complete: Count of videos with all augmentations
        - videos_incomplete: Count of videos needing augmentations
    """
    logger.info("PASS 1: Checking which videos need augmentations...")
    
    # Load input metadata
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            break
    
    if input_metadata_path is None:
        raise FileNotFoundError("Metadata file not found for resume mode")
    
    df = load_metadata(str(input_metadata_path))
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
    
    # Check existing metadata
    metadata_path_arrow = output_dir / "augmented_metadata.arrow"
    metadata_path_parquet = output_dir / "augmented_metadata.parquet"
    metadata_path_csv = output_dir / "augmented_metadata.csv"
    
    metadata_path = None
    if metadata_path_arrow.exists():
        metadata_path = metadata_path_arrow
    elif metadata_path_parquet.exists():
        metadata_path = metadata_path_parquet
    elif metadata_path_csv.exists():
        metadata_path = metadata_path_csv
    
    # Load existing metadata to check what's already done
    existing_video_ids_with_all_augs = set()
    if metadata_path and metadata_path.exists():
        try:
            if metadata_path.suffix == '.arrow':
                existing_metadata = pl.read_ipc(metadata_path)
            elif metadata_path.suffix == '.parquet':
                existing_metadata = pl.read_parquet(metadata_path)
            else:
                existing_metadata = pl.read_csv(str(metadata_path))
            
            # Count augmentations per video
            video_aug_counts = {}
            for row in existing_metadata.iter_rows(named=True):
                original_video = row.get("original_video", "")
                aug_idx = row.get("augmentation_idx", -1)
                if aug_idx >= 0:
                    video_path_obj = Path(original_video)
                    if len(video_path_obj.parts) >= 2:
                        video_id = video_path_obj.parts[-2]
                        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                        video_aug_counts[video_id] = video_aug_counts.get(video_id, 0) + 1
            
            for video_id, count in video_aug_counts.items():
                if count >= num_augmentations:
                    existing_video_ids_with_all_augs.add(video_id)
        except Exception as e:
            logger.warning(f"Could not load existing metadata: {e}")
    
    # Pass 1: Identify videos that need augmentations
    videos_needing_augmentations = []
    videos_complete = 0
    videos_incomplete = 0
    
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            if not Path(video_path).exists():
                videos_needing_augmentations.append(idx)
                videos_incomplete += 1
                continue
            
            # Extract video_id (same logic as in stage1_augment_videos)
            video_path_obj = Path(video_path)
            if len(video_path_obj.parts) >= 2:
                video_id = video_path_obj.parts[-2]
            else:
                import hashlib
                video_id = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
            
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            
            # Check if all augmentations exist (both in metadata and as files)
            all_augmentations_exist = True
            if video_id in existing_video_ids_with_all_augs:
                # Double-check files actually exist on disk
                for aug_idx in range(num_augmentations):
                    aug_path = output_dir / f"{video_id}_aug{aug_idx}.mp4"
                    if not aug_path.exists():
                        all_augmentations_exist = False
                        break
            else:
                all_augmentations_exist = False
            
            if all_augmentations_exist:
                videos_complete += 1
                if (idx + 1) % 100 == 0:
                    logger.info(f"Checked {idx + 1}/{df.height} videos... ({videos_complete} complete, {videos_incomplete} need augmentations)")
            else:
                videos_incomplete += 1
                videos_needing_augmentations.append(idx)
                
        except Exception as e:
            logger.debug(f"Error checking video {video_rel}: {e}")
            videos_needing_augmentations.append(idx)
            videos_incomplete += 1
    
    return videos_needing_augmentations, videos_complete, videos_incomplete


def main():
    """Run Stage 1: Video Augmentation."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate augmented versions of videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 10 augmentations per video
  python src/scripts/run_stage1_augmentation.py
  
  # Custom number of augmentations
  python src/scripts/run_stage1_augmentation.py --num-augmentations 5
  
  # Custom project root and output directory
  python src/scripts/run_stage1_augmentation.py --project-root /path/to/project --output-dir data/custom_augmented
        """
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=10,
        help="Number of augmentations to generate per video (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented_videos",
        help="Output directory for augmented videos (default: data/augmented_videos)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Two-pass mode: First pass checks which videos need augmentations, second pass generates only missing augmentations"
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing augmentations before regenerating (default: False, preserves existing)"
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
    
    args = parser.parse_args()
    
    # Convert to Path objects
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    
    # Logging setup - also log to file
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage1_augmentation_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Start logging
    logger.info("=" * 80)
    logger.info("STAGE 1: VIDEO AUGMENTATION")
    logger.info("=" * 80)
    logger.info("Project root: %s", project_root)
    logger.info("Output directory: %s", output_dir)
    logger.info("Number of augmentations per video: %d", args.num_augmentations)
    logger.info("Resume mode: %s", "Enabled" if args.resume else "Disabled")
    logger.info("Delete existing augmentations: %s", "Yes" if args.delete_existing else "No (preserved)")
    if args.start_idx is not None or args.end_idx is not None:
        logger.info("Video range: [%s, %s)", 
                   args.start_idx if args.start_idx is not None else "0",
                   args.end_idx if args.end_idx is not None else "all")
    logger.info("Log file: %s", log_file)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.debug("Command line arguments: %s", sys.argv)
    
    # Check prerequisites
    logger.info("=" * 80)
    logger.info("Checking prerequisites...")
    logger.info("=" * 80)
    
    # Check metadata file - prefer FVC_dup.csv, fallback to video_index_input.csv
    metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            metadata_path = candidate_path
            logger.info("✓ Metadata file found: %s", metadata_path)
            break
    
    if metadata_path is None:
        logger.error("Metadata file not found. Expected:")
        logger.error("  - %s", project_root / "data" / "FVC_dup.csv")
        logger.error("  - %s", project_root / "data" / "video_index_input.csv")
        logger.error("Please run data preparation first: python src/setup_fvc_dataset.py")
        return 1
    
    # Check videos directory
    videos_dir = project_root / "videos"
    if not videos_dir.exists():
        logger.warning("Videos directory not found: %s", videos_dir)
        logger.warning("Video path resolution may fail if videos are in different location")
    else:
        logger.info("✓ Videos directory found: %s", videos_dir)
    
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
    log_memory_stats("Stage 1: before augmentation", detailed=True)
    
    # Run Stage 1
    logger.info("=" * 80)
    logger.info("Starting Stage 1: Video Augmentation")
    logger.info("=" * 80)
    logger.info("This may take a while depending on dataset size...")
    logger.info("Progress will be logged in real-time")
    logger.info("=" * 80)
    
    stage_start = time.time()
    
    try:
        # Two-pass mode when --resume is enabled
        if args.resume:
            logger.info("=" * 80)
            logger.info("RESUME MODE: Two-Pass Augmentation")
            logger.info("=" * 80)
            
            # Pass 1: Check which videos need augmentations
            pass1_start = time.time()
            videos_needing_augmentations, videos_complete, videos_incomplete = check_videos_needing_augmentations(
                project_root=project_root,
                output_dir=output_dir,
                num_augmentations=args.num_augmentations,
                start_idx=args.start_idx,
                end_idx=args.end_idx
            )
            pass1_duration = time.time() - pass1_start
            
            logger.info("=" * 80)
            logger.info("PASS 1 COMPLETE: Augmentation Status Check")
            logger.info("=" * 80)
            logger.info(f"Check duration: {pass1_duration:.2f} seconds ({pass1_duration / 60:.2f} minutes)")
            logger.info(f"Total videos checked: {videos_complete + videos_incomplete}")
            logger.info(f"Videos with complete augmentations: {videos_complete}")
            logger.info(f"Videos needing augmentations: {videos_incomplete}")
            logger.info("=" * 80)
            
            if videos_incomplete == 0:
                logger.info("All videos already have augmentations. Nothing to do.")
                return 0
            
            # Pass 2: Generate augmentations only for videos that need them
            logger.info("=" * 80)
            logger.info("PASS 2: Generating augmentations for %d videos...", videos_incomplete)
            logger.info("=" * 80)
            
            # Load input metadata to create filtered version
            input_metadata_path = None
            for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
                candidate_path = project_root / "data" / csv_name
                if candidate_path.exists():
                    input_metadata_path = candidate_path
                    break
            
            if input_metadata_path is None:
                logger.error("Metadata file not found for resume mode")
                return 1
            
            df = load_metadata(str(input_metadata_path))
            df = filter_existing_videos(df, str(project_root))
            
            # Apply range filtering if specified
            total_videos = df.height
            if args.start_idx is not None or args.end_idx is not None:
                start = args.start_idx if args.start_idx is not None else 0
                end = args.end_idx if args.end_idx is not None else total_videos
                if start < 0:
                    start = 0
                if end > total_videos:
                    end = total_videos
                if start >= end:
                    logger.warning(f"Invalid range: start_idx={start}, end_idx={end}")
                    return 1
                df = df.slice(start, end - start)
            
            # Create filtered dataframe with only videos that need augmentations
            # Adjust indices if we're in a range
            base_idx = args.start_idx if args.start_idx is not None else 0
            adjusted_indices = [base_idx + idx for idx in videos_needing_augmentations]
            df_to_process = df.filter(pl.int_range(0, df.height).is_in(videos_needing_augmentations))
            
            # Save filtered metadata temporarily
            temp_metadata_path = project_root / "data" / ".temp_stage1_resume_metadata.csv"
            original_metadata_backup = None
            try:
                df_to_process.write_csv(temp_metadata_path)
                logger.info(f"Created temporary metadata file with {df_to_process.height} videos needing augmentations")
                
                # Temporarily backup and replace the original metadata file
                original_metadata_backup = project_root / "data" / f".backup_{input_metadata_path.name}"
                import shutil
                shutil.copy2(input_metadata_path, original_metadata_backup)
                shutil.copy2(temp_metadata_path, input_metadata_path)
                
                # Now call stage1_augment_videos - it will use the filtered metadata
                result_df = stage1_augment_videos(
                    project_root=str(project_root),
                    num_augmentations=args.num_augmentations,
                    output_dir=args.output_dir,
                    delete_existing=False,  # Never delete in resume mode
                    start_idx=None,  # Process all videos in filtered list
                    end_idx=None
                )
                
            except Exception as e:
                logger.error(f"Error in resume mode pass 2: {e}", exc_info=True)
                raise
            finally:
                # Always restore original metadata file and clean up
                if original_metadata_backup and original_metadata_backup.exists():
                    try:
                        shutil.copy2(original_metadata_backup, input_metadata_path)
                        original_metadata_backup.unlink()
                    except Exception as e:
                        logger.warning(f"Could not restore original metadata: {e}")
                # Clean up temp file
                if temp_metadata_path.exists():
                    try:
                        temp_metadata_path.unlink()
                    except Exception as e:
                        logger.warning(f"Could not remove temp metadata: {e}")
        else:
            # Normal single-pass mode
            result_df = stage1_augment_videos(
                project_root=str(project_root),
                num_augmentations=args.num_augmentations,
                output_dir=args.output_dir,
                delete_existing=args.delete_existing,
                start_idx=args.start_idx,
                end_idx=args.end_idx
            )
        
        stage_duration = time.time() - stage_start
        
        logger.info("=" * 80)
        logger.info("STAGE 1 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Execution time: %.2f seconds (%.2f minutes)", 
                   stage_duration, stage_duration / 60)
        logger.info("Output directory: %s", output_dir)
        logger.info("Metadata file: %s", output_dir / "augmented_metadata.csv")
        
        if result_df is not None and hasattr(result_df, 'height'):
            logger.info("Total videos processed: %d", result_df.height)
            logger.debug("Result DataFrame shape: %s", result_df.shape)
            
            # Log statistics
            try:
                if "is_original" in result_df.columns:
                    original_count = result_df.filter(pl.col("is_original") == True).height
                    augmented_count = result_df.filter(pl.col("is_original") == False).height
                    logger.info("Original videos: %d", original_count)
                    logger.info("Augmented videos: %d", augmented_count)
                    logger.info("Total videos (original + augmented): %d", result_df.height)
            except Exception as e:
                logger.debug("Could not compute statistics: %s", e)
        else:
            logger.warning("Result DataFrame is None or invalid")
        
        # Log final memory stats
        logger.info("=" * 80)
        logger.info("Final memory statistics:")
        logger.info("=" * 80)
        log_memory_stats("Stage 1: after augmentation", detailed=True)
        
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  - Run Stage 2: python src/scripts/run_stage2_features.py")
        logger.info("  - Or continue with full pipeline: python src/run_new_pipeline.py --skip-stage 1")
        logger.info("=" * 80)
        
        # Ensure all logs are flushed before exit
        sys.stdout.flush()
        sys.stderr.flush()
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("AUGMENTATION INTERRUPTED BY USER")
        logger.warning("=" * 80)
        logger.warning("Partial results may be available in: %s", output_dir)
        logger.warning("You can resume by running the script again")
        return 130
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("STAGE 1 FAILED")
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

