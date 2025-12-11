#!/usr/bin/env python3
"""
Repair Stage 3: Re-scale corrupted videos identified by Stage 4.

Reads corrupted video file names from data/corrupted_scaled_videos.txt
and re-scales them using the same parameters as Stage 3.

Usage:
    python src/scripts/repair_stage3_scaling.py
    python src/scripts/repair_stage3_scaling.py --corrupted-list data/corrupted_scaled_videos.txt
    python src/scripts/repair_stage3_scaling.py --target-size 256 --method autoencoder
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import List, Set, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import polars as pl

from lib.scaling import stage3_scale_videos, scale_video
from lib.utils.paths import resolve_video_path, load_metadata_flexible, validate_video_file
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
logging.getLogger("lib.scaling").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)


def load_corrupted_files(corrupted_list_path: Path) -> List[str]:
    """
    Load corrupted video file paths from text file.
    
    Args:
        corrupted_list_path: Path to text file with one video path per line
    
    Returns:
        List of video paths (relative paths from metadata)
    """
    if not corrupted_list_path.exists():
        logger.warning(f"Corrupted files list not found: {corrupted_list_path}")
        return []
    
    corrupted_files = []
    try:
        with open(corrupted_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    corrupted_files.append(line)
        
        logger.info(f"Loaded {len(corrupted_files)} corrupted file(s) from {corrupted_list_path}")
        return corrupted_files
    except Exception as e:
        logger.error(f"Failed to load corrupted files list: {e}")
        return []


def extract_video_info_from_scaled_path(scaled_path: str) -> Optional[dict]:
    """
    Extract video information from scaled video path.
    
    Handles formats:
    - {video_id}_scaled_original.mp4
    - {video_id}_scaled_aug{N}.mp4
    - data/scaled_videos/{video_id}_scaled_*.mp4
    
    Args:
        scaled_path: Path to scaled video (relative or absolute)
    
    Returns:
        Dictionary with keys: video_id, aug_idx, is_original, or None if cannot parse
    """
    from pathlib import Path
    
    # Extract filename
    filename = Path(scaled_path).name
    stem = Path(filename).stem  # Remove .mp4 extension
    
    if stem.endswith("_scaled_original"):
        video_id = stem[:-16]  # Remove "_scaled_original"
        return {
            "video_id": video_id,
            "is_original": True,
            "aug_idx": -1
        }
    elif "_scaled_aug" in stem:
        parts = stem.split("_scaled_aug")
        if len(parts) == 2:
            video_id = parts[0]
            try:
                aug_idx = int(parts[1])
                return {
                    "video_id": video_id,
                    "is_original": False,
                    "aug_idx": aug_idx
                }
            except ValueError:
                return None
    return None


def find_videos_in_metadata(
    corrupted_paths: List[str],
    augmented_metadata_path: str,
    project_root: Path
) -> pl.DataFrame:
    """
    Find corrupted videos in augmented metadata and create a filtered DataFrame.
    
    Args:
        corrupted_paths: List of corrupted video paths (relative paths from scaled metadata)
        augmented_metadata_path: Path to augmented metadata
        project_root: Project root directory
    
    Returns:
        DataFrame containing only the videos that need to be re-scaled
    """
    # Load augmented metadata
    df = load_metadata_flexible(augmented_metadata_path)
    if df is None:
        logger.error(f"Augmented metadata not found: {augmented_metadata_path}")
        return pl.DataFrame()
    
    # Extract video info from corrupted scaled paths
    corrupted_video_info = {}
    for path in corrupted_paths:
        info = extract_video_info_from_scaled_path(path)
        if info:
            key = (info["video_id"], info["aug_idx"])
            corrupted_video_info[key] = info
        else:
            logger.warning(f"Could not parse corrupted path: {path}")
    
    logger.info(f"Looking for {len(corrupted_video_info)} corrupted video(s) in metadata...")
    
    # Find matching videos in augmented metadata
    # Match by video_id and aug_idx
    matching_rows = []
    for row in df.iter_rows(named=True):
        video_path = row.get("video_path", "")
        if not video_path:
            continue
        
        # Extract video_id from augmented video path
        video_id = Path(video_path).stem
        # Remove augmentation suffix if present (e.g., "video_id_aug1" -> "video_id")
        if "_aug" in video_id:
            base_video_id = video_id.split("_aug")[0]
            try:
                aug_idx = int(video_id.split("_aug")[1])
            except (ValueError, IndexError):
                aug_idx = -1
        else:
            base_video_id = video_id
            aug_idx = row.get("augmentation_idx", -1)
        
        # Check if this matches any corrupted video
        key = (base_video_id, aug_idx)
        if key in corrupted_video_info:
            matching_rows.append(row)
            continue
        
        # Also try matching by video_id only (in case aug_idx differs)
        for (corrupted_id, corrupted_aug_idx), corrupted_info in corrupted_video_info.items():
            if base_video_id == corrupted_id:
                # If aug_idx matches or both are -1/original, it's a match
                if (aug_idx == corrupted_aug_idx) or (aug_idx == -1 and corrupted_aug_idx == -1):
                    matching_rows.append(row)
                    break
    
    if not matching_rows:
        logger.warning("No matching videos found in augmented metadata for corrupted files")
        logger.debug(f"Corrupted video info: {list(corrupted_video_info.keys())[:10]}")
        logger.debug(f"Metadata video_paths sample: {df['video_path'].head(5).to_list()}")
        return pl.DataFrame()
    
    result_df = pl.DataFrame(matching_rows)
    logger.info(f"Found {result_df.height} video(s) in metadata matching corrupted files")
    
    return result_df


def repair_corrupted_videos(
    project_root: str,
    corrupted_list_path: str,
    augmented_metadata_path: str,
    output_dir: str = "data/scaled_videos",
    target_size: int = 256,
    max_frames: Optional[int] = 500,
    chunk_size: int = 400,
    method: str = "autoencoder",
    autoencoder_model: Optional[str] = None,
    delete_corrupted: bool = True
) -> bool:
    """
    Repair corrupted scaled videos by re-scaling them.
    
    Args:
        project_root: Project root directory
        corrupted_list_path: Path to text file with corrupted video paths
        augmented_metadata_path: Path to augmented metadata
        output_dir: Directory to save scaled videos
        target_size: Target max dimension
        max_frames: Maximum frames to process per video
        chunk_size: Number of frames per chunk
        method: Scaling method ("letterbox" or "autoencoder")
        autoencoder_model: Hugging Face model name for autoencoder
        delete_corrupted: If True, delete corrupted scaled video files before re-scaling
    
    Returns:
        True if repair was successful, False otherwise
    """
    project_root = Path(project_root)
    corrupted_list_path = project_root / corrupted_list_path
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("REPAIR STAGE 3: RE-SCALING CORRUPTED VIDEOS")
    logger.info("=" * 80)
    
    # Load corrupted files list
    corrupted_paths = load_corrupted_files(corrupted_list_path)
    if not corrupted_paths:
        logger.warning("No corrupted files to repair")
        return False
    
    logger.info(f"Found {len(corrupted_paths)} corrupted video(s) to repair")
    
    # Find matching videos in augmented metadata
    videos_to_repair = find_videos_in_metadata(
        corrupted_paths,
        str(augmented_metadata_path),
        project_root
    )
    
    if videos_to_repair.height == 0:
        logger.error("No videos found in metadata to repair")
        return False
    
    logger.info(f"Repairing {videos_to_repair.height} video(s)")
    logger.info(f"Target size: {target_size}x{target_size}")
    logger.info(f"Method: {method}")
    
    # Map "resolution" to "letterbox" for backward compatibility
    if method == "resolution":
        method = "letterbox"
    
    # Load autoencoder if needed
    autoencoder = None
    if method == "autoencoder":
        try:
            model_name = autoencoder_model or "stabilityai/sd-vae-ft-mse"
            logger.info(f"Loading Hugging Face autoencoder: {model_name}")
            from lib.scaling.methods import load_autoencoder
            autoencoder = load_autoencoder(model_name)
            logger.info("Autoencoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load autoencoder: {e}")
            logger.warning("Falling back to letterbox method")
            method = "letterbox"
    
    # Process each video
    repaired_count = 0
    failed_count = 0
    
    for idx, row in enumerate(videos_to_repair.iter_rows(named=True)):
        video_rel = row["video_path"]
        label = row.get("label", "unknown")
        
        logger.info(f"Repairing video {idx + 1}/{videos_to_repair.height}: {video_rel}")
        
        try:
            # Resolve original video path
            original_video_path = resolve_video_path(video_rel, str(project_root))
            
            if not Path(original_video_path).exists():
                logger.warning(f"Original video not found: {original_video_path}, skipping")
                failed_count += 1
                continue
            
            # Validate original video
            is_valid, error_msg = validate_video_file(original_video_path, check_decode=True)
            if not is_valid:
                logger.warning(f"Original video is corrupted: {original_video_path} - {error_msg}, skipping")
                failed_count += 1
                continue
            
            # Determine output path for scaled video
            video_id = Path(original_video_path).stem
            # Handle augmentation suffix if present
            aug_idx = row.get("augmentation_idx", -1)
            if aug_idx >= 0:
                output_filename = f"{video_id}_aug{aug_idx}_scaled.mp4"
            else:
                output_filename = f"{video_id}_scaled.mp4"
            
            output_path = output_dir / output_filename
            
            # Delete corrupted scaled video if it exists
            if delete_corrupted and output_path.exists():
                try:
                    # Validate if it's actually corrupted
                    is_valid_scaled, _ = validate_video_file(output_path, check_decode=True)
                    if not is_valid_scaled:
                        logger.info(f"Deleting corrupted scaled video: {output_path}")
                        output_path.unlink()
                    else:
                        logger.info(f"Scaled video appears valid, skipping: {output_path}")
                        repaired_count += 1
                        continue
                except Exception as e:
                    logger.warning(f"Could not validate/delete existing scaled video: {e}")
                    # Try to delete anyway
                    try:
                        output_path.unlink()
                    except Exception:
                        pass
            
            # Re-scale the video
            logger.info(f"Re-scaling: {original_video_path} -> {output_path}")
            success = scale_video(
                video_path=str(original_video_path),
                output_path=str(output_path),
                target_size=target_size,
                max_frames=max_frames,
                chunk_size=chunk_size,
                method=method,
                autoencoder=autoencoder
            )
            
            if success:
                # Validate the newly scaled video
                is_valid_new, error_msg = validate_video_file(output_path, check_decode=True)
                if is_valid_new:
                    logger.info(f"✓ Successfully repaired: {output_path}")
                    repaired_count += 1
                else:
                    logger.error(f"✗ Re-scaled video is still corrupted: {output_path} - {error_msg}")
                    failed_count += 1
            else:
                logger.error(f"✗ Failed to re-scale: {original_video_path}")
                failed_count += 1
            
            # Memory cleanup
            if (idx + 1) % 10 == 0:
                log_memory_stats(f"Repair: processed {idx + 1}/{videos_to_repair.height} videos")
            
        except Exception as e:
            logger.error(f"Error repairing {video_rel}: {e}", exc_info=True)
            failed_count += 1
            continue
    
    logger.info("=" * 80)
    logger.info(f"REPAIR COMPLETE: {repaired_count} repaired, {failed_count} failed")
    logger.info("=" * 80)
    
    # Clear corrupted files list if all repairs successful
    if repaired_count > 0 and failed_count == 0:
        try:
            # Clear the corrupted files list (or move to archive)
            archive_path = corrupted_list_path.with_suffix('.txt.repaired')
            corrupted_list_path.rename(archive_path)
            logger.info(f"Moved corrupted files list to: {archive_path}")
        except Exception as e:
            logger.warning(f"Could not archive corrupted files list: {e}")
    
    return repaired_count > 0


def main():
    """Run Repair Stage 3: Re-scale corrupted videos."""
    parser = argparse.ArgumentParser(
        description="Repair Stage 3: Re-scale corrupted videos identified by Stage 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Repair using default settings
    python src/scripts/repair_stage3_scaling.py
    
    # Repair with custom corrupted list
    python src/scripts/repair_stage3_scaling.py --corrupted-list data/corrupted_scaled_videos.txt
    
    # Repair with specific method and target size
    python src/scripts/repair_stage3_scaling.py --method letterbox --target-size 224
        """
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default=str(project_root),
        help='Project root directory (default: auto-detect)'
    )
    
    parser.add_argument(
        '--corrupted-list',
        type=str,
        default='data/corrupted_scaled_videos.txt',
        help='Path to corrupted files list (default: data/corrupted_scaled_videos.txt)'
    )
    
    parser.add_argument(
        '--augmented-metadata',
        type=str,
        default='data/augmented_videos/augmented_metadata',
        help='Path to augmented metadata (default: data/augmented_videos/augmented_metadata)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/scaled_videos',
        help='Output directory for scaled videos (default: data/scaled_videos)'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        default=256,
        help='Target max dimension (default: 256)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['letterbox', 'autoencoder', 'resolution'],
        default='autoencoder',
        help='Scaling method (default: autoencoder)'
    )
    
    parser.add_argument(
        '--autoencoder-model',
        type=str,
        default=None,
        help='Hugging Face autoencoder model name (default: stabilityai/sd-vae-ft-mse)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=500,
        help='Maximum frames per video (default: 500)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=400,
        help='Frames per chunk (default: 400)'
    )
    
    parser.add_argument(
        '--delete-corrupted',
        action='store_true',
        default=True,
        help='Delete corrupted scaled videos before re-scaling (default: True)'
    )
    
    parser.add_argument(
        '--keep-corrupted',
        action='store_true',
        help='Keep corrupted scaled videos (do not delete before re-scaling)'
    )
    
    args = parser.parse_args()
    
    # Handle delete_corrupted flag
    delete_corrupted = args.delete_corrupted and not args.keep_corrupted
    
    # Run repair
    start_time = time.time()
    success = repair_corrupted_videos(
        project_root=args.project_root,
        corrupted_list_path=args.corrupted_list,
        augmented_metadata_path=args.augmented_metadata,
        output_dir=args.output_dir,
        target_size=args.target_size,
        max_frames=args.max_frames,
        chunk_size=args.chunk_size,
        method=args.method,
        autoencoder_model=args.autoencoder_model,
        delete_corrupted=delete_corrupted
    )
    
    duration = time.time() - start_time
    logger.info(f"Repair completed in {duration:.2f} seconds")
    
    if success:
        logger.info("✓ Repair successful")
        sys.exit(0)
    else:
        logger.error("✗ Repair failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

