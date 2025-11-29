"""
Video downscaling pipeline.

Downscales videos to target resolutions using letterbox resizing or other
methods while preserving aspect ratios.
"""

from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import av

from lib.data import load_metadata
from lib.utils.paths import resolve_video_path
from lib.utils.memory import aggressive_gc, log_memory_stats
from lib.downscaling.methods import downscale_video_frames, letterbox_resize
from lib.augmentation.io import load_frames, save_frames

logger = logging.getLogger(__name__)


def downscale_video(
    video_path: str,
    output_path: str,
    target_size: int = 224,
    max_frames: int = 1000,
    method: str = "letterbox"
) -> bool:
    """
    Downscale a single video.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        target_size: Target size for downscaling
        max_frames: Maximum frames to process
        method: Downscaling method ("letterbox" or "autoencoder")
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load frames
        frames, fps = load_frames(video_path, max_frames=max_frames)
        
        if not frames:
            logger.warning(f"No frames loaded from {video_path}")
            return False
        
        # Downscale frames
        downscaled_frames = []
        for frame in frames:
            downscaled_frame = letterbox_resize(frame, target_size)
            downscaled_frames.append(downscaled_frame)
        
        # Save downscaled video
        success = save_frames(downscaled_frames, output_path, fps=fps)
        
        # Clear memory
        del frames, downscaled_frames
        aggressive_gc(clear_cuda=False)
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to downscale video {video_path}: {e}")
        return False


def stage3_downscale_videos(
    project_root: str,
    augmented_metadata_path: str,
    output_dir: str = "data/downscaled_videos",
    target_size: int = 224,
    max_frames: int = 1000,
    method: str = "letterbox"
) -> pl.DataFrame:
    """
    Stage 3: Downscale all videos.
    
    Args:
        project_root: Project root directory
        augmented_metadata_path: Path to augmented metadata CSV
        output_dir: Directory to save downscaled videos
        target_size: Target size for downscaling (default: 224)
        max_frames: Maximum frames to process per video
        method: Downscaling method ("letterbox" or "autoencoder")
    
    Returns:
        DataFrame with downscaled video metadata
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load augmented metadata
    logger.info("Stage 3: Loading augmented metadata...")
    if not Path(augmented_metadata_path).exists():
        logger.error(f"Augmented metadata not found: {augmented_metadata_path}")
        return pl.DataFrame()
    
    df = pl.read_csv(augmented_metadata_path)
    logger.info(f"Stage 3: Processing {df.height} videos")
    logger.info(f"Stage 3: Target size: {target_size}x{target_size}")
    logger.info(f"Stage 3: Method: {method}")
    
    # Use incremental CSV writing
    metadata_path = output_dir / "downscaled_metadata.csv"
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label", "original_video", "augmentation_idx", "is_original"])
    
    total_videos_processed = 0
    
    # Process each video
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        original_video = row.get("original_video", video_rel)
        aug_idx = row.get("augmentation_idx", -1)
        is_original = row.get("is_original", False)
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            if idx % 10 == 0:
                log_memory_stats(f"Stage 3: processing video {idx + 1}/{df.height}")
            
            # Create output path
            video_id = Path(video_path).stem
            if is_original:
                output_filename = f"{video_id}_downscaled_original.mp4"
            else:
                output_filename = f"{video_id}_downscaled_aug{aug_idx}.mp4"
            
            output_path = output_dir / output_filename
            
            # Skip if already exists
            if output_path.exists():
                logger.debug(f"Downscaled video already exists: {output_path}")
                output_rel = str(output_path.relative_to(project_root))
                with open(metadata_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        output_rel,
                        label,
                        original_video,
                        aug_idx,
                        is_original
                    ])
                total_videos_processed += 1
                continue
            
            # Downscale video
            logger.info(f"Downscaling {Path(video_path).name} to {output_path.name}")
            success = downscale_video(
                video_path,
                str(output_path),
                target_size=target_size,
                max_frames=max_frames,
                method=method
            )
            
            if success:
                output_rel = str(output_path.relative_to(project_root))
                with open(metadata_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        output_rel,
                        label,
                        original_video,
                        aug_idx,
                        is_original
                    ])
                total_videos_processed += 1
                logger.info(f"✓ Downscaled: {output_path.name}")
            else:
                logger.error(f"✗ Failed to downscale: {video_path}")
            
            aggressive_gc(clear_cuda=False)
            
        except Exception as e:
            logger.error(f"Error processing {video_rel}: {e}", exc_info=True)
            continue
    
    # Load final metadata
    if metadata_path.exists() and total_videos_processed > 0:
        try:
            metadata_df = pl.read_csv(str(metadata_path))
            logger.info(f"\n✓ Stage 3 complete: Saved metadata to {metadata_path}")
            logger.info(f"✓ Stage 3: Downscaled {total_videos_processed} videos")
            return metadata_df
        except Exception as e:
            logger.error(f"Failed to read metadata CSV: {e}")
            return pl.DataFrame()
    else:
        logger.error("Stage 3: No videos processed!")
        return pl.DataFrame()

