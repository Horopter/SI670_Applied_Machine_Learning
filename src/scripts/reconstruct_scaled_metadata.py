#!/usr/bin/env python3
"""
Reconstruct scaled metadata from scaled video files.

Scans the scaled_videos directory and creates metadata file if it doesn't exist.
Tries to get labels from augmented_metadata if available.

Usage:
    python src/scripts/reconstruct_scaled_metadata.py
    python src/scripts/reconstruct_scaled_metadata.py --scaled-videos-dir data/scaled_videos
"""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import av
from lib.utils.paths import write_metadata_atomic, load_metadata_flexible

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_video_info_from_filename(filename: str) -> dict:
    """
    Extract video information from scaled video filename.
    
    Handles formats:
    - {video_id}_scaled_original.mp4
    - {video_id}_scaled_aug{N}.mp4
    
    Args:
        filename: Video filename (e.g., "07kv4UTfqgE_scaled_original.mp4")
    
    Returns:
        Dictionary with keys: video_id, is_original, aug_idx, expected_path
    """
    stem = Path(filename).stem  # Remove .mp4 extension
    
    if stem.endswith("_scaled_original"):
        video_id = stem[:-16]  # Remove "_scaled_original"
        return {
            "video_id": video_id,
            "is_original": True,
            "aug_idx": -1,
            "expected_path": filename
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
                    "aug_idx": aug_idx,
                    "expected_path": filename
                }
            except ValueError:
                logger.warning(f"Could not parse augmentation index from {filename}")
                return None
    else:
        logger.warning(f"Unexpected filename format: {filename}")
        return None


def reconstruct_scaled_metadata(
    project_root: str,
    scaled_videos_dir: str = "data/scaled_videos",
    augmented_metadata_path: str = None,
    output_path: str = None
) -> bool:
    """
    Reconstruct scaled metadata from scaled video files.
    
    Args:
        project_root: Project root directory
        scaled_videos_dir: Directory containing scaled videos (default: data/scaled_videos)
        augmented_metadata_path: Path to augmented metadata to get labels (optional)
        output_path: Path to save reconstructed metadata (default: auto-detect)
    
    Returns:
        True if successful, False otherwise
    """
    project_root = Path(project_root)
    scaled_videos_path = project_root / scaled_videos_dir
    
    logger.info("=" * 80)
    logger.info("RECONSTRUCTING SCALED METADATA")
    logger.info("=" * 80)
    logger.info(f"Scanning scaled videos directory: {scaled_videos_path}")
    
    if not scaled_videos_path.exists():
        logger.error(f"✗ Scaled videos directory not found: {scaled_videos_path}")
        return False
    
    # Find all scaled video files
    scaled_files = list(scaled_videos_path.glob("*_scaled*.mp4"))
    logger.info(f"Found {len(scaled_files)} scaled video files in directory")
    
    if len(scaled_files) == 0:
        logger.warning("⚠ No scaled video files found in directory")
        return False
    
    # Load augmented metadata to get labels if available
    label_map = {}  # Maps (video_id, aug_idx) to label
    original_video_map = {}  # Maps video_id to original_video path
    
    if augmented_metadata_path:
        aug_path = project_root / augmented_metadata_path if not Path(augmented_metadata_path).is_absolute() else Path(augmented_metadata_path)
        logger.info(f"Loading augmented metadata for labels: {aug_path}")
        aug_df = load_metadata_flexible(str(aug_path))
        
        if aug_df is not None and aug_df.height > 0:
            # Extract video_id from video_path
            for row in aug_df.iter_rows(named=True):
                video_path = row.get("video_path", "")
                label = row.get("label", "unknown")
                original_video = row.get("original_video", "")
                aug_idx = row.get("augmentation_idx", -1)
                
                # Extract video_id from path (try multiple methods)
                path_obj = Path(video_path)
                video_id = None
                
                # Method 1: Parent directory name
                if len(path_obj.parts) >= 2:
                    video_id = path_obj.parts[-2]
                # Method 2: Filename without extension and suffix
                if not video_id:
                    stem = path_obj.stem
                    if stem.endswith("_original"):
                        video_id = stem[:-9]
                    elif "_aug" in stem:
                        video_id = stem.split("_aug")[0]
                    else:
                        video_id = stem
                
                if video_id:
                    # Sanitize video_id to match how Stage 3 processes it
                    video_id = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in video_id)
                    label_map[(video_id, aug_idx)] = label
                    if original_video:
                        original_video_map[video_id] = original_video
            
            logger.info(f"Loaded {len(label_map)} label mappings from augmented metadata")
        else:
            logger.warning("Could not load augmented metadata, will use 'unknown' for labels")
    else:
        logger.info("No augmented metadata path provided, will use 'unknown' for labels")
    
    # Extract video info from filenames and build metadata
    metadata_rows = []
    
    for video_file in scaled_files:
        video_info = extract_video_info_from_filename(video_file.name)
        if not video_info:
            continue
        
        rel_path = str(video_file.relative_to(project_root))
        video_id = video_info["video_id"]
        aug_idx = video_info["aug_idx"]
        
        # Get label from augmented metadata if available
        label = label_map.get((video_id, aug_idx), "unknown")
        original_video = original_video_map.get(video_id, "unknown")
        
        # Get video dimensions
        original_width = None
        original_height = None
        try:
            container = av.open(str(video_file))
            if len(container.streams.video) > 0:
                stream = container.streams.video[0]
                original_width = stream.width
                original_height = stream.height
            container.close()
        except Exception as e:
            logger.debug(f"Could not get dimensions for {rel_path}: {e}")
        
        # Create metadata row
        metadata_row = {
            "video_path": rel_path,
            "label": label,
            "original_video": original_video,
            "augmentation_idx": aug_idx,
            "is_original": video_info["is_original"]
        }
        
        if original_width is not None and original_height is not None:
            metadata_row["original_width"] = original_width
            metadata_row["original_height"] = original_height
        
        metadata_rows.append(metadata_row)
    
    if not metadata_rows:
        logger.warning("⚠ No metadata rows generated")
        return False
    
    # Create DataFrame and save
    metadata_df = pl.DataFrame(metadata_rows)
    
    # Determine output path
    if output_path is None:
        output_path = scaled_videos_path / "scaled_metadata.arrow"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    
    logger.info(f"Saving reconstructed metadata: {len(metadata_rows)} entries to {output_path}")
    success = write_metadata_atomic(metadata_df, output_path, append=False)
    
    if success:
        logger.info(f"✓ Successfully reconstructed metadata: {len(metadata_rows)} entries saved to {output_path}")
        return True
    else:
        logger.error("✗ Failed to save reconstructed metadata")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reconstruct scaled metadata from scaled video files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--scaled-videos-dir",
        type=str,
        default="data/scaled_videos",
        help="Directory containing scaled videos (default: data/scaled_videos)"
    )
    parser.add_argument(
        "--augmented-metadata",
        type=str,
        default=None,
        help="Path to augmented metadata to get labels (optional, default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save reconstructed metadata (default: scaled_videos_dir/scaled_metadata.arrow)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect augmented metadata if not provided
    if args.augmented_metadata is None:
        project_root = Path(args.project_root)
        for candidate in [
            "data/augmented_videos/augmented_metadata.arrow",
            "data/augmented_videos/augmented_metadata.parquet",
            "data/augmented_videos/augmented_metadata.csv"
        ]:
            candidate_path = project_root / candidate
            if candidate_path.exists():
                args.augmented_metadata = candidate
                logger.info(f"Auto-detected augmented metadata: {candidate}")
                break
    
    success = reconstruct_scaled_metadata(
        project_root=args.project_root,
        scaled_videos_dir=args.scaled_videos_dir,
        augmented_metadata_path=args.augmented_metadata,
        output_path=args.output
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

