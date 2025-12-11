#!/usr/bin/env python3
"""
Stage 3 Completion Check Script

Verifies that all scaled videos in the scaled_videos directory are present in the metadata file.
Checks for:
- Missing videos (in directory but not in metadata)
- Extra entries (in metadata but file doesn't exist)
- Completeness statistics

Usage:
    python src/scripts/check_stage3_completion.py
    python src/scripts/check_stage3_completion.py --scaled-videos-dir data/scaled_videos
    python src/scripts/check_stage3_completion.py --metadata data/scaled_videos/scaled_metadata.arrow
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

from lib.utils.paths import load_metadata_flexible

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


def check_stage3_completion(
    project_root: str,
    scaled_videos_dir: str = "data/scaled_videos",
    metadata_path: str = None,
    reconstruct: bool = False
) -> dict:
    """
    Check if all scaled videos in directory are present in metadata.
    
    Args:
        project_root: Project root directory
        scaled_videos_dir: Directory containing scaled videos (default: data/scaled_videos)
        metadata_path: Path to scaled metadata file (default: auto-detect)
        reconstruct: If True, reconstruct metadata from directory files
    
    Returns:
        Dictionary with check results and statistics
    """
    project_root = Path(project_root)
    scaled_videos_path = project_root / scaled_videos_dir
    
    logger.info("=" * 80)
    logger.info("STAGE 3 COMPLETION CHECK")
    logger.info("=" * 80)
    logger.info(f"Scanning scaled videos directory: {scaled_videos_path}")
    
    if not scaled_videos_path.exists():
        logger.error(f"✗ Scaled videos directory not found: {scaled_videos_path}")
        return {
            "success": False,
            "error": f"Directory not found: {scaled_videos_path}"
        }
    
    # Find all scaled video files
    scaled_files = list(scaled_videos_path.glob("*_scaled*.mp4"))
    logger.info(f"Found {len(scaled_files)} scaled video files in directory")
    
    if len(scaled_files) == 0:
        logger.warning("⚠ No scaled video files found in directory")
        return {
            "success": True,
            "total_files": 0,
            "total_metadata_entries": 0,
            "missing_in_metadata": [],
            "extra_in_metadata": [],
            "completion_percentage": 0.0
        }
    
    # Extract video info from filenames
    files_by_path = {}
    files_by_video_id = defaultdict(list)
    
    for video_file in scaled_files:
        video_info = extract_video_info_from_filename(video_file.name)
        if video_info:
            # Store relative path from project root
            rel_path = str(video_file.relative_to(project_root))
            files_by_path[rel_path] = video_info
            files_by_video_id[video_info["video_id"]].append(rel_path)
    
    logger.info(f"Extracted info from {len(files_by_path)} video files")
    
    # Auto-detect metadata path if not provided
    if metadata_path is None:
        metadata_base = scaled_videos_path / "scaled_metadata"
        metadata_path = str(metadata_base)
        logger.info(f"Auto-detecting metadata file: {metadata_path}")
    
    # Load metadata
    logger.info(f"\nLoading metadata: {metadata_path}")
    metadata_df = load_metadata_flexible(str(project_root / metadata_path) if not Path(metadata_path).is_absolute() else metadata_path)
    
    if metadata_df is None or metadata_df.is_empty():
        logger.warning("⚠ Metadata file is empty or not found")
        if reconstruct:
            logger.info("Reconstructing metadata from directory files...")
            return _reconstruct_metadata(project_root, scaled_videos_path, files_by_path)
        else:
            return {
                "success": False,
                "total_files": len(files_by_path),
                "total_metadata_entries": 0,
                "missing_in_metadata": list(files_by_path.keys()),
                "extra_in_metadata": [],
                "completion_percentage": 0.0,
                "error": "Metadata file is empty or not found. Use --reconstruct to rebuild it."
            }
    
    logger.info(f"Metadata contains {metadata_df.height} entries")
    
    # Check for required columns
    if "video_path" not in metadata_df.columns:
        logger.error("✗ Metadata missing required column: video_path")
        return {
            "success": False,
            "error": "Metadata missing 'video_path' column"
        }
    
    # Get all video paths from metadata
    metadata_paths = set(metadata_df["video_path"].to_list())
    
    # Find missing videos (in directory but not in metadata)
    missing_in_metadata = []
    for file_path in files_by_path.keys():
        # Try exact match first
        if file_path not in metadata_paths:
            # Try with/without data/scaled_videos prefix
            alt_path1 = file_path.replace("data/scaled_videos/", "")
            alt_path2 = f"data/scaled_videos/{file_path}" if not file_path.startswith("data/scaled_videos/") else file_path
            
            if alt_path1 not in metadata_paths and alt_path2 not in metadata_paths:
                missing_in_metadata.append(file_path)
    
    # Find extra entries (in metadata but file doesn't exist)
    extra_in_metadata = []
    for meta_path in metadata_paths:
        # Try to find corresponding file
        meta_path_obj = Path(meta_path)
        if meta_path_obj.is_absolute():
            # Absolute path
            if not meta_path_obj.exists():
                extra_in_metadata.append(meta_path)
        else:
            # Relative path - try multiple locations
            candidates = [
                project_root / meta_path,
                project_root / "data" / "scaled_videos" / meta_path_obj.name,
                scaled_videos_path / meta_path_obj.name
            ]
            if not any(c.exists() for c in candidates):
                extra_in_metadata.append(meta_path)
    
    # Calculate statistics
    total_files = len(files_by_path)
    total_metadata = len(metadata_paths)
    matched_files = total_files - len(missing_in_metadata)
    completion_percentage = (matched_files / total_files * 100) if total_files > 0 else 0.0
    
    # Report results
    logger.info("=" * 80)
    logger.info("STAGE 3 COMPLETION CHECK RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total scaled video files in directory: {total_files}")
    logger.info(f"Total entries in metadata: {total_metadata}")
    logger.info(f"Matched files: {matched_files}")
    logger.info(f"Completion percentage: {completion_percentage:.2f}%")
    
    if missing_in_metadata:
        logger.warning(f"\n⚠ MISSING IN METADATA: {len(missing_in_metadata)} videos")
        logger.warning("These videos exist in the directory but are not in metadata:")
        for i, missing in enumerate(missing_in_metadata[:20], 1):  # Show first 20
            logger.warning(f"  {i}. {missing}")
        if len(missing_in_metadata) > 20:
            logger.warning(f"  ... and {len(missing_in_metadata) - 20} more")
    
    if extra_in_metadata:
        logger.warning(f"\n⚠ EXTRA IN METADATA: {len(extra_in_metadata)} entries")
        logger.warning("These entries are in metadata but files don't exist:")
        for i, extra in enumerate(extra_in_metadata[:20], 1):  # Show first 20
            logger.warning(f"  {i}. {extra}")
        if len(extra_in_metadata) > 20:
            logger.warning(f"  ... and {len(extra_in_metadata) - 20} more")
    
    if not missing_in_metadata and not extra_in_metadata:
        logger.info("\n✓ SUCCESS: All scaled videos are present in metadata!")
        logger.info("✓ Metadata is complete and consistent with directory contents")
    else:
        logger.warning("\n⚠ WARNING: Metadata is incomplete or inconsistent")
        if reconstruct:
            logger.info("Reconstructing metadata from directory files...")
            return _reconstruct_metadata(project_root, scaled_videos_path, files_by_path)
    
    return {
        "success": len(missing_in_metadata) == 0 and len(extra_in_metadata) == 0,
        "total_files": total_files,
        "total_metadata_entries": total_metadata,
        "matched_files": matched_files,
        "missing_in_metadata": missing_in_metadata,
        "extra_in_metadata": extra_in_metadata,
        "completion_percentage": completion_percentage
    }


def _reconstruct_metadata(
    project_root: Path,
    scaled_videos_path: Path,
    files_by_path: dict
) -> dict:
    """
    Reconstruct metadata from directory files.
    
    Args:
        project_root: Project root directory
        scaled_videos_path: Path to scaled videos directory
        files_by_path: Dictionary mapping file paths to video info
    
    Returns:
        Dictionary with reconstruction results
    """
    logger.info("Reconstructing metadata from scaled video files...")
    
    import polars as pl
    import av
    from lib.utils.paths import write_metadata_atomic
    
    metadata_rows = []
    
    for file_path, video_info in files_by_path.items():
        video_file = project_root / file_path
        if not video_file.exists():
            continue
        
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
            logger.debug(f"Could not get dimensions for {file_path}: {e}")
        
        # Create metadata row
        metadata_row = {
            "video_path": file_path,
            "label": "unknown",  # Can't determine from filename
            "original_video": "unknown",  # Can't determine from filename
            "augmentation_idx": video_info["aug_idx"],
            "is_original": video_info["is_original"]
        }
        
        if original_width is not None and original_height is not None:
            metadata_row["original_width"] = original_width
            metadata_row["original_height"] = original_height
        
        metadata_rows.append(metadata_row)
    
    if metadata_rows:
        metadata_df = pl.DataFrame(metadata_rows)
        metadata_path = scaled_videos_path / "scaled_metadata.arrow"
        
        success = write_metadata_atomic(metadata_df, metadata_path, append=False)
        
        if success:
            logger.info(f"✓ Reconstructed metadata: {len(metadata_rows)} entries saved to {metadata_path}")
            return {
                "success": True,
                "reconstructed": True,
                "total_files": len(files_by_path),
                "total_metadata_entries": len(metadata_rows),
                "missing_in_metadata": [],
                "extra_in_metadata": [],
                "completion_percentage": 100.0
            }
        else:
            logger.error("✗ Failed to save reconstructed metadata")
            return {
                "success": False,
                "error": "Failed to save reconstructed metadata"
            }
    else:
        logger.warning("⚠ No metadata rows to save")
        return {
            "success": False,
            "error": "No metadata rows generated"
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check Stage 3 completion: verify all scaled videos are in metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect paths
  python src/scripts/check_stage3_completion.py
  
  # Custom paths
  python src/scripts/check_stage3_completion.py --scaled-videos-dir data/scaled_videos --metadata data/scaled_videos/scaled_metadata.arrow
  
  # Reconstruct metadata if incomplete
  python src/scripts/check_stage3_completion.py --reconstruct
        """
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
        "--metadata",
        type=str,
        default=None,
        help="Path to scaled metadata file (default: auto-detect in scaled_videos_dir)"
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Reconstruct metadata from directory files if incomplete"
    )
    
    args = parser.parse_args()
    
    result = check_stage3_completion(
        project_root=args.project_root,
        scaled_videos_dir=args.scaled_videos_dir,
        metadata_path=args.metadata,
        reconstruct=args.reconstruct
    )
    
    if result.get("success", False):
        logger.info("\n✓ Stage 3 completion check PASSED")
        return 0
    else:
        logger.error("\n✗ Stage 3 completion check FAILED")
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

