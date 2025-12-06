#!/usr/bin/env python3
"""
Stage 1 Completion Check Script

Verifies that all videos from input metadata were processed and augmented correctly.
Checks for:
- Missing videos (in input but not in output)
- Incomplete augmentations (videos with fewer than expected augmentations)
- Extra videos (in output but not in input - should not happen)

Usage:
    python src/scripts/check_stage1_completion.py
    python src/scripts/check_stage1_completion.py --input-metadata data/video_index_input.csv
    python src/scripts/check_stage1_completion.py --output-metadata data/augmented_videos/augmented_metadata.arrow
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

from lib.data import load_metadata, filter_existing_videos
from lib.utils.paths import load_metadata_flexible

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_video_id(video_path: str) -> str:
    """
    Extract a unique video ID from a video path.
    
    For paths like "FVC1/youtube/07kv4UTfqgE/video.mp4", returns "07kv4UTfqgE"
    """
    path_obj = Path(video_path)
    if len(path_obj.parts) >= 2:
        # Parent directory is usually the unique ID
        video_id = path_obj.parts[-2]
        # Sanitize to match how Stage 1 processes it
        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
        return video_id
    # Fallback: use filename without extension
    return path_obj.stem


def check_stage1_completion(
    project_root: str,
    input_metadata_path: str,
    output_metadata_path: str,
    expected_augmentations: int = 10
) -> dict:
    """
    Check if all videos from Stage 1 input were processed and augmented correctly.
    
    Args:
        project_root: Project root directory
        input_metadata_path: Path to input metadata (video_index_input.csv or FVC_dup.csv)
        output_metadata_path: Path to Stage 1 output metadata (augmented_metadata.arrow/parquet/csv)
        expected_augmentations: Expected number of augmentations per video (default: 10)
    
    Returns:
        Dictionary with check results and statistics
    """
    project_root = Path(project_root)
    
    # Load input metadata
    logger.info("=" * 80)
    logger.info("STAGE 1 COMPLETION CHECK")
    logger.info("=" * 80)
    logger.info(f"Loading input metadata: {input_metadata_path}")
    
    input_df = load_metadata(str(project_root / input_metadata_path))
    logger.info(f"Input metadata: {input_df.height} videos")
    
    # Filter to only existing videos (same as Stage 1 does)
    logger.info("Filtering to existing videos (same as Stage 1)...")
    input_df_filtered = filter_existing_videos(input_df, str(project_root))
    logger.info(f"After filtering: {input_df_filtered.height} videos exist on disk")
    
    if input_df.height != input_df_filtered.height:
        missing_count = input_df.height - input_df_filtered.height
        logger.warning(f"⚠ {missing_count} videos from input don't exist on disk (will be skipped)")
    
    # Load output metadata
    logger.info(f"\nLoading Stage 1 output metadata: {output_metadata_path}")
    output_df = load_metadata_flexible(str(project_root / output_metadata_path))
    
    if output_df is None or output_df.is_empty():
        logger.error(f"✗ Stage 1 output metadata not found or empty: {output_metadata_path}")
        return {
            "status": "error",
            "input_count": input_df_filtered.height,
            "output_count": 0,
            "missing_videos": [],
            "incomplete_videos": [],
            "extra_videos": []
        }
    
    logger.info(f"Stage 1 output: {output_df.height} total entries")
    
    # Extract video IDs from input
    input_video_ids = set()
    input_video_to_path = {}
    for row in input_df_filtered.iter_rows(named=True):
        video_path = row["video_path"]
        video_id = extract_video_id(video_path)
        input_video_ids.add(video_id)
        input_video_to_path[video_id] = video_path
    
    logger.info(f"Input videos (unique IDs): {len(input_video_ids)}")
    
    # Analyze output metadata
    if "original_video" not in output_df.columns:
        logger.error("✗ Stage 1 output missing 'original_video' column")
        return {
            "status": "error",
            "input_count": input_df_filtered.height,
            "output_count": output_df.height,
            "missing_videos": [],
            "incomplete_videos": [],
            "extra_videos": []
        }
    
    # Count augmentations per original video
    video_augmentation_counts = defaultdict(int)
    video_original_counts = defaultdict(int)
    output_video_ids = set()
    
    for row in output_df.iter_rows(named=True):
        original_video = row.get("original_video", "")
        is_original = row.get("is_original", False)
        aug_idx = row.get("augmentation_idx", -1)
        
        if original_video:
            video_id = extract_video_id(original_video)
            output_video_ids.add(video_id)
            
            if is_original or aug_idx == -1:
                video_original_counts[video_id] += 1
            else:
                video_augmentation_counts[video_id] += 1
    
    logger.info(f"Output videos (unique IDs): {len(output_video_ids)}")
    
    # Check for missing videos
    missing_video_ids = input_video_ids - output_video_ids
    missing_videos = [
        {
            "video_id": vid,
            "video_path": input_video_to_path.get(vid, "unknown")
        }
        for vid in missing_video_ids
    ]
    
    # Check for incomplete augmentations
    incomplete_videos = []
    for video_id in output_video_ids:
        aug_count = video_augmentation_counts.get(video_id, 0)
        orig_count = video_original_counts.get(video_id, 0)
        total_expected = expected_augmentations + 1  # 10 augs + 1 original
        
        if orig_count == 0:
            logger.warning(f"⚠ Video {video_id} has no original entry (only augmentations)")
        
        if aug_count < expected_augmentations:
            incomplete_videos.append({
                "video_id": video_id,
                "video_path": input_video_to_path.get(video_id, "unknown"),
                "expected_augmentations": expected_augmentations,
                "actual_augmentations": aug_count,
                "has_original": orig_count > 0,
                "missing_count": expected_augmentations - aug_count
            })
    
    # Check for extra videos (in output but not in input)
    extra_video_ids = output_video_ids - input_video_ids
    extra_videos = list(extra_video_ids)
    
    # Calculate statistics
    total_expected = input_df_filtered.height * (expected_augmentations + 1)
    total_actual = output_df.height
    
    # Report results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"Input videos (after filtering): {input_df_filtered.height}")
    logger.info(f"Output entries: {output_df.height}")
    logger.info(f"Expected entries: {total_expected} ({input_df_filtered.height} videos × {expected_augmentations + 1})")
    logger.info(f"Difference: {total_actual - total_expected}")
    
    logger.info(f"\nOriginal videos in output: {sum(video_original_counts.values())}")
    logger.info(f"Augmented videos in output: {sum(video_augmentation_counts.values())}")
    
    if missing_videos:
        logger.warning(f"\n⚠ MISSING VIDEOS: {len(missing_videos)} videos from input are not in output")
        for missing in missing_videos[:10]:  # Show first 10
            logger.warning(f"  - {missing['video_id']}: {missing['video_path']}")
        if len(missing_videos) > 10:
            logger.warning(f"  ... and {len(missing_videos) - 10} more")
    else:
        logger.info(f"\n✓ All {len(input_video_ids)} input videos are present in output")
    
    if incomplete_videos:
        logger.warning(f"\n⚠ INCOMPLETE AUGMENTATIONS: {len(incomplete_videos)} videos have fewer than {expected_augmentations} augmentations")
        for incomplete in incomplete_videos[:10]:  # Show first 10
            logger.warning(
                f"  - {incomplete['video_id']}: {incomplete['actual_augmentations']}/{incomplete['expected_augmentations']} "
                f"augmentations (missing {incomplete['missing_count']})"
            )
        if len(incomplete_videos) > 10:
            logger.warning(f"  ... and {len(incomplete_videos) - 10} more")
    else:
        logger.info(f"\n✓ All videos have complete augmentations ({expected_augmentations} augs + 1 original)")
    
    if extra_videos:
        logger.warning(f"\n⚠ EXTRA VIDEOS: {len(extra_videos)} videos in output but not in input")
        for extra in extra_videos[:10]:  # Show first 10
            logger.warning(f"  - {extra}")
        if len(extra_videos) > 10:
            logger.warning(f"  ... and {len(extra_videos) - 10} more")
    else:
        logger.info(f"\n✓ No extra videos in output")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    status = "complete"
    if missing_videos:
        status = "incomplete"
        logger.error(f"✗ Stage 1 is INCOMPLETE: {len(missing_videos)} videos missing")
    elif incomplete_videos:
        status = "incomplete"
        logger.error(f"✗ Stage 1 is INCOMPLETE: {len(incomplete_videos)} videos have incomplete augmentations")
    else:
        logger.info(f"✓ Stage 1 is COMPLETE: All videos processed with full augmentations")
    
    if extra_videos:
        logger.warning(f"⚠ Note: {len(extra_videos)} extra videos found (may be from previous runs)")
    
    return {
        "status": status,
        "input_count": input_df_filtered.height,
        "output_count": output_df.height,
        "expected_count": total_expected,
        "missing_videos": missing_videos,
        "incomplete_videos": incomplete_videos,
        "extra_videos": extra_videos,
        "original_count": sum(video_original_counts.values()),
        "augmented_count": sum(video_augmentation_counts.values())
    }


def main():
    """Run Stage 1 completion check."""
    parser = argparse.ArgumentParser(
        description="Check if all videos from Stage 1 input were processed and augmented correctly",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--input-metadata",
        type=str,
        default=None,
        help="Path to input metadata (default: auto-detect video_index_input.csv or FVC_dup.csv)"
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default=None,
        help="Path to Stage 1 output metadata (default: auto-detect augmented_metadata.arrow/parquet/csv)"
    )
    parser.add_argument(
        "--expected-augmentations",
        type=int,
        default=10,
        help="Expected number of augmentations per video (default: 10)"
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    
    # Auto-detect input metadata
    if args.input_metadata:
        input_metadata_path = args.input_metadata
    else:
        # Try FVC_dup.csv first, then video_index_input.csv
        for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
            candidate = project_root / "data" / csv_name
            if candidate.exists():
                input_metadata_path = f"data/{csv_name}"
                logger.info(f"Auto-detected input metadata: {input_metadata_path}")
                break
        else:
            logger.error("Could not find input metadata. Please specify --input-metadata")
            return 1
    
    # Auto-detect output metadata
    if args.output_metadata:
        output_metadata_path = args.output_metadata
    else:
        # Try .arrow, .parquet, .csv in order
        for ext in [".arrow", ".parquet", ".csv"]:
            candidate = project_root / "data" / "augmented_videos" / f"augmented_metadata{ext}"
            if candidate.exists():
                output_metadata_path = f"data/augmented_videos/augmented_metadata{ext}"
                logger.info(f"Auto-detected output metadata: {output_metadata_path}")
                break
        else:
            logger.error("Could not find Stage 1 output metadata. Please specify --output-metadata")
            return 1
    
    # Run check
    try:
        results = check_stage1_completion(
            project_root=str(project_root),
            input_metadata_path=input_metadata_path,
            output_metadata_path=output_metadata_path,
            expected_augmentations=args.expected_augmentations
        )
        
        # Return exit code based on status
        if results["status"] == "complete":
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Error during check: {e}", exc_info=True)
        return 1


def calculate_processing_time_estimate(
    project_root: str,
    input_metadata_path: str = "data/video_index_input.csv",
    num_augmentations: int = 10,
    processing_rate_seconds_per_100_frames: float = 3.5
) -> dict:
    """
    Calculate processing time estimate for Stage 3 scaling.
    
    Loads original video metadata, counts frames for each original video,
    multiplies by (1 + num_augmentations) since augmented videos have same frame count,
    and calculates the estimated processing time based on a rate of seconds per 100 frames.
    
    Args:
        project_root: Project root directory
        input_metadata_path: Path to original video metadata (default: data/video_index_input.csv)
        num_augmentations: Number of augmentations per original video (default: 10)
        processing_rate_seconds_per_100_frames: Processing rate in seconds per 100 frames (default: 3.5)
    
    Returns:
        Dictionary with frame counts and time estimates
    """
    project_root = Path(project_root)
    
    logger.info("=" * 80)
    logger.info("STAGE 3 PROCESSING TIME ESTIMATE")
    logger.info("=" * 80)
    logger.info(f"Loading original video metadata: {input_metadata_path}")
    
    # Load original video metadata
    df = load_metadata(str(project_root / input_metadata_path))
    if df is None or df.is_empty():
        logger.error(f"Original video metadata not found or empty: {input_metadata_path}")
        return {
            "status": "error",
            "total_videos": 0,
            "total_frames": 0,
            "estimated_time_seconds": 0,
            "estimated_time_minutes": 0,
            "estimated_time_hours": 0
        }
    
    # Filter to existing videos (same as Stage 1 does)
    df = filter_existing_videos(df, str(project_root))
    
    num_original_videos = df.height
    logger.info(f"Found {num_original_videos} original videos")
    logger.info(f"Number of augmentations per video: {num_augmentations}")
    logger.info(f"Total videos to process: {num_original_videos} originals × {num_augmentations + 1} = {num_original_videos * (num_augmentations + 1)}")
    logger.info(f"Note: Augmented videos have same frame count as their originals")
    
    # Get frame counts directly from CSV (much faster!)
    logger.info("\nReading frame counts from CSV...")
    
    if "frame_count" not in df.columns:
        logger.error("CSV does not have 'frame_count' column")
        return {
            "status": "error",
            "total_videos": 0,
            "total_frames": 0,
            "estimated_time_seconds": 0,
            "estimated_time_minutes": 0,
            "estimated_time_hours": 0
        }
    
    # Sum frame counts from CSV
    total_original_frames = df["frame_count"].sum()
    processed_count = df.height
    
    # Calculate total frames: sum(original_frames * (1 + num_augmentations))
    # Each original video produces (1 original + num_augmentations) videos, all with same frame count
    total_frames = total_original_frames * (1 + num_augmentations)
    
    logger.info(f"\nFrame counting complete:")
    logger.info(f"  - Original videos: {processed_count}")
    logger.info(f"  - Total frames in original videos: {total_original_frames:,}")
    logger.info(f"  - Total frames to process (originals + {num_augmentations} augmentations each): {total_frames:,}")
    
    # Calculate time estimates
    # Rate: processing_rate_seconds_per_100_frames seconds per 100 frames
    # So: time = (total_frames / 100) * processing_rate_seconds_per_100_frames
    estimated_time_seconds = (total_frames / 100.0) * processing_rate_seconds_per_100_frames
    estimated_time_minutes = estimated_time_seconds / 60.0
    estimated_time_hours = estimated_time_minutes / 60.0
    
    logger.info("\n" + "=" * 80)
    logger.info("TIME ESTIMATES")
    logger.info("=" * 80)
    logger.info(f"Processing rate: {processing_rate_seconds_per_100_frames} seconds per 100 frames")
    logger.info(f"Total frames to process: {total_frames:,}")
    logger.info(f"\nEstimated processing time:")
    logger.info(f"  - {estimated_time_seconds:,.2f} seconds")
    logger.info(f"  - {estimated_time_minutes:,.2f} minutes")
    logger.info(f"  - {estimated_time_hours:,.2f} hours")
    
    # Calculate per-video average
    if processed_count > 0:
        avg_frames_per_original = total_original_frames / processed_count
        total_videos_to_process = processed_count * (1 + num_augmentations)
        avg_frames_per_video = total_frames / total_videos_to_process if total_videos_to_process > 0 else 0
        avg_time_per_video = (avg_frames_per_video / 100.0) * processing_rate_seconds_per_100_frames
        logger.info(f"\nAverage statistics:")
        logger.info(f"  - Frames per original video: {avg_frames_per_original:,.1f}")
        logger.info(f"  - Frames per video (same for original and augmentations): {avg_frames_per_original:,.1f}")
        logger.info(f"  - Time per video: {avg_time_per_video:.2f} seconds")
    
    logger.info("=" * 80)
    
    return {
        "status": "success",
        "original_videos": num_original_videos,
        "processed_originals": processed_count,
        "num_augmentations": num_augmentations,
        "total_videos_to_process": processed_count * (1 + num_augmentations),
        "total_original_frames": total_original_frames,
        "total_frames": total_frames,
        "estimated_time_seconds": estimated_time_seconds,
        "estimated_time_minutes": estimated_time_minutes,
        "estimated_time_hours": estimated_time_hours,
        "processing_rate_seconds_per_100_frames": processing_rate_seconds_per_100_frames
    }


if __name__ == "__main__":
    # Check if --estimate-time flag is provided
    if "--estimate-time" in sys.argv or "-e" in sys.argv:
        # Run time estimation
        parser = argparse.ArgumentParser(
            description="Calculate Stage 3 processing time estimate",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument(
            "--estimate-time",
            "-e",
            action="store_true",
            help="Calculate processing time estimate for Stage 3"
        )
        parser.add_argument(
            "--project-root",
            type=str,
            default=str(Path.cwd()),
            help="Project root directory (default: current working directory)"
        )
        parser.add_argument(
            "--input-metadata",
            type=str,
            default=None,
            help="Path to original video metadata (default: auto-detect data/video_index_input.csv or FVC_dup.csv)"
        )
        parser.add_argument(
            "--num-augmentations",
            type=int,
            default=10,
            help="Number of augmentations per original video (default: 10)"
        )
        parser.add_argument(
            "--processing-rate",
            type=float,
            default=3.5,
            help="Processing rate in seconds per 100 frames (default: 3.5)"
        )
        
        args = parser.parse_args()
        
        project_root = Path(args.project_root).resolve()
        
        # Auto-detect input metadata if not provided
        if args.input_metadata:
            input_metadata_path = args.input_metadata
        else:
            # Try FVC_dup.csv first, then video_index_input.csv
            for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
                candidate = project_root / "data" / csv_name
                if candidate.exists():
                    input_metadata_path = f"data/{csv_name}"
                    logger.info(f"Auto-detected input metadata: {input_metadata_path}")
                    break
            else:
                logger.error("Could not find input metadata. Please specify --input-metadata")
                sys.exit(1)
        
        try:
            results = calculate_processing_time_estimate(
                project_root=str(project_root),
                input_metadata_path=input_metadata_path,
                num_augmentations=args.num_augmentations,
                processing_rate_seconds_per_100_frames=args.processing_rate
            )
            
            if results["status"] == "success":
                sys.exit(0)
            else:
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error during time estimation: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Run normal completion check
        sys.exit(main())

