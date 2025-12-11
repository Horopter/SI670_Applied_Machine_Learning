"""
Video augmentation pipeline.

Generates multiple augmented versions of each video using spatial and temporal
transformations. Creates augmented clips for training data augmentation.
"""

from __future__ import annotations

import logging
import hashlib
import random
import csv
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np
import polars as pl
import av

from lib.data import load_metadata, filter_existing_videos
from lib.utils.paths import resolve_video_path, write_metadata_atomic, get_video_metadata_cache_path
from lib.utils.memory import aggressive_gc, log_memory_stats
from .io import load_frames, save_frames, concatenate_videos
from .transforms import apply_simple_augmentation

logger = logging.getLogger(__name__)


def augment_video(
    video_path: str,
    num_augmentations: int = 10,
    augmentation_types: Optional[List[str]] = None,
    max_frames: Optional[int] = 250,
    chunk_size: int = 250,
    checkpoint_dir: Optional[Path] = None,
    resume: bool = True,
    output_dir: Optional[Path] = None,
    video_id: Optional[str] = None,
) -> List[str]:
    """
    Generate augmented versions of a video, processing in chunks to handle long videos.
    
    Processes videos in chunks of 1000 frames, saves intermediate checkpoints, and stitches
    them together. Supports resuming from checkpoints if processing is interrupted.
    
    Args:
        video_path: Path to video file
        num_augmentations: Number of augmentations to generate
        augmentation_types: List of augmentation types to use
        max_frames: Maximum frames to load per chunk (default: 1000)
        chunk_size: Number of frames to process per chunk (default: 1000)
        checkpoint_dir: Directory to save intermediate chunk files (None = use temp dir)
        resume: If True and checkpoint_dir is set, resume from existing checkpoints
    
    Returns:
        List of final augmented video paths (one per augmentation)
    """
    logger.info(f"Loading video: {video_path}")
    
    if max_frames is None:
        max_frames = 250
    if chunk_size is None:
        chunk_size = 250
    
    # Use cached metadata to avoid duplicate frame counting
    from lib.utils.video_cache import get_video_metadata
    
    # Use persistent cache file for cross-stage caching
    cache_file = get_video_metadata_cache_path(output_dir.parent if output_dir else None)
    metadata = get_video_metadata(video_path, use_cache=True, cache_file=cache_file)
    total_frames = metadata['total_frames']
    fps = metadata['fps']
    
    if total_frames == 0:
        logger.error(f"Video has no frames: {video_path}")
        return []
    
    logger.info(f"Video has {total_frames} frames (fps: {fps:.2f}), processing in chunks of {chunk_size}")
    
    # Generate deterministic seed from video path
    video_path_str = str(video_path)
    base_seed = int(hashlib.md5(video_path_str.encode()).hexdigest()[:8], 16) % (2**31)
    
    # Default augmentation types
    if augmentation_types is None:
        augmentation_types = [
            'rotation', 'flip', 'brightness', 'contrast', 'saturation',
            'gaussian_noise', 'gaussian_blur', 'affine', 'elastic', 'cutout'
        ]
    
    # Ensure diversity: use each augmentation type at least once
    if num_augmentations <= len(augmentation_types):
        selected_types = augmentation_types[:num_augmentations].copy()
        random.seed(base_seed)
        random.shuffle(selected_types)
    else:
        selected_types = []
        for i in range(num_augmentations):
            selected_types.append(augmentation_types[i % len(augmentation_types)])
        random.seed(base_seed)
        random.shuffle(selected_types[:len(augmentation_types)])
    
    # Process video in chunks, saving intermediates and stitching at the end
    num_chunks = (total_frames + chunk_size - 1) // chunk_size  # Ceiling division
    logger.info(f"Processing {num_chunks} chunk(s)")
    
    # Create checkpoint directory for intermediate chunk files
    if checkpoint_dir is None:
        # Use temp directory if no checkpoint dir specified
        temp_dir = Path(tempfile.mkdtemp(prefix="aug_chunks_"))
        use_checkpoints = False
    else:
        # Use persistent checkpoint directory
        temp_dir = checkpoint_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        use_checkpoints = True
    
    logger.debug(f"Using {'checkpoint' if use_checkpoints else 'temporary'} directory for chunks: {temp_dir}")
    
    try:
        # Store intermediate file paths for each augmentation
        intermediate_files = [[] for _ in range(num_augmentations)]
        
        # Check for existing checkpoints if resuming
        if resume and use_checkpoints:
            logger.info("Checking for existing checkpoints...")
            for aug_idx in range(num_augmentations):
                for chunk_idx in range(num_chunks):
                    checkpoint_path = temp_dir / f"chunk_{chunk_idx}_aug_{aug_idx}.mp4"
                    if checkpoint_path.exists():
                        intermediate_files[aug_idx].append(str(checkpoint_path))
                        logger.debug(f"Found checkpoint: chunk {chunk_idx + 1} for augmentation {aug_idx + 1}")
            
            # Sort by chunk index to ensure correct order
            for aug_idx in range(num_augmentations):
                intermediate_files[aug_idx].sort(key=lambda x: int(Path(x).stem.split('_')[1]))
            
            # Determine which chunks are already complete
            completed_augmentations = set()
            for aug_idx in range(num_augmentations):
                if len(intermediate_files[aug_idx]) == num_chunks:
                    # All chunks complete for this augmentation
                    completed_augmentations.add(aug_idx)
            
            if completed_augmentations:
                logger.info(f"Found complete checkpoints for {len(completed_augmentations)} augmentation(s), will skip processing")
            elif any(len(files) > 0 for files in intermediate_files):
                max_completed_chunk = max(
                    (int(Path(f).stem.split('_')[1]) for files in intermediate_files for f in files if f),
                    default=-1
                )
                if max_completed_chunk >= 0:
                    logger.info(f"Resuming from chunk {max_completed_chunk + 2}/{num_chunks}")
        
        # Process chunks that haven't been checkpointed
        consecutive_empty_chunks = 0
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, total_frames)
            frames_in_chunk = end_frame - start_frame
            
            # Early break if we've passed the video length
            if start_frame >= total_frames:
                logger.info(f"Reached end of video at chunk {chunk_idx + 1} (start_frame={start_frame} >= total_frames={total_frames})")
                break
            
            # Check if this chunk is already complete for all augmentations
            chunk_complete = True
            for aug_idx in range(num_augmentations):
                checkpoint_path = temp_dir / f"chunk_{chunk_idx}_aug_{aug_idx}.mp4"
                if not checkpoint_path.exists():
                    chunk_complete = False
                    break
            
            if chunk_complete and resume:
                logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} already checkpointed, skipping...")
                # Ensure it's in intermediate_files (sorted by chunk index)
                for aug_idx in range(num_augmentations):
                    checkpoint_path = temp_dir / f"chunk_{chunk_idx}_aug_{aug_idx}.mp4"
                    checkpoint_str = str(checkpoint_path)
                    if checkpoint_str not in intermediate_files[aug_idx]:
                        # Insert in sorted order
                        intermediate_files[aug_idx].append(checkpoint_str)
                        intermediate_files[aug_idx].sort(key=lambda x: int(Path(x).stem.split('_')[1]))
                consecutive_empty_chunks = 0  # Reset counter
                continue
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_frame}-{end_frame-1})")
            
            # Load chunk
            chunk_frames, chunk_fps = load_frames(video_path, max_frames=chunk_size, start_frame=start_frame)
            
            if not chunk_frames:
                consecutive_empty_chunks += 1
                logger.warning(f"No frames loaded for chunk {chunk_idx + 1}, skipping")
                # If we get multiple consecutive empty chunks, we've likely reached the end
                if consecutive_empty_chunks >= 2:
                    logger.info(f"Got {consecutive_empty_chunks} consecutive empty chunks, reached end of video")
                    break
                continue
            
            consecutive_empty_chunks = 0  # Reset counter on successful load
            
            if len(chunk_frames) != frames_in_chunk:
                logger.warning(f"Expected {frames_in_chunk} frames, got {len(chunk_frames)}")
            
            # Augment each chunk and save to intermediate file
            for aug_idx in range(num_augmentations):
                aug_seed = base_seed + aug_idx + (chunk_idx * 1000)  # Different seed per chunk
                random.seed(aug_seed)
                np.random.seed(aug_seed)
                
                # Use pre-selected augmentation type
                aug_type = selected_types[aug_idx]
                
                # Apply augmentation to all frames in chunk
                augmented_chunk = []
                for frame_idx, frame in enumerate(chunk_frames):
                    augmented_frame = apply_simple_augmentation(frame, aug_type, aug_seed + frame_idx)
                    augmented_chunk.append(augmented_frame)
                    
                    # Aggressive GC every 100 frames
                    if (frame_idx + 1) % 100 == 0:
                        aggressive_gc(clear_cuda=False)
                
                # Save chunk to intermediate file (checkpoint)
                intermediate_path = temp_dir / f"chunk_{chunk_idx}_aug_{aug_idx}.mp4"
                if save_frames(augmented_chunk, str(intermediate_path), fps=chunk_fps):
                    # Add to list in sorted order
                    checkpoint_str = str(intermediate_path)
                    if checkpoint_str not in intermediate_files[aug_idx]:
                        intermediate_files[aug_idx].append(checkpoint_str)
                        intermediate_files[aug_idx].sort(key=lambda x: int(Path(x).stem.split('_')[1]))
                    logger.info(f"Chunk {chunk_idx + 1}: Checkpointed augmentation {aug_idx + 1}/{num_augmentations} with type '{aug_type}' ({len(augmented_chunk)} frames)")
                else:
                    logger.error(f"Failed to save intermediate chunk {chunk_idx + 1} for augmentation {aug_idx + 1}")
                
                del augmented_chunk
                aggressive_gc(clear_cuda=False)
            
            del chunk_frames
            aggressive_gc(clear_cuda=False)
        
        # Stitch all intermediate files together for each augmentation
        logger.info("Stitching intermediate chunks together...")
        augmented_paths: List[str] = []
        
        for aug_idx in range(num_augmentations):
            if not intermediate_files[aug_idx]:
                logger.warning(f"No intermediate files for augmentation {aug_idx + 1}, skipping")
                augmented_paths.append("")
                continue
            
            # Determine final output path for this augmentation
            if output_dir is not None and video_id is not None:
                final_path = output_dir / f"{video_id}_aug{aug_idx}.mp4"
            else:
                # Fallback: place stitched file in temp directory
                final_path = temp_dir / f"stitched_aug_{aug_idx}.mp4"
            
            if concatenate_videos(intermediate_files[aug_idx], str(final_path), fps=fps):
                logger.info(
                    f"Stitched augmentation {aug_idx + 1}/{num_augmentations} into {final_path} "
                    f"from {len(intermediate_files[aug_idx])} chunks"
                )
                augmented_paths.append(str(final_path))
            else:
                logger.error(f"Failed to stitch augmentation {aug_idx + 1}")
                augmented_paths.append("")
        
        # Clean up intermediate chunk files after successful stitching
        if use_checkpoints:
            logger.info("Cleaning up checkpoint files after successful stitching...")
            try:
                # Only delete if we successfully created all augmentations
                if all(path for path in augmented_paths):
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Deleted checkpoint directory: {temp_dir}")
                else:
                    logger.warning("Some augmentations failed, keeping checkpoints for resume")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint directory {temp_dir}: {e}")
        else:
            # Always clean up temp directory
            logger.info("Cleaning up temporary chunk files...")
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Deleted temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary directory {temp_dir}: {e}")
        
        logger.info(f"Completed augmentation. Generated files: {augmented_paths}")
        return augmented_paths
        
    except Exception as e:
        logger.error(f"Error during chunked processing: {e}", exc_info=True)
        # Only clean up temp directory on error, keep checkpoints for resume
        if not use_checkpoints:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception:
                pass
        else:
            logger.info("Checkpoints preserved for resume. Fix errors and re-run to continue.")
        return []


def _reconstruct_metadata_from_files(
    metadata_path: Path,
    output_dir: Path,
    project_root: Path,
    df: pl.DataFrame,
    num_augmentations: int
) -> None:
    """
    Reconstruct metadata CSV from existing augmentation files in the output directory.
    
    Scans for all *_aug*.mp4 and *_original.mp4 files and creates metadata entries.
    This function loads the full original metadata to match video_ids correctly.
    """
    logger.info("Reconstructing metadata from existing augmentation files...")
    
    # Load full original metadata (not just the filtered df) to match all video_ids
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            break
    
    if input_metadata_path is None:
        logger.error("Cannot reconstruct metadata: original metadata file not found")
        return
    
    try:
        full_df = load_metadata(str(input_metadata_path))
    except Exception as e:
        logger.error(f"Cannot reconstruct metadata: failed to load original metadata: {e}")
        return
    
    # Create a mapping from video_id to original video path and label
    # Check all videos in the full dataset, not just the filtered range
    video_id_to_info = {}
    for row in full_df.iter_rows(named=True):
        video_rel = row["video_path"]
        label = row["label"]
        try:
            video_path = resolve_video_path(video_rel, project_root)
            if Path(video_path).exists():
                video_path_obj = Path(video_path)
                video_path_parts = video_path_obj.parts
                if len(video_path_parts) >= 2:
                    video_id = video_path_parts[-2]
                    video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                    video_id_to_info[video_id] = {
                        'original_video': video_rel,
                        'label': label
                    }
        except Exception:
            continue
    
    # Scan for all augmentation and original files in the output directory
    entries = []
    
    # Find all *_aug*.mp4 files
    for aug_file in output_dir.glob("*_aug*.mp4"):
        aug_filename = aug_file.stem  # Remove .mp4 extension
        if "_aug" in aug_filename:
            video_id = aug_filename.split("_aug")[0]
            aug_idx_str = aug_filename.split("_aug")[1]
            try:
                aug_idx = int(aug_idx_str)
            except ValueError:
                continue
            
            if video_id in video_id_to_info:
                info = video_id_to_info[video_id]
                aug_path_rel = str(aug_file.relative_to(project_root))
                entries.append({
                    'video_path': aug_path_rel,
                    'label': info['label'],
                    'original_video': info['original_video'],
                    'augmentation_idx': aug_idx,
                    'is_original': False
                })
    
    # Find all *_original.mp4 files
    for orig_file in output_dir.glob("*_original.mp4"):
        orig_filename = orig_file.stem  # Remove .mp4 extension
        if orig_filename.endswith("_original"):
            video_id = orig_filename[:-9]  # Remove "_original" suffix
            
            if video_id in video_id_to_info:
                info = video_id_to_info[video_id]
                orig_path_rel = str(orig_file.relative_to(project_root))
                entries.append({
                    'video_path': orig_path_rel,
                    'label': info['label'],
                    'original_video': info['original_video'],
                    'augmentation_idx': -1,  # -1 for original videos
                    'is_original': True
                })
    
    # Load existing metadata if it exists (support Arrow/Parquet/CSV)
    existing_entries = set()
    existing_df = None
    if metadata_path.exists():
        try:
            metadata_path_obj = Path(metadata_path)
            if metadata_path_obj.suffix == '.arrow':
                existing_df = pl.read_ipc(metadata_path_obj)
            elif metadata_path_obj.suffix == '.parquet':
                existing_df = pl.read_parquet(metadata_path_obj)
            else:
                existing_df = pl.read_csv(str(metadata_path))
            for row in existing_df.iter_rows(named=True):
                existing_entries.add((
                    row.get('video_path', ''),
                    row.get('original_video', ''),
                    row.get('augmentation_idx', -999)
                ))
        except Exception:
            pass
    
    # Collect new entries
    new_entries = []
    for entry in entries:
        entry_key = (entry['video_path'], entry['original_video'], entry['augmentation_idx'])
        if entry_key not in existing_entries:
            new_entries.append(entry)
    
    # Append new entries to existing DataFrame or create new one
    if new_entries:
        new_df = pl.DataFrame(new_entries)
        if existing_df is not None and existing_df.height > 0:
            combined_df = pl.concat([existing_df, new_df])
        else:
            combined_df = new_df
        
        # Save atomically with file locking to prevent race conditions
        metadata_path_arrow = metadata_path.with_suffix('.arrow')
        success = write_metadata_atomic(combined_df, metadata_path_arrow)
        
        if not success:
            # Fallback to Parquet
            metadata_path_parquet = metadata_path.with_suffix('.parquet')
            success = write_metadata_atomic(combined_df, metadata_path_parquet)
            if success:
                logger.debug(f"Saved metadata as Parquet: {metadata_path_parquet}")
        else:
            logger.debug(f"Saved metadata as Arrow IPC: {metadata_path_arrow}")
            # Remove old CSV/Parquet if it exists
            if metadata_path.exists() and metadata_path.suffix != '.arrow':
                try:
                    metadata_path.unlink()
                except Exception:
                    pass
                existing_entries.add(entry_key)
    
    if new_entries:
        logger.info(f"✓ Reconstructed metadata: Added {len(new_entries)} new entries (total {len(entries)} found)")
    else:
        logger.info(f"✓ Metadata reconstruction: All {len(entries)} entries already exist in metadata")


def stage1_augment_videos(
    project_root: str,
    num_augmentations: int = 10,
    output_dir: str = "data/augmented_videos",
    delete_existing: bool = False,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None
) -> pl.DataFrame:
    """
    Stage 1: Augment all videos.
    
    Args:
        project_root: Project root directory
        num_augmentations: Number of augmentations per video (default: 10)
        output_dir: Directory to save augmented videos
        delete_existing: If True, delete existing augmentations before regenerating (default: False)
        start_idx: Start index for video range (0-based, inclusive). If None, starts from 0.
        end_idx: End index for video range (0-based, exclusive). If None, processes all videos.
    
    Returns:
        DataFrame with metadata for all videos (original + augmented)
    """
    import numpy as np
    
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata - check for FVC_dup.csv first, then video_index_input.csv
    logger.info("Stage 1: Loading video metadata...")
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            logger.info(f"Using metadata file: {input_metadata_path}")
            break
    
    if input_metadata_path is None:
        logger.error(f"Metadata file not found. Expected: {project_root / 'data' / 'FVC_dup.csv'} or {project_root / 'data' / 'video_index_input.csv'}")
        return pl.DataFrame()
    
    df = load_metadata(str(input_metadata_path))
    df = filter_existing_videos(df, str(project_root))
    
    total_videos = df.height
    
    # Apply range filtering if specified
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else total_videos
        if start < 0:
            start = 0
        if end > total_videos:
            end = total_videos
        if start >= end:
            logger.warning(f"Invalid range: start_idx={start}, end_idx={end}, total_videos={total_videos}. Skipping.")
            return pl.DataFrame()
        df = df.slice(start, end - start)
        logger.info(f"Stage 1: Processing video range [{start}, {end}) of {total_videos} total videos")
    else:
        logger.info(f"Stage 1: Processing all {total_videos} videos")
    
    logger.info(f"Stage 1: Found {df.height} videos to process")
    logger.info(f"Stage 1: Generating {num_augmentations} augmentation(s) per video")
    logger.info(f"Stage 1: Output directory: {output_dir}")
    logger.info(f"Stage 1: Delete existing augmentations: {delete_existing}")
    
    # Use Arrow IPC for metadata (faster and type-safe)
    # Check for existing CSV file and migrate it to Arrow format
    metadata_path_arrow = output_dir / "augmented_metadata.arrow"
    metadata_path_csv = output_dir / "augmented_metadata.csv"
    metadata_path_parquet = output_dir / "augmented_metadata.parquet"
    
    # Migrate CSV to Arrow if CSV exists and Arrow doesn't
    if metadata_path_csv.exists() and not metadata_path_arrow.exists() and not metadata_path_parquet.exists():
        logger.info("Stage 1: Found existing CSV metadata, migrating to Arrow format...")
        try:
            csv_df = pl.read_csv(str(metadata_path_csv))
            if write_metadata_atomic(csv_df, metadata_path_arrow):
                logger.info(f"✓ Migrated {csv_df.height} entries from CSV to Arrow format")
                # Keep CSV as backup for now (can delete later if needed)
                logger.debug(f"CSV file kept as backup: {metadata_path_csv}")
            else:
                logger.warning("Failed to migrate CSV to Arrow atomically")
        except Exception as e:
            logger.warning(f"Failed to migrate CSV to Arrow: {e}, will use CSV format")
    
    # Determine which metadata file to use (prefer Arrow, then Parquet, then CSV)
    metadata_path = None
    if metadata_path_arrow.exists():
        metadata_path = metadata_path_arrow
    elif metadata_path_parquet.exists():
        metadata_path = metadata_path_parquet
    elif metadata_path_csv.exists():
        metadata_path = metadata_path_csv
    
    # Load existing metadata if it exists and we're not deleting (support CSV/Arrow/Parquet)
    existing_metadata = None
    existing_video_ids_with_all_augs = set()  # Videos that have all augmentations
    if metadata_path and metadata_path.exists() and not delete_existing:
        try:
            metadata_path_obj = Path(metadata_path)
            if metadata_path_obj.suffix == '.arrow':
                existing_metadata = pl.read_ipc(metadata_path_obj)
            elif metadata_path_obj.suffix == '.parquet':
                existing_metadata = pl.read_parquet(metadata_path_obj)
            else:
                existing_metadata = pl.read_csv(str(metadata_path))
            # Count augmentations per video to find which have all augmentations
            video_aug_counts = {}
            for row in existing_metadata.iter_rows(named=True):
                original_video = row.get("original_video", "")
                aug_idx = row.get("augmentation_idx", -1)
                if aug_idx >= 0:  # This is an augmentation
                    # Extract video_id from original_video path
                    video_path_obj = Path(original_video)
                    if len(video_path_obj.parts) >= 2:
                        video_id = video_path_obj.parts[-2]
                        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                        video_aug_counts[video_id] = video_aug_counts.get(video_id, 0) + 1
            
            # Only mark videos that have all required augmentations
            for video_id, count in video_aug_counts.items():
                if count >= num_augmentations:
                    existing_video_ids_with_all_augs.add(video_id)
            
            logger.info(f"Stage 1: Found {len(existing_video_ids_with_all_augs)} videos with all {num_augmentations} augmentations")
        except Exception as e:
            logger.warning(f"Could not load existing metadata: {e}, will regenerate")
            existing_metadata = None
    
    # If deleting existing, remove augmented files only in the specified range
    if delete_existing:
        logger.info("Stage 1: Deleting existing augmentations in range...")
        
        # Get the video IDs that will be processed in this range
        video_ids_in_range = set()
        for idx in range(df.height):
            row = df.row(idx, named=True)
            video_rel = row["video_path"]
            try:
                video_path = resolve_video_path(video_rel, project_root)
                if Path(video_path).exists():
                    video_path_obj = Path(video_path)
                    video_path_parts = video_path_obj.parts
                    if len(video_path_parts) >= 2:
                        video_id = video_path_parts[-2]
                        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                        video_ids_in_range.add(video_id)
            except Exception:
                continue
        
        logger.info(f"Stage 1: Will delete augmentations for {len(video_ids_in_range)} videos in range")
        
        # Delete augmented video files only for videos in this range
        # SAFETY: Only delete files matching our augmentation pattern in our output directory
        # CRITICAL: Never delete files from other stages (e.g., scaled videos, features, etc.)
        aug_files_deleted = 0
        for aug_file in output_dir.glob("*_aug*.mp4"):
            # SAFETY CHECK: Ensure we're only deleting Stage 1 augmentation files
            # Pattern must be: {video_id}_aug{idx}.mp4 (exactly, no other patterns)
            aug_filename = aug_file.stem  # Remove .mp4 extension
            if "_aug" not in aug_filename:
                logger.warning(f"SKIPPING deletion - not an augmentation file: {aug_file.name}")
                continue
            
            # Extract video_id from filename (format: {video_id}_aug{idx}.mp4)
            parts = aug_filename.split("_aug")
            if len(parts) != 2:
                logger.warning(f"SKIPPING deletion - invalid augmentation filename format: {aug_file.name}")
                continue
            
            file_video_id = parts[0]
            try:
                aug_idx = int(parts[1])  # Should be a number
            except ValueError:
                logger.warning(f"SKIPPING deletion - invalid augmentation index: {aug_file.name}")
                continue
            
            # Only delete if this video_id is in our processing range
            if file_video_id in video_ids_in_range:
                # SAFETY: Final check - ensure filename matches expected pattern exactly
                expected_pattern = f"{file_video_id}_aug{aug_idx}.mp4"
                if aug_file.name == expected_pattern:
                    aug_file.unlink()
                    aug_files_deleted += 1
                    logger.debug(f"Deleted existing augmentation: {aug_file.name}")
                else:
                    logger.warning(f"SKIPPING deletion - filename mismatch: {aug_file.name} != {expected_pattern}")
        
        logger.info(f"Stage 1: Deleted {aug_files_deleted} existing augmentation files in range")
        
        # Delete metadata entries for videos in this range (if metadata exists)
        # Support CSV/Arrow/Parquet
        metadata_paths = [
            output_dir / "augmented_metadata.arrow",
            output_dir / "augmented_metadata.parquet",
            output_dir / "augmented_metadata.csv"
        ]
        metadata_path_to_use = None
        for mp in metadata_paths:
            if mp.exists():
                metadata_path_to_use = mp
                break
        
        if metadata_path_to_use and metadata_path_to_use.exists():
            try:
                if metadata_path_to_use.suffix == '.arrow':
                    existing_metadata = pl.read_ipc(metadata_path_to_use)
                elif metadata_path_to_use.suffix == '.parquet':
                    existing_metadata = pl.read_parquet(metadata_path_to_use)
                else:
                    existing_metadata = pl.read_csv(str(metadata_path_to_use))
                # Filter out entries for videos in this range
                rows_to_keep = []
                for row in existing_metadata.iter_rows(named=True):
                    original_video = row.get("original_video", "")
                    # Extract video_id from original_video path
                    video_path_obj = Path(original_video)
                    if len(video_path_obj.parts) >= 2:
                        video_id = video_path_obj.parts[-2]
                        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                        if video_id not in video_ids_in_range:
                            rows_to_keep.append(row)
                
                # Rewrite metadata without deleted entries
                if len(rows_to_keep) < existing_metadata.height:
                    deleted_count = existing_metadata.height - len(rows_to_keep)
                    logger.info(f"Stage 1: Removing {deleted_count} metadata entries for videos in range")
                    # Create new DataFrame from kept rows
                    if rows_to_keep:
                        new_metadata = pl.DataFrame(rows_to_keep)
                        # Save atomically with file locking
                        new_metadata_path = output_dir / "augmented_metadata.arrow"
                        success = write_metadata_atomic(new_metadata, new_metadata_path)
                        
                        if not success:
                            # Fallback to Parquet
                            new_metadata_path = output_dir / "augmented_metadata.parquet"
                            success = write_metadata_atomic(new_metadata, new_metadata_path)
                        
                        if success:
                            # Remove old file if different format
                            if metadata_path_to_use != new_metadata_path and metadata_path_to_use.exists():
                                try:
                                    metadata_path_to_use.unlink()
                                except Exception:
                                    pass
                            logger.info(f"Stage 1: Updated metadata file, kept {len(rows_to_keep)} entries")
                        else:
                            logger.warning(f"Stage 1: Failed to update metadata file atomically")
                    else:
                        # No entries left, delete metadata file
                        metadata_path.unlink()
                        logger.info(f"Stage 1: Deleted metadata file (no entries remaining)")
                else:
                    logger.info(f"Stage 1: No metadata entries to delete for this range")
            except Exception as e:
                logger.warning(f"Could not update metadata file: {e}, will regenerate entries")
        
        logger.info("Stage 1: Range-specific cleanup complete")
    
    # Prepare metadata rows list (we'll write to Arrow/Parquet at the end)
    # For backward compatibility, we'll collect all new rows and append to existing metadata
    metadata_rows = []
    total_videos_processed = 0
    
    # Process each video one at a time
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Stage 1: Processing video {idx + 1}/{df.height}: {Path(video_path).name}")
            logger.info(f"{'='*80}")
            
            log_memory_stats(f"Stage 1: before video {idx + 1}")
            
            # Save original video metadata
            # Extract unique ID from video path (e.g., "IJfOsFABDwY" from "FVC1/youtube/IJfOsFABDwY/video.mp4")
            video_path_obj = Path(video_path)
            video_path_parts = video_path_obj.parts
            
            # Get unique identifier from parent directory or use hash
            if len(video_path_parts) >= 2:
                # Parent directory is usually the unique ID (e.g., "IJfOsFABDwY")
                video_id = video_path_parts[-2]  # Parent of "video.mp4"
            else:
                # Fallback: use hash of full path
                import hashlib
                video_id = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
            
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            
            # Check if this video already has all augmentations
            if video_id in existing_video_ids_with_all_augs and not delete_existing:
                # Double-check that all augmentation files actually exist
                all_augmentations_exist = True
                for aug_idx in range(num_augmentations):
                    aug_path = output_dir / f"{video_id}_aug{aug_idx}.mp4"
                    if not aug_path.exists():
                        all_augmentations_exist = False
                        logger.warning(f"Video {video_id} missing augmentation {aug_idx}, will regenerate")
                        break
                
                if all_augmentations_exist:
                    logger.info(f"Video {video_id} already has all {num_augmentations} augmentations, skipping...")
                    continue
            
            original_output = output_dir / f"{video_id}_original.mp4"
            if not original_output.exists():
                import shutil
                # Ensure output directory exists before copying
                original_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_path, original_output)
            
            # Check if original is already in metadata
            original_already_in_metadata = False
            if existing_metadata is not None:
                for row in existing_metadata.iter_rows(named=True):
                    if row.get("original_video") == video_rel and row.get("is_original") == True:
                        original_already_in_metadata = True
                        break
            
            if not original_already_in_metadata:
                metadata_rows.append({
                    "video_path": str(original_output.relative_to(project_root)),
                    "label": label,
                    "original_video": video_rel,
                    "augmentation_idx": -1,  # -1 indicates original
                    "is_original": True
                })
                total_videos_processed += 1
            
            # Generate augmentations with chunked processing
            # Process in smaller chunks (default 250 frames) to reduce memory usage
            chunk_size = 250
            
            # Create checkpoint directory for this video (persistent, can resume)
            checkpoint_dir = output_dir / ".checkpoints" / video_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            augmented_paths = augment_video(
                video_path, 
                num_augmentations=num_augmentations,
                chunk_size=chunk_size,
                checkpoint_dir=checkpoint_dir,
                resume=not delete_existing,  # Resume if not deleting existing
                output_dir=output_dir,
                video_id=video_id,
            )
            
            # Get FPS from original video
            try:
                container = av.open(video_path)
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                container.close()
            except Exception as e:
                logger.warning(f"Failed to get FPS from {video_path}: {e}, using default 30.0")
                fps = 30.0
            
            if not augmented_paths:
                logger.warning(f"No augmentations generated for {video_path}")
                continue
            
            # Save augmented video metadata
            logger.info(f"Generated {len(augmented_paths)} augmentations, saving metadata...")
            for aug_idx in range(num_augmentations):
                aug_filename = f"{video_id}_aug{aug_idx}.mp4"
                aug_path = output_dir / aug_filename
                
                # At this point augment_video has already written the final files to disk.
                # We just need to verify existence and write metadata.
                if not aug_path.exists():
                    logger.error(f"✗ Augmentation {aug_idx + 1} missing at expected path {aug_path}")
                    continue
                
                aug_path_rel = str(aug_path.relative_to(project_root))
                
                # If augmentation already exists and we're not deleting, ensure metadata is present
                metadata_entry_exists = False
                if existing_metadata is not None:
                    for row in existing_metadata.iter_rows(named=True):
                        if (row.get("video_path") == aug_path_rel and 
                            row.get("original_video") == video_rel and 
                            row.get("augmentation_idx") == aug_idx):
                            metadata_entry_exists = True
                            break
                
                if not metadata_entry_exists:
                    metadata_rows.append({
                        "video_path": aug_path_rel,
                        "label": label,
                        "original_video": video_rel,
                        "augmentation_idx": aug_idx,
                        "is_original": False
                    })
                total_videos_processed += 1
                logger.info(f"✓ Augmentation {aug_idx + 1}/{num_augmentations} available at {aug_path}")
            
        except Exception as e:
            logger.error(f"Error processing video {video_rel}: {e}", exc_info=True)
            continue
    
    # Reconstruct metadata from existing files if needed
    # This ensures that if all augmentations already exist and were skipped,
    # or if the metadata file is missing/corrupted, we rebuild it from the files
    needs_reconstruction = False
    # Check if any metadata file exists
    metadata_exists = (metadata_path_arrow.exists() or 
                      metadata_path_parquet.exists() or 
                      metadata_path_csv.exists())
    if not metadata_exists:
        needs_reconstruction = True
        logger.info("Stage 1: Metadata file missing, reconstructing from existing augmentation files...")
    elif metadata_path and metadata_path.exists() and metadata_path.stat().st_size == 0:
        needs_reconstruction = True
        logger.info("Stage 1: Metadata file is empty, reconstructing from existing augmentation files...")
    else:
        # Check if metadata is incomplete (has fewer entries than expected files)
        try:
            if metadata_path.suffix == '.arrow':
                existing_metadata_df = pl.read_ipc(metadata_path)
            elif metadata_path.suffix == '.parquet':
                existing_metadata_df = pl.read_parquet(metadata_path)
            else:
                existing_metadata_df = pl.read_csv(str(metadata_path))
            # Count expected files: original + augmentations for each video in range
            expected_entries = df.height * (1 + num_augmentations)  # 1 original + num_augmentations per video
            if existing_metadata_df.height < expected_entries * 0.5:  # If less than 50% of expected
                needs_reconstruction = True
                logger.info(f"Stage 1: Metadata appears incomplete ({existing_metadata_df.height} entries, expected ~{expected_entries}), reconstructing...")
        except Exception:
            needs_reconstruction = True
            logger.info("Stage 1: Could not read metadata file, reconstructing from existing augmentation files...")
    
    if needs_reconstruction:
        _reconstruct_metadata_from_files(metadata_path, output_dir, project_root, df, num_augmentations)
    
    # Load final metadata (support Arrow/Parquet/CSV)
    metadata_paths = [
        output_dir / "augmented_metadata.arrow",
        output_dir / "augmented_metadata.parquet",
        output_dir / "augmented_metadata.csv"
    ]
    metadata_path_found = None
    for mp in metadata_paths:
        if mp.exists():
            metadata_path_found = mp
            break
    
    if metadata_path_found:
        try:
            if metadata_path_found.suffix == '.arrow':
                metadata_df = pl.read_ipc(metadata_path_found)
            elif metadata_path_found.suffix == '.parquet':
                metadata_df = pl.read_parquet(metadata_path_found)
            else:
                metadata_df = pl.read_csv(str(metadata_path_found))
            logger.info(f"\n✓ Stage 1 complete: Metadata available at {metadata_path_found}")
            logger.info(f"✓ Stage 1: Total entries in metadata: {metadata_df.height}")
            if total_videos_processed > 0:
                logger.info(f"✓ Stage 1: Processed {total_videos_processed} videos in this run")
            return metadata_df
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
            return pl.DataFrame()
    else:
        logger.error("Stage 1: No metadata file found and could not reconstruct!")
        return pl.DataFrame()

