"""
Pre-generation pipeline for video augmentations.

This module generates and stores augmented video data BEFORE training,
allowing for:
- Faster training (no augmentation overhead during training)
- Reproducibility (same augmentations across runs)
- Memory efficiency (can pre-process and cache)
"""

from __future__ import annotations

import os
import logging
import random
from pathlib import Path
from typing import List, Optional
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from .video_modeling import VideoConfig, _read_video_wrapper
from .video_augmentations import build_comprehensive_frame_transforms, apply_temporal_augmentations

logger = logging.getLogger(__name__)


def generate_augmented_clips(
    video_path: str,
    config: VideoConfig,
    num_augmentations: int = 1,
    save_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Generate augmented clips from a video.
    
    Args:
        video_path: Path to source video
        config: VideoConfig with augmentation settings
        num_augmentations: Number of augmented versions to generate
        save_dir: Directory to save augmented clips (optional)
        seed: Random seed for deterministic augmentations (uses video path hash if None)
    
    Returns:
        List of augmented clip tensors (each is T, C, H, W)
    """
    # Generate deterministic seed from video path if not provided
    # This ensures same video gets same augmentations across runs/folds
    if seed is None:
        import hashlib
        video_path_str = str(video_path)
        seed = int(hashlib.md5(video_path_str.encode()).hexdigest()[:8], 16) % (2**31)
    
    # CRITICAL MEMORY OPTIMIZATION: Don't load entire video into memory
    # Instead, get video metadata first, then decode only the frames we need
    from .mlops_utils import log_memory_stats, aggressive_gc
    
    # Get video frame count without loading all frames
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames == 0:
            # Fallback: try to get from duration and fps
            total_frames = int(stream.duration * stream.average_rate / stream.time_base) if stream.duration else 0
        container.close()
    except Exception:
        # Fallback: load video to get frame count (unavoidable for some formats)
        log_memory_stats(f"before loading video (fallback): {Path(video_path).name}")
        video = _read_video_wrapper(video_path)
        total_frames = video.shape[0]
        if total_frames == 0:
            logger.warning(f"Video has no frames: {video_path}")
            return []
        del video  # Clear immediately
        aggressive_gc(clear_cuda=False)
    
    if total_frames == 0:
        logger.warning(f"Video has no frames: {video_path}")
        return []
    
    augmented_clips = []
    
    for aug_idx in range(num_augmentations):
        # Set seed for this augmentation (deterministic per video + aug_idx)
        aug_seed = seed + aug_idx
        random.seed(aug_seed)
        np.random.seed(aug_seed)
        torch.manual_seed(aug_seed)
        
        # Build augmentation transforms (with seed for reproducibility)
        spatial_transform, post_tensor_transform = build_comprehensive_frame_transforms(
            train=True,  # Always use augmentations for generation
            fixed_size=config.fixed_size,
            max_size=config.max_size,
            augmentation_config=config.augmentation_config,
        )
        
        # Sample frame indices (uniform sampling)
        from .video_modeling import uniform_sample_indices
        indices = uniform_sample_indices(total_frames, config.num_frames)
        
        # CRITICAL: Decode only the frames we need, not the entire video
        log_memory_stats(f"before decoding frames for aug {aug_idx}: {Path(video_path).name}")
        
        # Decode only selected frames using PyAV (memory-efficient)
        frames = []
        try:
            import av
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            # Get frame rate and time base for seeking
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            time_base = float(stream.time_base) if stream.time_base else 1.0 / fps
            
            # Decode frames one by one (only the ones we need)
            for frame_idx in sorted(indices):  # Sort to decode in order
                # Seek to approximate timestamp (in seconds)
                timestamp_sec = frame_idx / fps
                timestamp_pts = int(timestamp_sec / time_base)
                
                try:
                    # Seek to frame
                    container.seek(timestamp_pts, stream=stream)
                    
                    # Decode frames until we get the one we want
                    frame_count = 0
                    for packet in container.demux(stream):
                        for frame in packet.decode():
                            if frame_count == frame_idx or abs(frame_count - frame_idx) < 2:
                                # Convert to numpy array
                                frame_array = frame.to_ndarray(format='rgb24')  # (H, W, 3)
                                frame_tensor = spatial_transform(frame_array)  # (C, H, W)
                                
                                # Apply post-tensor augmentations
                                if post_tensor_transform is not None:
                                    frame_tensor = post_tensor_transform(frame_tensor)
                                
                                frames.append(frame_tensor)
                                break
                            frame_count += 1
                        if len(frames) > len(indices):
                            break
                    if len(frames) >= len(indices):
                        break
                except Exception as seek_error:
                    logger.debug(f"Seek failed for frame {frame_idx}: {seek_error}, trying alternative method")
                    # Alternative: decode all frames but only keep the ones we need
                    container.seek(0)
                    frame_count = 0
                    for packet in container.demux(stream):
                        for frame in packet.decode():
                            if frame_count in indices:
                                frame_array = frame.to_ndarray(format='rgb24')
                                frame_tensor = spatial_transform(frame_array)
                                if post_tensor_transform is not None:
                                    frame_tensor = post_tensor_transform(frame_tensor)
                                frames.append(frame_tensor)
                            frame_count += 1
                            if len(frames) >= len(indices):
                                break
                        if len(frames) >= len(indices):
                            break
                    break
            
            container.close()
            
            # If we didn't get enough frames, fall back to full video loading
            if len(frames) < len(indices):
                raise ValueError(f"Only decoded {len(frames)}/{len(indices)} frames")
                
        except Exception as e:
            # Fallback: load entire video if frame-by-frame decoding fails
            logger.warning(f"Frame-by-frame decoding failed for {video_path}: {e}. Loading full video (memory intensive).")
            log_memory_stats(f"before loading full video (fallback): {Path(video_path).name}")
            video = _read_video_wrapper(video_path)
            video_size_mb = video.numel() * video.element_size() / 1024 / 1024
            logger.warning("Loaded full video: %.2f MB (shape: %s) - this is memory intensive!", video_size_mb, video.shape)
            
            for i in indices:
                if i < video.shape[0]:
                    frame = video[i].numpy().astype(np.uint8)  # (H, W, C)
                    frame_tensor = spatial_transform(frame)  # (C, H, W)
                    
                    # Apply post-tensor augmentations
                    if post_tensor_transform is not None:
                        frame_tensor = post_tensor_transform(frame_tensor)
                    
                    frames.append(frame_tensor)
            
            # Clear video immediately after extracting frames
            del video
            aggressive_gc(clear_cuda=False)
        
        log_memory_stats(f"after decoding frames for aug {aug_idx}: {Path(video_path).name}")
        
        # Apply temporal augmentations (with seed for reproducibility)
        frames = apply_temporal_augmentations(
            frames,
            train=True,
            frame_drop_prob=config.temporal_augmentation_config.get('frame_drop_prob', 0.1) if config.temporal_augmentation_config else 0.1,
            frame_dup_prob=config.temporal_augmentation_config.get('frame_dup_prob', 0.1) if config.temporal_augmentation_config else 0.1,
            reverse_prob=config.temporal_augmentation_config.get('reverse_prob', 0.1) if config.temporal_augmentation_config else 0.1,
            seed=aug_seed,  # Use same seed for temporal augmentations
        )
        
        # Ensure we have the right number of frames (pad or truncate if needed)
        target_frames = config.num_frames
        if len(frames) < target_frames:
            # Pad with last frame (repeat last frame)
            last_frame = frames[-1] if frames else None
            if last_frame is not None:
                while len(frames) < target_frames:
                    frames.append(last_frame.clone())
        elif len(frames) > target_frames:
            # Truncate to target_frames
            frames = frames[:target_frames]
        
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        
        # Clear frames from memory immediately after stacking
        del frames
        
        # Always save if save_dir provided
        if save_dir:
            video_id = Path(video_path).stem
            # Sanitize filename (remove special characters)
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            save_path = Path(save_dir) / f"{video_id}_aug{aug_idx}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(clip, save_path)
            logger.debug("Saved augmented clip: %s", save_path)
        
        # CRITICAL: Don't keep clips in memory - we only need the count
        # Clear clip immediately after saving (we'll verify file exists later)
        del clip
        
        # Aggressive GC after each augmentation
        from .mlops_utils import aggressive_gc
        aggressive_gc(clear_cuda=False)
    
    # Return count instead of actual clips to save memory
    # The caller can verify files exist by checking disk
    return [None] * num_augmentations  # Return list with correct length for counting


def pregenerate_augmented_dataset(
    df: pl.DataFrame,
    project_root: str,
    config: VideoConfig,
    output_dir: str,
    num_augmentations_per_video: int = 3,
    batch_size: int = 1,  # Extreme conservative: process one video at a time to minimize memory
) -> pl.DataFrame:
    """
    Pre-generate augmented clips for all videos in the dataset.
    
    Args:
        df: Polars DataFrame with video_path and label columns
        project_root: Project root directory
        config: VideoConfig with augmentation settings
        output_dir: Directory to save augmented clips
        num_augmentations_per_video: Number of augmented versions per video
        num_workers: Number of parallel workers (not used yet, for future)
        batch_size: Batch size for processing (not used yet)
    
    Returns:
        New DataFrame with paths to augmented clips
    """
    from .video_paths import resolve_video_path
    from .video_modeling import uniform_sample_indices  # Import once for efficiency
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Pre-generating augmented clips for %d videos...", df.height)
    logger.info("Output directory: %s", output_dir)
    logger.info("Augmentations per video: %d", num_augmentations_per_video)
    logger.info("This will create %d total augmented clips", df.height * num_augmentations_per_video)
    logger.info("Processing in batches of %d videos to reduce memory usage", batch_size)
    
    # CRITICAL: Write incrementally to avoid memory accumulation
    # Instead of accumulating all rows in memory, write to CSV incrementally
    metadata_path = Path(output_dir) / "augmented_metadata_temp.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize CSV with header
    import csv
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label", "original_video", "augmentation_idx"])
    
    total_rows_written = 0
    
    # Process videos one at a time to minimize memory
    import gc
    from .mlops_utils import aggressive_gc
    
    for batch_start in tqdm(range(0, df.height, batch_size), desc="Processing videos"):
        batch_end = min(batch_start + batch_size, df.height)
        batch_df = df.slice(batch_start, batch_end - batch_start)
        
        # Log memory before processing batch (every 10 videos)
        if batch_start % 10 == 0:
            from .mlops_utils import log_memory_stats
            log_memory_stats(f"before processing video {batch_start + 1}", detailed=True)
        
        for idx in range(batch_df.height):
            row = batch_df.row(idx, named=True)
            video_rel = row["video_path"]
            label = row["label"]
            
            try:
                video_path = resolve_video_path(video_rel, project_root)
                
                # Log memory before processing this video
                if idx == 0:  # Log for first video in batch
                    from .mlops_utils import log_memory_stats
                    log_memory_stats(f"before processing video: {Path(video_path).name}")
                
                # Generate augmented clips (saves automatically to save_dir)
                # Seed is deterministic based on video path, ensuring same augmentations across runs/folds
                clips = generate_augmented_clips(
                    video_path,
                    config,
                    num_augmentations=num_augmentations_per_video,
                    save_dir=str(output_path),
                    seed=None,  # Will use video path hash for deterministic seed
                )
                
                if not clips:
                    logger.warning("No clips generated for video: %s", video_path)
                    continue
                
                # Clips are already saved by generate_augmented_clips
                # We only need the count, not the actual clip tensors
                num_clips = len(clips)
                
                # Clear clips list immediately (clips are already saved to disk)
                del clips
                
                # Create entries for each augmented clip
                video_id = Path(video_path).stem
                # Sanitize filename (same as in generate_augmented_clips)
                video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                
                for aug_idx in range(num_clips):
                    clip_filename = f"{video_id}_aug{aug_idx}.pt"
                    clip_path = output_path / clip_filename
                    
                    # Verify clip was saved
                    if not clip_path.exists():
                        logger.warning("Clip not saved: %s", clip_path)
                        continue
                    
                    # Create metadata row (use relative path from project_root)
                    try:
                        clip_path_rel = str(clip_path.relative_to(Path(project_root)))
                    except ValueError:
                        # If not relative, use absolute path
                        clip_path_rel = str(clip_path)
                    
                    # Write immediately to CSV to avoid memory accumulation
                    with open(metadata_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([clip_path_rel, label, video_rel, aug_idx])
                    total_rows_written += 1
                
                # Aggressive GC after each video to prevent memory accumulation
                aggressive_gc(clear_cuda=False)
                
            except Exception as e:
                logger.error("Failed to process video %s: %s", video_rel, str(e))
                continue
        
        # Aggressive GC after each batch
        aggressive_gc(clear_cuda=True)
        
        # Log memory after batch if it's a checkpoint batch (every 10 videos)
        if (batch_start + batch_size) % 10 == 0 or batch_end >= df.height:
            from .mlops_utils import log_memory_stats
            log_memory_stats(f"after processing {batch_end} videos", detailed=True)
            logger.info("Written %d augmented clip rows so far", total_rows_written)
    
    # Load final DataFrame from CSV
    if total_rows_written > 0:
        logger.info("Loading augmented metadata from CSV...")
        augmented_df = pl.read_csv(str(metadata_path))
        
        # Rename temp file to final name
        final_metadata_path = Path(output_dir) / "augmented_metadata.csv"
        if final_metadata_path.exists():
            final_metadata_path.unlink()
        metadata_path.rename(final_metadata_path)
        
        logger.info("✓ Generated %d augmented clips from %d videos", 
                   augmented_df.height, df.height)
        logger.info("  Average: %.1f augmentations per video", 
                   augmented_df.height / max(1, df.height))
        return augmented_df
    else:
        logger.error("No augmented clips generated!")
        # Clean up temp file
        if metadata_path.exists():
            metadata_path.unlink()
        return pl.DataFrame()


def load_precomputed_clip(clip_path: str) -> torch.Tensor:
    """Load a pre-computed augmented clip."""
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Pre-computed clip not found: {clip_path}")
    return torch.load(clip_path, map_location='cpu')  # Load on CPU, will be moved to GPU by DataLoader


__all__ = [
    "generate_augmented_clips",
    "pregenerate_augmented_dataset",
    "load_precomputed_clip",
]

