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
    
    # Read video
    video = _read_video_wrapper(video_path)
    if video.shape[0] == 0:
        logger.warning(f"Video has no frames: {video_path}")
        return []
    
    total_frames = video.shape[0]
    
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
        
        # Sample frames (uniform sampling)
        from .video_modeling import uniform_sample_indices
        indices = uniform_sample_indices(total_frames, config.num_frames)
        
        frames = []
        for i in indices:
            frame = video[i].numpy().astype(np.uint8)  # (H, W, C)
            frame_tensor = spatial_transform(frame)  # (C, H, W)
            
            # Apply post-tensor augmentations
            if post_tensor_transform is not None:
                frame_tensor = post_tensor_transform(frame_tensor)
            
            frames.append(frame_tensor)
        
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
        augmented_clips.append(clip)
        
        # Always save if save_dir provided
        if save_dir:
            video_id = Path(video_path).stem
            # Sanitize filename (remove special characters)
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            save_path = Path(save_dir) / f"{video_id}_aug{aug_idx}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(clip, save_path)
            logger.debug("Saved augmented clip: %s", save_path)
    
    return augmented_clips


def pregenerate_augmented_dataset(
    df: pl.DataFrame,
    project_root: str,
    config: VideoConfig,
    output_dir: str,
    num_augmentations_per_video: int = 3,
    batch_size: int = 10,  # Process videos in batches to reduce memory
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
    
    augmented_rows = []
    
    # Process videos in batches to reduce memory usage
    import gc
    from .mlops_utils import aggressive_gc
    
    for batch_start in tqdm(range(0, df.height, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, df.height)
        batch_df = df.slice(batch_start, batch_end - batch_start)
        
        for idx in range(batch_df.height):
            row = batch_df.row(idx, named=True)
            video_rel = row["video_path"]
            label = row["label"]
            
            try:
                video_path = resolve_video_path(video_rel, project_root)
                
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
                
                # Create entries for each augmented clip
                # Clips are already saved by generate_augmented_clips
                video_id = Path(video_path).stem
                # Sanitize filename (same as in generate_augmented_clips)
                video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
                
                for aug_idx in range(len(clips)):
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
                    
                    augmented_rows.append({
                        "video_path": clip_path_rel,
                        "label": label,
                        "original_video": video_rel,
                        "augmentation_idx": aug_idx,
                    })
                
                # Clear video from memory after processing
                del clips
                
            except Exception as e:
                logger.error("Failed to process video %s: %s", video_rel, str(e))
                continue
        
        # Aggressive GC after each batch
        aggressive_gc(clear_cuda=True)
    
    # Create new DataFrame
    if augmented_rows:
        augmented_df = pl.DataFrame(augmented_rows)
        logger.info("✓ Generated %d augmented clips from %d videos", 
                   augmented_df.height, df.height)
        logger.info("  Average: %.1f augmentations per video", 
                   augmented_df.height / max(1, df.height))
        return augmented_df
    else:
        logger.error("No augmented clips generated!")
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

