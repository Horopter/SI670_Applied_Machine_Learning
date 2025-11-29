#!/usr/bin/env python3
"""
Simple one-video-at-a-time preprocessing script.

Processes videos sequentially:
1. Load one video
2. Generate N augmentations (configurable, default 1)
3. Save augmented videos to disk
4. Move to next video

No pipeline overhead, no fixed-size preprocessing, original resolution preserved.
"""

from __future__ import annotations

import os
import sys
import logging
import hashlib
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
import polars as pl
import av

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.video_data import load_metadata, filter_existing_videos
from lib.video_paths import resolve_video_path
from lib.mlops_utils import aggressive_gc, log_memory_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_video_frames(video_path: str) -> List[np.ndarray]:
    """Load all frames from a video, preserving original resolution."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        frames = []
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                # Convert to RGB numpy array
                frame_array = frame.to_ndarray(format='rgb24')  # (H, W, 3)
                frames.append(frame_array)
        
        container.close()
        return frames
    except Exception as e:
        logger.error(f"Failed to load video {video_path}: {e}")
        return []


def save_video_frames(frames: List[np.ndarray], output_path: str, fps: float = 30.0):
    """Save frames as a video file, preserving original resolution."""
    if not frames:
        logger.warning(f"No frames to save for {output_path}")
        return False
    
    try:
        # Get frame dimensions from first frame
        height, width = frames[0].shape[:2]
        
        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open output container
        container = av.open(str(output_path), mode='w')
        stream = container.add_stream('libx264', rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        
        # Encode frames
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
        
        container.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save video {output_path}: {e}")
        return False


def apply_spatial_augmentation(frame: np.ndarray, aug_type: str, seed: int) -> np.ndarray:
    """Apply a single spatial augmentation to a frame."""
    random.seed(seed)
    np.random.seed(seed)
    
    if aug_type == 'none':
        return frame
    
    # Convert to PIL for some transforms
    from PIL import Image
    pil_image = Image.fromarray(frame)
    
    if aug_type == 'rotation':
        angle = random.uniform(-15, 15)
        pil_image = pil_image.rotate(angle, fillcolor=(0, 0, 0))
    
    elif aug_type == 'flip':
        if random.random() < 0.5:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    
    elif aug_type == 'brightness':
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(pil_image)
        factor = random.uniform(0.7, 1.3)
        pil_image = enhancer.enhance(factor)
    
    elif aug_type == 'contrast':
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(pil_image)
        factor = random.uniform(0.7, 1.3)
        pil_image = enhancer.enhance(factor)
    
    elif aug_type == 'gaussian_noise':
        frame_array = np.array(pil_image)
        noise = np.random.normal(0, 10, frame_array.shape).astype(np.float32)
        frame_array = np.clip(frame_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_array)
    
    return np.array(pil_image)


def augment_video(
    video_path: str,
    num_augmentations: int = 1,
    augmentation_types: Optional[List[str]] = None
) -> List[List[np.ndarray]]:
    """
    Generate augmented versions of a video.
    
    Args:
        video_path: Path to input video
        num_augmentations: Number of augmented versions to generate
        augmentation_types: List of augmentation types to apply (or None for random selection)
    
    Returns:
        List of augmented frame sequences (each is List[np.ndarray])
    """
    # Load original video
    logger.info(f"Loading video: {video_path}")
    original_frames = load_video_frames(video_path)
    
    if not original_frames:
        logger.warning(f"No frames loaded from {video_path}")
        return []
    
    logger.info(f"Loaded {len(original_frames)} frames from {video_path}")
    
    # Generate deterministic seed from video path
    video_path_str = str(video_path)
    base_seed = int(hashlib.md5(video_path_str.encode()).hexdigest()[:8], 16) % (2**31)
    
    # Default augmentation types
    if augmentation_types is None:
        augmentation_types = ['rotation', 'flip', 'brightness', 'contrast', 'gaussian_noise']
    
    augmented_videos = []
    
    for aug_idx in range(num_augmentations):
        # Set seed for this augmentation
        aug_seed = base_seed + aug_idx
        random.seed(aug_seed)
        np.random.seed(aug_seed)
        
        # Select augmentation type for this version
        aug_type = random.choice(augmentation_types) if len(augmentation_types) > 1 else augmentation_types[0]
        
        # Apply augmentation to all frames
        augmented_frames = []
        for frame in original_frames:
            augmented_frame = apply_spatial_augmentation(frame, aug_type, aug_seed + len(augmented_frames))
            augmented_frames.append(augmented_frame)
        
        augmented_videos.append(augmented_frames)
        logger.info(f"Generated augmentation {aug_idx + 1}/{num_augmentations} with type '{aug_type}'")
    
    return augmented_videos


def process_videos(
    project_root: str,
    num_augmentations: int = 1,
    output_dir: str = "videos_augmented",
    augmentation_types: Optional[List[str]] = None
):
    """
    Process all videos one at a time.
    
    Args:
        project_root: Project root directory
        num_augmentations: Number of augmentations per video
        output_dir: Directory to save augmented videos
        augmentation_types: List of augmentation types to use
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    logger.info("Loading video metadata...")
    metadata_path = project_root / "data" / "video_index_input.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return
    
    df = load_metadata(str(metadata_path))
    df = filter_existing_videos(df, str(project_root))
    
    logger.info(f"Found {df.height} videos to process")
    logger.info(f"Generating {num_augmentations} augmentation(s) per video")
    logger.info(f"Output directory: {output_dir}")
    
    # Metadata for augmented videos
    augmented_metadata = []
    
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
            logger.info(f"Processing video {idx + 1}/{df.height}: {Path(video_path).name}")
            logger.info(f"{'='*80}")
            
            # Log memory before processing
            log_memory_stats(f"before processing video {idx + 1}")
            
            # Generate augmentations
            augmented_videos = augment_video(
                video_path,
                num_augmentations=num_augmentations,
                augmentation_types=augmentation_types
            )
            
            if not augmented_videos:
                logger.warning(f"No augmentations generated for {video_path}")
                continue
            
            # Save augmented videos
            video_id = Path(video_path).stem
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            
            for aug_idx, aug_frames in enumerate(augmented_videos):
                # Determine output path
                aug_filename = f"{video_id}_aug{aug_idx}.mp4"
                aug_path = output_dir / aug_filename
                
                # Get original video FPS (default to 30)
                try:
                    container = av.open(video_path)
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate) if stream.average_rate else 30.0
                    container.close()
                except:
                    fps = 30.0
                
                # Save augmented video
                logger.info(f"Saving augmentation {aug_idx + 1} to {aug_path}")
                success = save_video_frames(aug_frames, str(aug_path), fps=fps)
                
                if success:
                    # Record metadata
                    aug_path_rel = str(aug_path.relative_to(project_root))
                    augmented_metadata.append({
                        "video_path": aug_path_rel,
                        "label": label,
                        "original_video": video_rel,
                        "augmentation_idx": aug_idx,
                    })
                    logger.info(f"✓ Saved: {aug_path}")
                else:
                    logger.error(f"✗ Failed to save: {aug_path}")
            
            # Clear memory
            del augmented_videos
            aggressive_gc(clear_cuda=False)
            
            # Log memory after processing
            log_memory_stats(f"after processing video {idx + 1}")
            
        except Exception as e:
            logger.error(f"Failed to process video {video_rel}: {e}", exc_info=True)
            continue
    
    # Save metadata
    if augmented_metadata:
        metadata_df = pl.DataFrame(augmented_metadata)
        metadata_path = output_dir / "augmented_metadata.csv"
        metadata_df.write_csv(str(metadata_path))
        logger.info(f"\n✓ Saved metadata to {metadata_path}")
        logger.info(f"✓ Generated {len(augmented_metadata)} augmented videos")
    else:
        logger.error("No augmented videos generated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple one-video-at-a-time preprocessing")
    parser.add_argument(
        "--project-root",
        type=str,
        default=os.getcwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=1,
        help="Number of augmentations per video (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos_augmented",
        help="Output directory for augmented videos (default: videos_augmented)"
    )
    parser.add_argument(
        "--augmentation-types",
        type=str,
        nargs="+",
        default=None,
        help="Augmentation types to use (default: random selection from rotation, flip, brightness, contrast, gaussian_noise)"
    )
    
    args = parser.parse_args()
    
    process_videos(
        project_root=args.project_root,
        num_augmentations=args.num_augmentations,
        output_dir=args.output_dir,
        augmentation_types=args.augmentation_types
    )

