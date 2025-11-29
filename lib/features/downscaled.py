"""
Extract features from downscaled videos.

Extracts features that are detectable after downscaling, focusing on:
- Edge preservation metrics
- Texture uniformity
- Compression artifact visibility
- Color consistency
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import av
import cv2

from lib.data import load_metadata
from lib.utils.paths import resolve_video_path
from lib.utils.memory import aggressive_gc, log_memory_stats
from lib.features.handcrafted import HandcraftedFeatureExtractor

logger = logging.getLogger(__name__)


def extract_downscaled_features(
    video_path: str,
    num_frames: int = 5
) -> dict:
    """
    Extract features specific to downscaled videos.
    
    Focuses on features that are detectable after downscaling:
    - Edge preservation metrics
    - Texture uniformity
    - Compression artifact visibility
    - Color consistency
    
    Args:
        video_path: Path to downscaled video file
        num_frames: Number of frames to sample
    
    Returns:
        Dictionary of downscaled-specific features
    """
    container = None
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames if stream.frames > 0 else 0
        
        if total_frames == 0:
            logger.warning(f"Video has no frames: {video_path}")
            return {}
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        all_features = []
        frame_count = 0
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_count in frame_indices:
                    frame_array = frame.to_ndarray(format='rgb24')
                    
                    # Extract downscaled-specific features
                    features = {}
                    
                    # Edge preservation (Canny edges)
                    gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    features["edge_density"] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
                    
                    # Texture uniformity (variance of local means)
                    kernel = np.ones((5, 5), np.float32) / 25
                    local_means = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                    features["texture_uniformity"] = float(1.0 / (1.0 + np.std(local_means)))
                    
                    # Color consistency (variance across channels)
                    features["color_consistency_r"] = float(np.std(frame_array[:, :, 0]))
                    features["color_consistency_g"] = float(np.std(frame_array[:, :, 1]))
                    features["color_consistency_b"] = float(np.std(frame_array[:, :, 2]))
                    
                    # Compression artifacts (blockiness)
                    h, w = gray.shape
                    block_size = 8
                    blockiness = 0.0
                    for i in range(0, h - block_size, block_size):
                        for j in range(0, w - block_size, block_size):
                            block = gray[i:i+block_size, j:j+block_size]
                            # Measure horizontal and vertical discontinuities
                            h_diff = np.mean(np.abs(np.diff(block, axis=1)))
                            v_diff = np.mean(np.abs(np.diff(block, axis=0)))
                            blockiness += h_diff + v_diff
                    features["compression_artifacts"] = float(blockiness / ((h // block_size) * (w // block_size)))
                    
                    all_features.append(features)
                
                frame_count += 1
                if frame_count >= total_frames or len(all_features) >= num_frames:
                    break
            
            if frame_count >= total_frames or len(all_features) >= num_frames:
                break
        
        # Aggregate features across frames (mean)
        if not all_features:
            return {}
        
        aggregated = {}
        for key in all_features[0].keys():
            values = [f[key] for f in all_features if key in f]
            aggregated[key] = float(np.mean(values)) if values else 0.0
        
        return aggregated
        
    except Exception as e:
        logger.error(f"Failed to extract downscaled features from {video_path}: {e}")
        return {}
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        aggressive_gc(clear_cuda=False)


def stage4_extract_downscaled_features(
    project_root: str,
    downscaled_metadata_path: str,
    output_dir: str = "data/features_stage4",
    num_frames: int = 5
) -> pl.DataFrame:
    """
    Stage 4: Extract additional features from downscaled videos.
    
    Args:
        project_root: Project root directory
        downscaled_metadata_path: Path to downscaled metadata CSV
        output_dir: Directory to save features
        num_frames: Number of frames to sample per video
    
    Returns:
        DataFrame with feature metadata
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load downscaled metadata
    logger.info("Stage 4: Loading downscaled metadata...")
    if not Path(downscaled_metadata_path).exists():
        logger.error(f"Downscaled metadata not found: {downscaled_metadata_path}")
        return pl.DataFrame()
    
    df = pl.read_csv(downscaled_metadata_path)
    logger.info(f"Stage 4: Processing {df.height} downscaled videos")
    
    feature_rows = []
    
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            if idx % 10 == 0:
                log_memory_stats(f"Stage 4: processing video {idx + 1}/{df.height}")
            
            # Extract downscaled-specific features
            features = extract_downscaled_features(video_path, num_frames)
            
            if not features:
                logger.warning(f"No features extracted from {video_path}")
                continue
            
            # Save features as .npy
            video_id = Path(video_path).stem
            feature_path = output_dir / f"{video_id}_downscaled_features.npy"
            np.save(str(feature_path), features)
            
            # Create metadata row
            feature_row = {
                "video_path": video_rel,
                "label": label,
                "feature_path": str(feature_path.relative_to(project_root)),
            }
            feature_row.update(features)  # Add all feature values
            feature_rows.append(feature_row)
            
            aggressive_gc(clear_cuda=False)
            
        except Exception as e:
            logger.error(f"Error processing {video_rel}: {e}", exc_info=True)
            continue
    
    if not feature_rows:
        logger.error("Stage 4: No features extracted!")
        return pl.DataFrame()
    
    # Create DataFrame
    features_df = pl.DataFrame(feature_rows)
    
    # Save metadata
    metadata_path = output_dir / "features_downscaled_metadata.csv"
    features_df.write_csv(str(metadata_path))
    logger.info(f"✓ Stage 4 complete: Saved features to {output_dir}")
    logger.info(f"✓ Stage 4: Extracted features from {len(feature_rows)} videos")
    
    return features_df

