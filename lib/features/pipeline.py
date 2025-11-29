"""
Handcrafted feature extraction pipeline.

Extracts handcrafted features from videos including:
- Noise residual features
- DCT statistics
- Blur/sharpness metrics
- Boundary inconsistency
- Codec cues
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import av

from lib.data import load_metadata
from lib.utils.paths import resolve_video_path
from lib.utils.memory import aggressive_gc, log_memory_stats
from lib.features.handcrafted import extract_all_features, HandcraftedFeatureExtractor

logger = logging.getLogger(__name__)


def extract_features_from_video(
    video_path: str,
    num_frames: int = 5,
    extractor: Optional[HandcraftedFeatureExtractor] = None
) -> dict:
    """
    Extract features from a video by sampling frames.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        extractor: Feature extractor instance
    
    Returns:
        Dictionary of aggregated features
    """
    if extractor is None:
        extractor = HandcraftedFeatureExtractor()
    
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
                    features = extractor.extract(frame_array, video_path)
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
        logger.error(f"Failed to extract features from {video_path}: {e}")
        return {}
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        aggressive_gc(clear_cuda=False)


def stage2_extract_features(
    project_root: str,
    augmented_metadata_path: str,
    output_dir: str = "data/features_stage2",
    num_frames: int = 5
) -> pl.DataFrame:
    """
    Stage 2: Extract handcrafted features from all augmented videos.
    
    Args:
        project_root: Project root directory
        augmented_metadata_path: Path to augmented metadata CSV
        output_dir: Directory to save features
        num_frames: Number of frames to sample per video
    
    Returns:
        DataFrame with feature metadata
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load augmented metadata
    logger.info("Stage 2: Loading augmented metadata...")
    if not Path(augmented_metadata_path).exists():
        logger.error(f"Augmented metadata not found: {augmented_metadata_path}")
        return pl.DataFrame()
    
    df = pl.read_csv(augmented_metadata_path)
    logger.info(f"Stage 2: Processing {df.height} videos")
    
    extractor = HandcraftedFeatureExtractor()
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
                log_memory_stats(f"Stage 2: processing video {idx + 1}/{df.height}")
            
            # Extract features
            features = extract_features_from_video(video_path, num_frames, extractor)
            
            if not features:
                logger.warning(f"No features extracted from {video_path}")
                continue
            
            # Save features as .npy
            video_id = Path(video_path).stem
            feature_path = output_dir / f"{video_id}_features.npy"
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
        logger.error("Stage 2: No features extracted!")
        return pl.DataFrame()
    
    # Create DataFrame
    features_df = pl.DataFrame(feature_rows)
    
    # Save metadata
    metadata_path = output_dir / "features_metadata.csv"
    features_df.write_csv(str(metadata_path))
    logger.info(f"✓ Stage 2 complete: Saved features to {output_dir}")
    logger.info(f"✓ Stage 2: Extracted features from {len(feature_rows)} videos")
    
    return features_df

