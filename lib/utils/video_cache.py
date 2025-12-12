"""
Video metadata cache to avoid duplicate frame counting and video opening.

This module provides a caching system for video metadata (frame count, FPS, etc.)
to eliminate the massive waste of decoding entire videos multiple times.
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import av

logger = logging.getLogger(__name__)

# In-memory cache (persists for process lifetime)
_video_metadata_cache: Dict[str, Dict] = {}


def get_video_metadata_hash(video_path: str) -> str:
    """Get hash of video path and modification time for cache invalidation."""
    path = Path(video_path)
    if not path.exists():
        return ""
    
    # Hash based on path and modification time
    mtime = path.stat().st_mtime
    content = f"{video_path}:{mtime}"
    return hashlib.md5(content.encode()).hexdigest()


def get_video_metadata(
    video_path: str,
    use_cache: bool = True,
    cache_file: Optional[Path] = None
) -> Dict[str, any]:
    """
    Get video metadata (frame count, FPS, dimensions) with caching.
    
    This function caches results to avoid decoding videos multiple times.
    Cache is invalidated if video file is modified.
    
    Args:
        video_path: Path to video file
        use_cache: If True, use cache (default: True)
        cache_file: Optional persistent cache file path
    
    Returns:
        Dictionary with keys: 'total_frames', 'fps', 'width', 'height', 'duration'
    
    Raises:
        ValueError: If video_path is invalid
    """
    # Input validation
    if not video_path or not isinstance(video_path, str):
        raise ValueError(f"video_path must be a non-empty string, got: {type(video_path)}")
    
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        logger.warning(f"Video file not found: {video_path}")
        return {
            'total_frames': 0,
            'fps': 30.0,
            'width': 0,
            'height': 0,
            'duration': 0.0
        }
    
    if not video_path_obj.is_file():
        logger.warning(f"Video path is not a file: {video_path}")
        return {
            'total_frames': 0,
            'fps': 30.0,
            'width': 0,
            'height': 0,
            'duration': 0.0
        }
    
    # Check cache
    cache_key = get_video_metadata_hash(video_path)
    if use_cache and cache_key in _video_metadata_cache:
        return _video_metadata_cache[cache_key].copy()
    
    # Load from persistent cache if available
    if use_cache and cache_file and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                persistent_cache = json.load(f)
                if cache_key in persistent_cache:
                    metadata = persistent_cache[cache_key]
                    _video_metadata_cache[cache_key] = metadata
                    return metadata.copy()
        except Exception as e:
            logger.debug(f"Failed to load persistent cache: {e}")
    
    # Compute metadata (only once per video)
    container = None
    try:
        container = av.open(str(video_path))
        if len(container.streams.video) == 0:
            logger.warning(f"Video has no video streams: {video_path}")
            return {
                'total_frames': 0,
                'fps': 30.0,
                'width': 0,
                'height': 0,
                'duration': 0.0
            }
        stream = container.streams.video[0]
        
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        width = stream.width if stream.width else 0
        height = stream.height if stream.height else 0
        
        # Get duration
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            try:
                duration = float(stream.duration * stream.time_base)
            except Exception:
                pass
        
        # Count frames manually (only when cache miss)
        MAX_REASONABLE_FRAMES = 10_000_000
        frame_count = 0
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_count += 1
                if frame_count > MAX_REASONABLE_FRAMES:
                    logger.warning(f"Frame count exceeds {MAX_REASONABLE_FRAMES} for {video_path}")
                    break
            if frame_count > MAX_REASONABLE_FRAMES:
                break
        
        metadata = {
            'total_frames': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration
        }
        
        # Cache result
        if use_cache:
            _video_metadata_cache[cache_key] = metadata
            
            # Save to persistent cache
            if cache_file:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    persistent_cache = {}
                    if cache_file.exists():
                        with open(cache_file, 'r') as f:
                            persistent_cache = json.load(f)
                    persistent_cache[cache_key] = metadata
                    with open(cache_file, 'w') as f:
                        json.dump(persistent_cache, f, indent=2)
                except Exception as e:
                    logger.debug(f"Failed to save persistent cache: {e}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata for {video_path}: {e}")
        return {
            'total_frames': 0,
            'fps': 30.0,
            'width': 0,
            'height': 0,
            'duration': 0.0
        }
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def clear_cache(cache_file: Optional[Path] = None):
    """
    Clear in-memory cache and optionally persistent cache file.
    
    Args:
        cache_file: Optional path to persistent cache file to delete
    """
    _video_metadata_cache.clear()
    
    # Clear persistent cache file if provided
    if cache_file and cache_file.exists():
        try:
            cache_file.unlink()
            logger.info(f"Cleared persistent cache file: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to delete persistent cache file {cache_file}: {e}")


__all__ = [
    "get_video_metadata",
    "get_video_metadata_hash",
    "clear_cache",
]

