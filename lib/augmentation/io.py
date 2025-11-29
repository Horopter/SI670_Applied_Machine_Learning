"""
Video I/O utilities for augmentation.

Provides efficient frame loading and saving with memory management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
from fractions import Fraction
import numpy as np
import av

from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


def load_frames(
    video_path: str, 
    max_frames: Optional[int] = 1000
) -> tuple[List[np.ndarray], float]:
    """
    Load frames from a video, preserving original resolution.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to load (default: 1000 to prevent OOM)
    
    Returns:
        Tuple of (frames list, fps)
    """
    frames = []
    fps = 30.0
    container = None
    
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        
        total_frames = stream.frames if stream.frames > 0 else 0
        if total_frames > max_frames:
            logger.warning(
                f"Video has {total_frames} frames, loading only {max_frames} to prevent OOM"
            )
        
        frame_count = 0
        for packet in container.demux(stream):
            if frame_count >= max_frames:
                break
            for frame in packet.decode():
                if frame_count >= max_frames:
                    break
                frame_array = frame.to_ndarray(format='rgb24')
                frames.append(frame_array)
                frame_count += 1
            if frame_count >= max_frames:
                break
        
        logger.debug(f"Loaded {len(frames)} frames from {Path(video_path).name}")
        return frames, fps
    except Exception as e:
        logger.error(f"Failed to load video {video_path}: {e}")
        return [], 30.0
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        aggressive_gc(clear_cuda=False)


def save_frames(
    frames: List[np.ndarray], 
    output_path: str, 
    fps: float = 30.0
) -> bool:
    """
    Save frames as a video file, preserving original resolution.
    
    Args:
        frames: List of frame arrays (H, W, 3)
        output_path: Output video path
        fps: Frames per second
    
    Returns:
        True if successful, False otherwise
    """
    if not frames:
        logger.warning(f"No frames to save for {output_path}")
        return False
    
    container = None
    try:
        height, width = frames[0].shape[:2]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        container = av.open(str(output_path), mode='w')
        # Convert float FPS to fraction (PyAV requires fraction)
        fps_fraction = Fraction(int(fps * 1000), 1000)
        stream = container.add_stream('libx264', rate=fps_fraction)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        
        for packet in stream.encode():
            container.mux(packet)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save video {output_path}: {e}")
        return False
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

