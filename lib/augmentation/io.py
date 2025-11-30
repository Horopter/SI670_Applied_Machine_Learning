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
    max_frames: Optional[int] = 1000,
    start_frame: int = 0
) -> tuple[List[np.ndarray], float]:
    """
    Load frames from a video, preserving original resolution.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to load (default: 1000 to prevent OOM)
        start_frame: Starting frame index (default: 0)
    
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
        end_frame = start_frame + max_frames if max_frames else total_frames
        
        if total_frames > 0 and end_frame > total_frames:
            end_frame = total_frames
        
        frame_count = 0
        frames_to_skip = start_frame if start_frame > 0 else 0
        frames_to_load = end_frame - start_frame
        
        # Try to seek to approximate position if start_frame > 0
        if start_frame > 0:
            try:
                # Calculate approximate timestamp
                timestamp = start_frame / fps
                # Seek to timestamp (in seconds)
                container.seek(int(timestamp * 1000000))  # av.seek expects microseconds
            except Exception as e:
                logger.debug(f"Could not seek to frame {start_frame}: {e}, will skip frames manually")
        
        for packet in container.demux(stream):
            if frame_count >= frames_to_load:
                break
            for frame in packet.decode():
                # Skip frames until we reach start_frame
                if frames_to_skip > 0:
                    frames_to_skip -= 1
                    continue
                
                if frame_count >= frames_to_load:
                    break
                frame_array = frame.to_ndarray(format='rgb24')
                frames.append(frame_array)
                frame_count += 1
            if frame_count >= frames_to_load:
                break
        
        logger.debug(f"Loaded {len(frames)} frames from {Path(video_path).name} (frames {start_frame}-{start_frame+len(frames)-1})")
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

