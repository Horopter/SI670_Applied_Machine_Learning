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
        
        # Always count frames manually for reliability (metadata can be corrupted)
        MAX_REASONABLE_FRAMES = 10_000_000
        
        # Get metadata for sanity check (but don't trust it)
        stream_frames_metadata = stream.frames if stream.frames > 0 else 0
        duration_frames_metadata = 0
        if stream.duration is not None and stream.time_base is not None:
            try:
                duration_seconds = float(stream.duration * stream.time_base)
                duration_frames_metadata = int(duration_seconds * fps)
            except Exception:
                pass
        
        # Count frames manually for accurate count
        logger.debug(f"Counting frames manually for reliable frame count...")
        frame_count_manual = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_count_manual += 1
                if frame_count_manual > MAX_REASONABLE_FRAMES:
                    logger.warning(f"Frame count exceeds {MAX_REASONABLE_FRAMES}, stopping count")
                    break
            if frame_count_manual > MAX_REASONABLE_FRAMES:
                break
        
        total_frames = frame_count_manual
        
        # Sanity check: compare with metadata
        if (stream_frames_metadata > 0 or duration_frames_metadata > 0) and total_frames > 0:
            metadata_count = stream_frames_metadata if stream_frames_metadata > 0 else duration_frames_metadata
            if metadata_count > 0:
                ratio = max(total_frames, metadata_count) / min(total_frames, metadata_count) if min(total_frames, metadata_count) > 0 else 1.0
                if ratio > 1.5:
                    logger.debug(
                        f"Metadata frame count ({metadata_count}) differs from manual count ({total_frames}), "
                        f"using manual count"
                    )
        
        # Reopen container for actual frame loading
        container.close()
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # Early return if start_frame exceeds video length
        if start_frame >= total_frames:
            logger.debug(f"start_frame ({start_frame}) >= total_frames ({total_frames}), returning empty")
            return [], fps
        
        end_frame = start_frame + max_frames if max_frames else total_frames
        if end_frame > total_frames:
            end_frame = total_frames
        
        frames_to_load = end_frame - start_frame
        
        # Safety check
        if frames_to_load <= 0:
            logger.debug(f"frames_to_load ({frames_to_load}) <= 0, returning empty")
            return [], fps
        
        # Decode frames from the start and skip until we reach start_frame
        # Note: We don't use seeking here because it's unreliable for frame-accurate positioning
        # Decoding from start and skipping is slower but guarantees correctness
        frame_count = 0
        current_frame_idx = 0
        
        for packet in container.demux(stream):
            if frame_count >= frames_to_load:
                break
            for frame in packet.decode():
                # Skip frames until we reach start_frame
                if current_frame_idx < start_frame:
                    current_frame_idx += 1
                    continue
                
                # We've reached start_frame, now load frames
                if frame_count >= frames_to_load:
                    break
                frame_array = frame.to_ndarray(format='rgb24')
                frames.append(frame_array)
                frame_count += 1
                current_frame_idx += 1
            if frame_count >= frames_to_load:
                break
        
        if len(frames) > 0:
            logger.debug(f"Loaded {len(frames)} frames from {Path(video_path).name} (frames {start_frame}-{start_frame+len(frames)-1})")
        else:
            logger.warning(
                f"Loaded 0 frames from {Path(video_path).name} "
                f"(start_frame={start_frame}, total_frames={total_frames}, "
                f"frames_to_load={frames_to_load}, current_frame_idx={current_frame_idx})"
            )
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
        # Aggressively free CPU/GPU memory after writing a video
        aggressive_gc(clear_cuda=False)


def concatenate_videos(
    video_paths: List[str],
    output_path: str,
    fps: float = 30.0
) -> bool:
    """
    Concatenate multiple video files into a single video.
    
    Args:
        video_paths: List of paths to video files to concatenate
        output_path: Output video path
        fps: Frames per second (should match input videos)
    
    Returns:
        True if successful, False otherwise
    """
    if not video_paths:
        logger.warning("No video files to concatenate")
        return False
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get video properties from first video
    first_video = av.open(video_paths[0])
    first_stream = first_video.streams.video[0]
    width = first_stream.width
    height = first_stream.height
    first_video.close()
    
    # Create output container
    output_container = None
    try:
        output_container = av.open(str(output_path), mode='w')
        fps_fraction = Fraction(int(fps * 1000), 1000)
        output_stream = output_container.add_stream('libx264', rate=fps_fraction)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuv420p'
        
        # Process each input video
        # Decode and re-encode frames one video at a time to maintain proper timestamps
        # This ensures monotonically increasing DTS/PTS across all chunks
        for video_path in video_paths:
            if not Path(video_path).exists():
                logger.warning(f"Intermediate video not found: {video_path}, skipping")
                continue
            
            input_container = av.open(video_path)
            input_stream = input_container.streams.video[0]
            
            # Decode frames to numpy arrays and immediately re-encode
            # This ensures proper timestamp continuity across chunks
            for frame in input_container.decode(video=0):
                # Convert to numpy array
                frame_array = frame.to_ndarray(format='rgb24')
                # Create new frame from array (this resets timestamps)
                new_frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                # Let encoder assign sequential timestamps automatically
                for packet in output_stream.encode(new_frame):
                    output_container.mux(packet)
            
            input_container.close()
        
        # Flush encoder
        for packet in output_stream.encode():
            output_container.mux(packet)
        
        return True
    except Exception as e:
        logger.error(f"Failed to concatenate videos: {e}")
        return False
    finally:
        if output_container is not None:
            try:
                output_container.close()
            except Exception:
                pass
        # Aggressively free CPU/GPU memory after concatenation
        aggressive_gc(clear_cuda=False)

