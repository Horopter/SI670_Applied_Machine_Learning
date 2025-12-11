"""
Optical flow extraction utilities for video processing.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def extract_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    method: str = "farneback"
) -> np.ndarray:
    """
    Extract optical flow between two frames.
    
    Args:
        frame1: First frame (H, W, 3) RGB or (H, W) grayscale
        frame2: Second frame (H, W, 3) RGB or (H, W) grayscale
        method: Flow method - "farneback" or "lucas_kanade"
    
    Returns:
        Optical flow (H, W, 2) - flow vectors (x, y) for each pixel
    """
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = frame1
        gray2 = frame2
    
    if method == "farneback":
        # Dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    elif method == "lucas_kanade":
        # Sparse optical flow using Lucas-Kanade method
        # For dense flow, we'll use corner detection + tracking
        corners = cv2.goodFeaturesToTrack(
            gray1,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )
        
        if corners is not None and len(corners) > 0:
            # Track corners
            next_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                gray1, gray2,
                corners, None
            )
            
            # Create dense flow field (simplified: interpolate from sparse points)
            h, w = gray1.shape
            flow = np.zeros((h, w, 2), dtype=np.float32)
            
            # Interpolate flow from tracked corners
            for i, (corner, next_corner, st) in enumerate(zip(corners, next_corners, status)):
                if st[0] == 1:  # Successfully tracked
                    x, y = int(corner[0][0]), int(corner[0][1])
                    dx = next_corner[0][0] - corner[0][0]
                    dy = next_corner[0][1] - corner[0][1]
                    
                    if 0 <= x < w and 0 <= y < h:
                        flow[y, x, 0] = dx
                        flow[y, x, 1] = dy
            
            # Fill in missing values with nearest neighbor or zero
            # For simplicity, we'll use a simple Gaussian blur from OpenCV
            flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (5, 5), 0)
            flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (5, 5), 0)
        else:
            # No corners found, return zero flow
            h, w = gray1.shape
            flow = np.zeros((h, w, 2), dtype=np.float32)
    else:
        raise ValueError(f"Unknown optical flow method: {method}")
    
    return flow


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to RGB visualization.
    
    Args:
        flow: Optical flow (H, W, 2) - flow vectors
    
    Returns:
        RGB visualization (H, W, 3) uint8
    """
    # Compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize magnitude to [0, 1]
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: full
    hsv[..., 2] = np.clip(magnitude * 255, 0, 255)  # Value: magnitude
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def extract_optical_flow_sequence(
    frames: List[np.ndarray],
    method: str = "farneback"
) -> List[np.ndarray]:
    """
    Extract optical flow for a sequence of frames.
    
    Args:
        frames: List of frames (each is H, W, 3 or H, W)
        method: Flow method
    
    Returns:
        List of optical flow arrays (H, W, 2) between consecutive frames
    """
    flows = []
    
    for i in range(len(frames) - 1):
        flow = extract_optical_flow(frames[i], frames[i + 1], method=method)
        flows.append(flow)
    
    return flows


def extract_optical_flow_video(
    video_path: str,
    num_frames: int = 8,
    method: str = "farneback",
    frame_indices: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Extract optical flow from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (if frame_indices not provided)
        method: Flow method
        frame_indices: Specific frame indices to extract (optional)
    
    Returns:
        List of optical flow arrays (H, W, 2)
    """
    import av
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    frames = []
    frame_count = 0
    
    # Collect frames
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame_indices is not None:
                if frame_count in frame_indices:
                    frame_array = frame.to_ndarray(format='rgb24')
                    frames.append(frame_array)
            else:
                if frame_count < num_frames:
                    frame_array = frame.to_ndarray(format='rgb24')
                    frames.append(frame_array)
            
            frame_count += 1
            
            if frame_indices is not None:
                if len(frames) >= len(frame_indices):
                    break
            else:
                if len(frames) >= num_frames:
                    break
        
        if frame_indices is not None:
            if len(frames) >= len(frame_indices):
                break
        else:
            if len(frames) >= num_frames:
                break
    
    container.close()
    
    if len(frames) < 2:
        logger.warning(f"Not enough frames in {video_path} for optical flow")
        # Return zero flow
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            return [np.zeros((h, w, 2), dtype=np.float32)]
        else:
            return []
    
    # Extract optical flow between consecutive frames
    flows = extract_optical_flow_sequence(frames, method=method)
    
    return flows


__all__ = [
    "extract_optical_flow",
    "flow_to_rgb",
    "extract_optical_flow_sequence",
    "extract_optical_flow_video"
]

