"""
Handcrafted feature extractors.

Provides feature extraction functions for:
- Noise residual energy
- DCT band statistics
- Blur/sharpness metrics
- Block boundary inconsistency
- Codec cues
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def extract_noise_residual(frame: np.ndarray) -> Dict[str, float]:
    """
    Extract noise residual energy features.
    
    Args:
        frame: Input frame (H, W, 3) or (H, W)
    
    Returns:
        Dictionary of noise residual features
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Convert to float for processing
    gray = gray.astype(np.float32)
    
    # High-pass filter to extract noise
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)
    noise_residual = cv2.filter2D(gray, -1, kernel)
    
    # Compute statistics
    energy = np.sum(noise_residual ** 2)
    mean_energy = np.mean(noise_residual ** 2)
    std_energy = np.std(noise_residual ** 2)
    
    return {
        "noise_energy": float(energy),
        "noise_mean": float(mean_energy),
        "noise_std": float(std_energy),
    }


def extract_dct_statistics(frame: np.ndarray, block_size: int = 8) -> Dict[str, float]:
    """
    Extract DCT band statistics.
    
    Args:
        frame: Input frame (H, W, 3) or (H, W)
        block_size: DCT block size (default: 8)
    
    Returns:
        Dictionary of DCT statistics
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    gray = gray.astype(np.float32)
    
    # Pad to multiple of block_size
    h, w = gray.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
    
    # Compute DCT for each block
    dct_coeffs = []
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            block = gray[i:i+block_size, j:j+block_size]
            dct_block = cv2.dct(block)
            dct_coeffs.append(dct_block)
    
    dct_coeffs = np.array(dct_coeffs)
    
    # Extract DC and AC coefficients
    dc_coeffs = dct_coeffs[:, 0, 0]
    ac_coeffs = dct_coeffs[:, :, :].copy()
    ac_coeffs[:, 0, 0] = 0  # Remove DC component
    
    return {
        "dct_dc_mean": float(np.mean(dc_coeffs)),
        "dct_dc_std": float(np.std(dc_coeffs)),
        "dct_ac_mean": float(np.mean(ac_coeffs)),
        "dct_ac_std": float(np.std(ac_coeffs)),
        "dct_ac_energy": float(np.sum(ac_coeffs ** 2)),
    }


def extract_blur_sharpness(frame: np.ndarray) -> Dict[str, float]:
    """
    Extract blur and sharpness metrics.
    
    Args:
        frame: Input frame (H, W, 3) or (H, W)
    
    Returns:
        Dictionary of blur/sharpness features
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Convert to uint8 for OpenCV operations to avoid dtype mismatch with CV_64F
    if gray.dtype != np.uint8:
        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    else:
        gray_uint8 = gray
    
    # Laplacian variance (sharpness metric)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    laplacian_var = float(np.var(laplacian))
    
    # Sobel gradient magnitude
    sobelx = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = float(np.mean(gradient_magnitude))
    gradient_std = float(np.std(gradient_magnitude))
    
    return {
        "laplacian_var": laplacian_var,
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
    }


def extract_boundary_inconsistency(frame: np.ndarray, block_size: int = 8) -> float:
    """
    Extract block boundary inconsistency metric.
    
    Args:
        frame: Input frame (H, W, 3) or (H, W)
        block_size: Block size for boundary detection (default: 8)
    
    Returns:
        Boundary inconsistency score
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    # Compute boundary inconsistencies
    inconsistencies = []
    
    # Horizontal boundaries
    for i in range(block_size, h, block_size):
        if i < h:
            boundary = gray[i, :]
            inconsistency = np.std(boundary)
            inconsistencies.append(inconsistency)
    
    # Vertical boundaries
    for j in range(block_size, w, block_size):
        if j < w:
            boundary = gray[:, j]
            inconsistency = np.std(boundary)
            inconsistencies.append(inconsistency)
    
    return float(np.mean(inconsistencies)) if inconsistencies else 0.0


def extract_codec_cues(video_path: str) -> Dict[str, float]:
    """
    Extract codec-related cues using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary of codec features
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name,bit_rate,width,height,r_frame_rate',
             '-of', 'json', str(video_path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            logger.warning(f"ffprobe failed for {video_path}")
            return {
                "codec_bitrate": 0.0,
                "codec_fps": 30.0,
                "codec_resolution": 0.0,
            }
        
        import json
        data = json.loads(result.stdout)
        
        if 'streams' not in data or len(data['streams']) == 0:
            return {
                "codec_bitrate": 0.0,
                "codec_fps": 30.0,
                "codec_resolution": 0.0,
            }
        
        stream = data['streams'][0]
        bit_rate = float(stream.get('bit_rate', 0))
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        resolution = width * height
        
        # Parse frame rate
        fps_str = stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 30.0
        else:
            fps = float(fps_str) if fps_str else 30.0
        
        return {
            "codec_bitrate": bit_rate,
            "codec_fps": fps,
            "codec_resolution": float(resolution),
        }
    except Exception as e:
        logger.warning(f"Failed to extract codec cues from {video_path}: {e}")
        return {
            "codec_bitrate": 0.0,
            "codec_fps": 30.0,
            "codec_resolution": 0.0,
        }


def extract_all_features(frame: np.ndarray, video_path: Optional[str] = None) -> Dict[str, float]:
    """
    Extract all handcrafted features from a frame.
    
    Args:
        frame: Input frame (H, W, 3) or (H, W)
        video_path: Optional video path for codec cues
    
    Returns:
        Dictionary of all features
    """
    features = {}
    
    # Noise residual
    features.update(extract_noise_residual(frame))
    
    # DCT statistics
    features.update(extract_dct_statistics(frame))
    
    # Blur/sharpness
    features.update(extract_blur_sharpness(frame))
    
    # Boundary inconsistency
    features["boundary_inconsistency"] = extract_boundary_inconsistency(frame)
    
    # Codec cues (if video path provided)
    if video_path:
        features.update(extract_codec_cues(video_path))
    
    return features


class HandcraftedFeatureExtractor:
    """
    Feature extractor that caches results to disk.
    Compatible with baselines.py interface.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, num_frames: int = 8, include_codec: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            cache_dir: Directory to cache extracted features (unused, kept for compatibility)
            num_frames: Number of frames to sample (unused, kept for compatibility)
            include_codec: Whether to include codec cues
        """
        self.cache_dir = cache_dir
        self.num_frames = num_frames
        self.include_codec = include_codec
    
    def extract(self, frame: np.ndarray, video_path: Optional[str] = None) -> Dict[str, float]:
        """Extract all features from a frame."""
        return extract_all_features(frame, video_path if self.include_codec else None)
    
    def extract_batch(
        self,
        video_paths: List[str],
        project_root: str,
        batch_size: int = 1
    ) -> np.ndarray:
        """
        Extract features from a batch of videos.
        
        Args:
            video_paths: List of video paths
            project_root: Project root directory
            batch_size: Batch size for processing (unused, kept for compatibility)
        
        Returns:
            Feature matrix (N, M) where N is number of videos, M is number of features
        """
        import av
        from lib.utils.paths import resolve_video_path
        
        all_features = []
        
        for video_rel in video_paths:
            try:
                video_path = resolve_video_path(video_rel, project_root)
                
                # Load a few frames from video
                container = av.open(video_path)
                stream = container.streams.video[0]
                frames = []
                frame_count = 0
                max_frames = min(self.num_frames, 8)  # Sample up to 8 frames
                
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
                
                container.close()
                
                if not frames:
                    # Return zero features if no frames
                    feature_dict = extract_all_features(
                        np.zeros((224, 224, 3), dtype=np.uint8),
                        video_path if self.include_codec else None
                    )
                else:
                    # Extract features from first frame (or average across frames)
                    feature_dict = extract_all_features(
                        frames[0],
                        video_path if self.include_codec else None
                    )
                
                # Convert to array
                feature_array = np.array(list(feature_dict.values()))
                all_features.append(feature_array)
                
            except Exception as e:
                logger.warning(f"Failed to extract features from {video_rel}: {e}")
                # Return zero features on error
                feature_dict = extract_all_features(
                    np.zeros((224, 224, 3), dtype=np.uint8),
                    None
                )
                feature_array = np.array(list(feature_dict.values()))
                all_features.append(feature_array)
        
        return np.array(all_features)
    
    def extract_from_video(
        self,
        video_path: str,
        num_frames: int = 8,
        project_root: str = None
    ) -> np.ndarray:
        """
        Extract features from a single video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            project_root: Project root for path resolution
        
        Returns:
            Feature vector
        """
        from lib.utils.paths import resolve_video_path
        import av
        
        if project_root:
            video_path = resolve_video_path(video_path, project_root)
        
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            frames = []
            frame_count = 0
            
            for packet in container.demux(stream):
                if frame_count >= num_frames:
                    break
                for frame in packet.decode():
                    if frame_count >= num_frames:
                        break
                    frame_array = frame.to_ndarray(format='rgb24')
                    frames.append(frame_array)
                    frame_count += 1
                if frame_count >= num_frames:
                    break
            
            container.close()
            
            if not frames:
                feature_dict = extract_all_features(
                    np.zeros((224, 224, 3), dtype=np.uint8),
                    video_path if self.include_codec else None
                )
            else:
                feature_dict = extract_all_features(
                    frames[0],
                    video_path if self.include_codec else None
                )
            
            return np.array(list(feature_dict.values()))
            
        except Exception as e:
            logger.warning(f"Failed to extract features from {video_path}: {e}")
            feature_dict = extract_all_features(
                np.zeros((224, 224, 3), dtype=np.uint8),
                None
            )
            return np.array(list(feature_dict.values()))

