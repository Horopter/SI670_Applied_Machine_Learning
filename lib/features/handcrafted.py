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
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Check if ffprobe is available (check once at module load)
_FFPROBE_AVAILABLE = None

def _check_ffprobe_available() -> bool:
    """Check if ffprobe is available in PATH."""
    global _FFPROBE_AVAILABLE
    if _FFPROBE_AVAILABLE is None:
        _FFPROBE_AVAILABLE = shutil.which('ffprobe') is not None
        if not _FFPROBE_AVAILABLE:
            logger.debug("ffprobe not found in PATH - codec cue extraction will be disabled")
    return _FFPROBE_AVAILABLE


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
    
    If ffprobe is not available, returns default values without logging warnings.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary of codec features
    """
    # Check if ffprobe is available - if not, return defaults silently
    if not _check_ffprobe_available():
        return {
            "codec_bitrate": 0.0,
            "codec_fps": 30.0,
            "codec_resolution": 0.0,
        }
    
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
            # Only log at debug level - ffprobe might fail for some videos
            logger.debug(f"ffprobe failed for {video_path} (returncode: {result.returncode})")
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
    except FileNotFoundError:
        # ffprobe not found - this should not happen if check passed, but handle gracefully
        # Only log once at debug level
        logger.debug("ffprobe command not found (should have been caught by availability check)")
        return {
            "codec_bitrate": 0.0,
            "codec_fps": 30.0,
            "codec_resolution": 0.0,
        }
    except Exception as e:
        # Other errors (timeout, permission, etc.) - log at debug level only
        logger.debug(f"Failed to extract codec cues from {video_path}: {e}")
        return {
            "codec_bitrate": 0.0,
            "codec_fps": 30.0,
            "codec_resolution": 0.0,
        }


# Define fixed feature order to ensure consistent array shapes
_FEATURE_ORDER = [
    # Noise residual (3 features)
    "noise_energy", "noise_mean", "noise_std",
    # DCT statistics (varies by implementation, but we'll ensure consistency)
    # Blur/sharpness (varies)
    # Boundary inconsistency (1 feature)
    "boundary_inconsistency",
    # Codec cues (3 features)
    "codec_bitrate", "codec_fps", "codec_resolution",
]

def extract_all_features(frame: np.ndarray, video_path: Optional[str] = None) -> Dict[str, float]:
    """
    Extract all handcrafted features from a frame.
    
    Always returns the same set of features in a consistent order,
    even if codec cues cannot be extracted (returns defaults).
    
    Args:
        frame: Input frame (H, W, 3) or (H, W)
        video_path: Optional video path for codec cues
    
    Returns:
        Dictionary of all features (always includes all feature types)
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
    
    # Codec cues - always include (with defaults if video_path is None or extraction fails)
    if video_path:
        codec_features = extract_codec_cues(video_path)
    else:
        # Return defaults if no video path provided
        codec_features = {
            "codec_bitrate": 0.0,
            "codec_fps": 30.0,
            "codec_resolution": 0.0,
        }
    features.update(codec_features)
    
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
                video_path_resolved = resolve_video_path(video_rel, project_root)
                # Ensure we have a Path object (resolve_video_path returns str)
                if isinstance(video_path_resolved, Path):
                    video_path = video_path_resolved
                else:
                    video_path = Path(video_path_resolved)
                
                # Check if video file exists and is readable
                if not video_path.exists():
                    logger.debug(f"Video file not found: {video_path}, using zero features")
                    feature_dict = extract_all_features(
                        np.zeros((224, 224, 3), dtype=np.uint8),
                        None
                    )
                    feature_values = list(feature_dict.values())
                    feature_array = np.array(feature_values, dtype=np.float32)
                    all_features.append(feature_array)
                    continue
                
                # Load a few frames from video with robust error handling
                container = None
                try:
                    container = av.open(str(video_path))
                    if not container.streams.video:
                        raise ValueError(f"No video stream found in {video_path}")
                    
                    stream = container.streams.video[0]
                    frames = []
                    frame_count = 0
                    max_frames = min(self.num_frames, 8)  # Sample up to 8 frames
                    
                    for packet in container.demux(stream):
                        if frame_count >= max_frames:
                            break
                        try:
                            for frame in packet.decode():
                                if frame_count >= max_frames:
                                    break
                                frame_array = frame.to_ndarray(format='rgb24')
                                frames.append(frame_array)
                                frame_count += 1
                        except Exception as decode_error:
                            # Skip corrupted packets
                            logger.debug(f"Skipping corrupted packet in {video_rel}: {decode_error}")
                            continue
                        if frame_count >= max_frames:
                            break
                    
                    if container:
                        container.close()
                        container = None
                    
                    if not frames:
                        # Return zero features if no frames decoded
                        logger.debug(f"No frames decoded from {video_rel}, using zero features")
                        feature_dict = extract_all_features(
                            np.zeros((224, 224, 3), dtype=np.uint8),
                            str(video_path) if self.include_codec else None
                        )
                    else:
                        # Extract features from first frame (or average across frames)
                        feature_dict = extract_all_features(
                            frames[0],
                            str(video_path) if self.include_codec else None
                        )
                    
                    # Convert to array - ensure consistent ordering
                    # Get all feature values in a consistent order
                    feature_values = list(feature_dict.values())
                    feature_array = np.array(feature_values, dtype=np.float32)
                    all_features.append(feature_array)
                    
                except (av.AVError, ValueError, OSError, IOError) as video_error:
                    # Handle video corruption, missing moov atom, etc.
                    error_msg = str(video_error).lower()
                    if 'moov atom' in error_msg or 'invalid data' in error_msg or 'corrupt' in error_msg:
                        logger.debug(f"Corrupted video file {video_rel}: {video_error}, using zero features")
                    else:
                        logger.debug(f"Failed to decode video {video_rel}: {video_error}, using zero features")
                    
                    if container:
                        try:
                            container.close()
                        except Exception:
                            pass
                    
                    # Return zero features on error
                    feature_dict = extract_all_features(
                        np.zeros((224, 224, 3), dtype=np.uint8),
                        None
                    )
                    feature_values = list(feature_dict.values())
                    feature_array = np.array(feature_values, dtype=np.float32)
                    all_features.append(feature_array)
                
            except Exception as e:
                # Catch-all for any other errors
                logger.debug(f"Unexpected error processing {video_rel}: {e}, using zero features")
                # Return zero features on error
                feature_dict = extract_all_features(
                    np.zeros((224, 224, 3), dtype=np.uint8),
                    None
                )
                feature_values = list(feature_dict.values())
                feature_array = np.array(feature_values, dtype=np.float32)
                all_features.append(feature_array)
        
        # Convert to numpy array - ensure all arrays have the same shape
        if not all_features:
            # Return empty array with correct shape
            return np.array([]).reshape(0, 0)
        
        # Check if all feature arrays have the same length
        feature_lengths = [len(f) for f in all_features]
        if len(set(feature_lengths)) > 1:
            # Inconsistent feature lengths - pad or truncate to match the most common length
            from collections import Counter
            most_common_length = Counter(feature_lengths).most_common(1)[0][0]
            logger.warning(
                f"Inconsistent feature lengths detected: {set(feature_lengths)}. "
                f"Padding/truncating to {most_common_length} features."
            )
            normalized_features = []
            for feat in all_features:
                if len(feat) < most_common_length:
                    # Pad with zeros
                    padded = np.pad(feat, (0, most_common_length - len(feat)), mode='constant', constant_values=0.0)
                    normalized_features.append(padded)
                elif len(feat) > most_common_length:
                    # Truncate
                    normalized_features.append(feat[:most_common_length])
                else:
                    normalized_features.append(feat)
            all_features = normalized_features
        
        return np.array(all_features, dtype=np.float32)
    
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
        
        # Check if video file exists
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            logger.debug(f"Video file not found: {video_path}, using zero features")
            feature_dict = extract_all_features(
                np.zeros((224, 224, 3), dtype=np.uint8),
                None
            )
            feature_values = list(feature_dict.values())
            return np.array(feature_values, dtype=np.float32)
        
        container = None
        try:
            container = av.open(str(video_path))
            if not container.streams.video:
                raise ValueError(f"No video stream found in {video_path}")
            
            stream = container.streams.video[0]
            frames = []
            frame_count = 0
            
            for packet in container.demux(stream):
                if frame_count >= num_frames:
                    break
                try:
                    for frame in packet.decode():
                        if frame_count >= num_frames:
                            break
                        frame_array = frame.to_ndarray(format='rgb24')
                        frames.append(frame_array)
                        frame_count += 1
                except Exception as decode_error:
                    # Skip corrupted packets
                    logger.debug(f"Skipping corrupted packet in {video_path}: {decode_error}")
                    continue
                if frame_count >= num_frames:
                    break
            
            if container:
                container.close()
                container = None
            
            if not frames:
                logger.debug(f"No frames decoded from {video_path}, using zero features")
                feature_dict = extract_all_features(
                    np.zeros((224, 224, 3), dtype=np.uint8),
                    str(video_path) if self.include_codec else None
                )
            else:
                feature_dict = extract_all_features(
                    frames[0],
                    str(video_path) if self.include_codec else None
                )
            
            feature_values = list(feature_dict.values())
            return np.array(feature_values, dtype=np.float32)
            
        except (av.AVError, ValueError, OSError, IOError) as video_error:
            # Handle video corruption, missing moov atom, etc.
            error_msg = str(video_error).lower()
            if 'moov atom' in error_msg or 'invalid data' in error_msg or 'corrupt' in error_msg:
                logger.debug(f"Corrupted video file {video_path}: {video_error}, using zero features")
            else:
                logger.debug(f"Failed to decode video {video_path}: {video_error}, using zero features")
            
            if container:
                try:
                    container.close()
                except Exception:
                    pass
            
            feature_dict = extract_all_features(
                np.zeros((224, 224, 3), dtype=np.uint8),
                None
            )
            feature_values = list(feature_dict.values())
            return np.array(feature_values, dtype=np.float32)
            
        except Exception as e:
            # Catch-all for any other errors
            logger.debug(f"Unexpected error processing {video_path}: {e}, using zero features")
            if container:
                try:
                    container.close()
                except Exception:
                    pass
            feature_dict = extract_all_features(
                np.zeros((224, 224, 3), dtype=np.uint8),
                None
            )
            feature_values = list(feature_dict.values())
            return np.array(feature_values, dtype=np.float32)

