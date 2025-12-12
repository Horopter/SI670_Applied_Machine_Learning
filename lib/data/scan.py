"""Scan video folders and extract video information"""
import os
import json
import subprocess
import logging
from typing import List, Dict, Optional
import cv2
from tqdm import tqdm
from .config import FVCConfig

logger = logging.getLogger(__name__)

VIDEO_FILENAME_CANDIDATES = ["video.mp4", "video.mkv", "video.webm"]


def find_video_file(folder: str) -> Optional[str]:
    """Find video file in a folder"""
    if not os.path.isdir(folder):
        return None
    
    for fname in os.listdir(folder):
        if fname.lower() in VIDEO_FILENAME_CANDIDATES:
            return os.path.join(folder, fname)
    return None


def _probe_video_stats_ffprobe(path: str) -> Dict:
    """Get comprehensive video stats using ffprobe."""
    stats = {
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_sec": None,
        "codec_name": None,
        "codec_long_name": None,
        "bitrate": None,
        "file_size_bytes": None,
        "pixel_format": None,
        "aspect_ratio": None,
        "total_bitrate": None,
    }
    
    try:
        # Get file size
        if os.path.exists(path):
            stats["file_size_bytes"] = os.path.getsize(path)
        
        # Use ffprobe to get detailed video information
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path
        ]
        
        # Python 3.6 compatible: use stdout/stderr instead of capture_output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        # Decode bytes to string
        result.stdout = result.stdout.decode('utf-8') if result.stdout else ""
        result.stderr = result.stderr.decode('utf-8') if result.stderr else ""
        if result.returncode != 0:
            return stats
        
        data = json.loads(result.stdout)
        
        # Extract format info
        if "format" in data:
            format_info = data["format"]
            if "duration" in format_info:
                try:
                    stats["duration_sec"] = float(format_info["duration"])
                except (ValueError, TypeError):
                    pass
            if "bit_rate" in format_info:
                try:
                    stats["total_bitrate"] = int(format_info["bit_rate"])
                except (ValueError, TypeError):
                    pass
            if "size" in format_info:
                try:
                    stats["file_size_bytes"] = int(format_info["size"])
                except (ValueError, TypeError):
                    pass
        
        # Extract video stream info
        if "streams" in data:
            for stream in data["streams"]:
                if stream.get("codec_type") == "video":
                    if "width" in stream:
                        stats["width"] = int(stream["width"])
                    if "height" in stream:
                        stats["height"] = int(stream["height"])
                    if "r_frame_rate" in stream:
                        # Parse frame rate (e.g., "30/1" or "29.97/1")
                        try:
                            num, den = map(int, stream["r_frame_rate"].split("/"))
                            if den > 0:
                                stats["fps"] = num / den
                        except (ValueError, ZeroDivisionError):
                            pass
                    if "nb_frames" in stream:
                        try:
                            stats["frame_count"] = int(stream["nb_frames"])
                        except (ValueError, TypeError):
                            pass
                    if "codec_name" in stream:
                        stats["codec_name"] = stream["codec_name"]
                    if "codec_long_name" in stream:
                        stats["codec_long_name"] = stream["codec_long_name"]
                    if "bit_rate" in stream:
                        try:
                            stats["bitrate"] = int(stream["bit_rate"])
                        except (ValueError, TypeError):
                            pass
                    if "pix_fmt" in stream:
                        stats["pixel_format"] = stream["pix_fmt"]
                    if "display_aspect_ratio" in stream:
                        stats["aspect_ratio"] = stream["display_aspect_ratio"]
                    break  # Use first video stream
        
        # Calculate frame count from duration and fps if not available
        if stats["frame_count"] is None and stats["duration_sec"] and stats["fps"]:
            stats["frame_count"] = int(stats["duration_sec"] * stats["fps"])
        
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, Exception):
        # Fallback to OpenCV if ffprobe fails
        pass
    
    return stats


def _probe_video_stats_opencv(path: str) -> Dict:
    """Fallback video stats using OpenCV."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {
                "width": None,
                "height": None,
                "fps": None,
                "frame_count": None,
                "duration_sec": None
            }
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        duration = None
        if fps and fps > 0:
            duration = frame_count / fps
        
        cap.release()
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration,
        }
    except Exception:
        return {
            "width": None,
            "height": None,
            "fps": None,
            "frame_count": None,
            "duration_sec": None
        }


def _probe_video_stats(path: str) -> Dict:
    """Get comprehensive video stats using ffprobe, fallback to OpenCV."""
    # Try ffprobe first
    stats = _probe_video_stats_ffprobe(path)
    
    # Fill in missing basic stats with OpenCV if needed
    if stats.get("width") is None or stats.get("height") is None:
        opencv_stats = _probe_video_stats_opencv(path)
        for key in ["width", "height", "fps", "frame_count", "duration_sec"]:
            if stats.get(key) is None:
                stats[key] = opencv_stats.get(key)
    
    return stats


def scan_videos(cfg: FVCConfig, compute_stats: bool = True) -> List[Dict]:
    """
    Scan video folders and collect video information.
    
    Args:
        cfg: FVCConfig instance
        compute_stats: Whether to compute video statistics (slower)
    
    Returns:
        List of dictionaries with video information
    """
    records: List[Dict] = []
    
    for subset in cfg.subsets:
        subset_root = os.path.join(cfg.videos_dir, subset)
        if not os.path.isdir(subset_root):
            # maybe not downloaded yet, skip
            logger.warning(f"{subset_root} not found, skipping...")
            continue
        
        # Check structure: FVC1 has platform subfolders, FVC2/FVC3 may not
        first_level = [d for d in os.listdir(subset_root) 
                      if os.path.isdir(os.path.join(subset_root, d))]
        
        # Check if first level items are platforms (twitter/youtube) or video_ids
        # Platforms are typically lowercase strings, video_ids are typically numeric/alphanumeric
        has_platform_subfolders = any(p in first_level for p in ['twitter', 'youtube', 'facebook'])
        
        if has_platform_subfolders:
            # Structure: FVC*/platform/video_id/video.mp4 (FVC1 style)
            for platform in first_level:
                platform_root = os.path.join(subset_root, platform)
                if not os.path.isdir(platform_root):
                    continue
                
                video_ids = [d for d in os.listdir(platform_root) 
                            if os.path.isdir(os.path.join(platform_root, d))]
                
                for video_id in tqdm(video_ids, desc=f"Scanning {subset}/{platform}"):
                    video_folder = os.path.join(platform_root, video_id)
                    if not os.path.isdir(video_folder):
                        continue
                    
                    video_file = find_video_file(video_folder)
                    if video_file is None:
                        continue
                    
                    rel_path = os.path.relpath(video_file, cfg.videos_dir)
                    # Prepend "videos/" to maintain relative path from root
                    rel_path = os.path.join("videos", rel_path)
                    
                    rec = {
                        "subset": subset,
                        "platform": platform,
                        "video_id": video_id,
                        "video_path": rel_path,
                    }
                    
                    if compute_stats:
                        stats = _probe_video_stats(video_file)
                        rec.update(stats)
                    
                    records.append(rec)
        else:
            # Structure: FVC*/video_id/video.mp4 (FVC2/FVC3 style, likely Facebook)
            # Try to infer platform from video_id format or default to 'facebook'
            for video_id in tqdm(first_level, desc=f"Scanning {subset}"):
                video_folder = os.path.join(subset_root, video_id)
                if not os.path.isdir(video_folder):
                    continue
                
                video_file = find_video_file(video_folder)
                if video_file is None:
                    continue
                
                rel_path = os.path.relpath(video_file, cfg.videos_dir)
                # Prepend "videos/" to maintain relative path from root
                rel_path = os.path.join("videos", rel_path)
                
                # Infer platform: numeric IDs are likely Facebook, alphanumeric could be YouTube/Twitter
                # Default to 'facebook' for FVC2/FVC3 structure
                platform = "facebook" if video_id.isdigit() else "unknown"
                
                rec = {
                    "subset": subset,
                    "platform": platform,
                    "video_id": video_id,
                    "video_path": rel_path,
                }
                
                if compute_stats:
                    stats = _probe_video_stats(video_file)
                    rec.update(stats)
                
                records.append(rec)
    
    return records

