"""
Path resolution utilities.

Provides:
- Video path resolution
- Path validation
- Atomic file operations with locking
- Video file validation
"""

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import polars as pl

# File locking support (Unix only, with fallback)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


def resolve_video_path(video_rel: str, project_root: str) -> str:
    """
    Resolve a relative video path to an absolute path.
    
    Tries multiple path resolution strategies in order:
    1. Add 'videos/' prefix (most common: CSV has FVC1/... but files are at videos/FVC1/...)
    2. Direct relative path from project_root
    3. Remove 'videos/' prefix if present
    
    Args:
        video_rel: Relative video path from CSV (e.g., "FVC1/youtube/.../video.mp4")
        project_root: Project root directory
        
    Returns:
        Absolute path to the video file (first existing path found, or most likely path)
    """
    if not video_rel:
        raise ValueError("video_rel cannot be empty")
    
    video_rel = str(video_rel).strip()
    project_root = Path(project_root).resolve()
    
    # Strategy 1: Add 'videos/' prefix (most common case)
    candidate1 = project_root / "videos" / video_rel
    if candidate1.exists():
        return str(candidate1)
    
    # Strategy 2: Direct relative path from project_root
    candidate2 = project_root / video_rel
    if candidate2.exists():
        return str(candidate2)
    
    # Strategy 3: Remove 'videos/' prefix if present
    if video_rel.startswith("videos/"):
        video_rel_no_prefix = video_rel[7:]  # Remove "videos/"
        candidate3 = project_root / video_rel_no_prefix
        if candidate3.exists():
            return str(candidate3)
    
    # If none exist, return the most likely path (strategy 1)
    return str(candidate1)


def get_video_path_candidates(video_rel: str, project_root: str) -> List[str]:
    """
    Get all possible candidate paths for a video.
    
    Returns:
        List of candidate absolute paths in order of likelihood
    """
    if not video_rel:
        return []
    
    video_rel = str(video_rel).strip()
    project_root = Path(project_root).resolve()
    
    candidates = []
    
    # Strategy 1: Add 'videos/' prefix
    candidates.append(str(project_root / "videos" / video_rel))
    
    # Strategy 2: Direct relative path
    candidates.append(str(project_root / video_rel))
    
    # Strategy 3: Remove 'videos/' prefix if present
    if video_rel.startswith("videos/"):
        video_rel_no_prefix = video_rel[7:]
        candidates.append(str(project_root / video_rel_no_prefix))
    
    return candidates


def check_video_path_exists(video_rel: str, project_root: str) -> bool:
    """
    Check if a video file exists at any of the candidate paths.
    
    Args:
        video_rel: Relative video path
        project_root: Project root directory
        
    Returns:
        True if video exists at any candidate path, False otherwise
    """
    candidates = get_video_path_candidates(video_rel, project_root)
    return any(os.path.exists(c) for c in candidates)


def find_metadata_file(base_path: Path, filename_base: str) -> Optional[Path]:
    """
    Find metadata file with any supported extension (.arrow, .parquet, .csv).
    
    Tries in order: .arrow, .parquet, .csv (preferring Arrow format).
    
    Args:
        base_path: Base directory path
        filename_base: Filename without extension (e.g., "augmented_metadata")
    
    Returns:
        Path to existing metadata file, or None if not found
    """
    base_path = Path(base_path)
    for ext in ['.arrow', '.parquet', '.csv']:
        candidate = base_path / f"{filename_base}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_metadata_flexible(path: str, max_retries: int = 3, retry_delay: float = 0.5) -> Optional[pl.DataFrame]:
    """
    Load metadata from CSV, Arrow, or Parquet format.
    
    Tries formats in order: .arrow, .parquet, .csv (preferring Arrow format).
    Handles race conditions and corrupted files gracefully with retry logic.
    
    Args:
        path: Path to metadata file (with or without extension)
        max_retries: Maximum number of retries if file read fails (default: 3)
        retry_delay: Delay between retries in seconds (default: 0.5)
    
    Returns:
        Polars DataFrame, or None if file doesn't exist or is corrupted
    """
    import logging
    logger = logging.getLogger(__name__)
    
    path_obj = Path(path)
    
    # If path doesn't exist, try with different extensions
    if not path_obj.exists():
        for ext in ['.arrow', '.parquet', '.csv']:
            candidate = path_obj.with_suffix(ext)
            if candidate.exists():
                path_obj = candidate
                break
        else:
            return None
    
    # Check if file is empty (likely corrupted or being written)
    try:
        if path_obj.stat().st_size == 0:
            logger.warning(f"Metadata file is empty: {path_obj}, trying other formats...")
            # Try other formats
            base_path = path_obj.with_suffix('')
            for ext in ['.parquet', '.csv']:
                candidate = base_path.with_suffix(ext)
                if candidate.exists() and candidate.stat().st_size > 0:
                    path_obj = candidate
                    break
            else:
                logger.warning(f"No valid metadata file found for {path}")
                return None
    except OSError as e:
        logger.warning(f"Could not stat file {path_obj}: {e}")
        return None
    
    # Retry logic for race conditions with immediate fallback for corrupted files
    last_error = None
    for attempt in range(max_retries):
        try:
            # Load based on extension
            if path_obj.suffix == '.arrow':
                try:
                    return pl.read_ipc(path_obj)
                except (OSError, ValueError) as e:
                    # Arrow file might be corrupted or being written
                    error_str = str(e).lower()
                    if 'invalid argument' in error_str or 'os error 22' in error_str:
                        logger.warning(f"Arrow file appears corrupted or locked (attempt {attempt + 1}/{max_retries}): {e}")
                        # Immediately try fallback formats instead of retrying corrupted Arrow
                        base_path = path_obj.with_suffix('')
                        for ext in ['.parquet', '.csv']:
                            candidate = base_path.with_suffix(ext)
                            if candidate.exists():
                                try:
                                    if candidate.stat().st_size == 0:
                                        continue  # Skip empty files
                                    if candidate.suffix == '.parquet':
                                        logger.info(f"Falling back to Parquet format: {candidate}")
                                        return pl.read_parquet(candidate)
                                    else:
                                        logger.info(f"Falling back to CSV format: {candidate}")
                                        return pl.read_csv(candidate)
                                except Exception as fallback_error:
                                    logger.debug(f"Fallback to {candidate} failed: {fallback_error}")
                                    continue
                        # If all fallbacks failed and we have retries left, wait and retry Arrow
                        if attempt < max_retries - 1:
                            logger.debug(f"All formats failed, retrying Arrow after delay (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        logger.error(f"All metadata formats failed for {path} after {max_retries} attempts")
                        return None
                    # For other errors, re-raise
                    raise
            elif path_obj.suffix == '.parquet':
                try:
                    return pl.read_parquet(path_obj)
                except (OSError, ValueError) as e:
                    error_str = str(e).lower()
                    if 'invalid argument' in error_str or 'os error 22' in error_str:
                        logger.warning(f"Parquet file appears corrupted, trying CSV fallback: {e}")
                        csv_candidate = path_obj.with_suffix('.csv')
                        if csv_candidate.exists() and csv_candidate.stat().st_size > 0:
                            try:
                                return pl.read_csv(csv_candidate)
                            except Exception:
                                pass
                    if attempt < max_retries - 1:
                        logger.debug(f"Parquet read failed (attempt {attempt + 1}/{max_retries}), retrying: {e}")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise
            else:
                # CSV - no fallback, just retry
                try:
                    return pl.read_csv(path_obj)
                except (OSError, ValueError) as e:
                    error_str = str(e).lower()
                    if 'invalid argument' in error_str or 'os error 22' in error_str or 'permission denied' in error_str:
                        if attempt < max_retries - 1:
                            logger.debug(f"CSV read failed (attempt {attempt + 1}/{max_retries}), retrying: {e}")
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                    raise
        except (OSError, ValueError) as e:
            last_error = e
            error_str = str(e).lower()
            if 'invalid argument' in error_str or 'os error 22' in error_str or 'permission denied' in error_str:
                # File might be locked or being written
                if attempt < max_retries - 1:
                    logger.debug(f"File read failed (attempt {attempt + 1}/{max_retries}), retrying: {e}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
            else:
                # Other errors, don't retry
                logger.error(f"Failed to load metadata from {path_obj}: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error loading metadata from {path_obj}: {e}")
            raise
    
    # All retries failed
    logger.error(f"Failed to load metadata from {path_obj} after {max_retries} attempts: {last_error}")
    return None


def validate_metadata_columns(df: pl.DataFrame, required_columns: List[str], stage_name: str = "") -> None:
    """
    Validate that a metadata DataFrame has all required columns.
    
    Args:
        df: Polars DataFrame to validate
        required_columns: List of required column names
        stage_name: Name of the stage (for error messages)
    
    Raises:
        ValueError: If any required columns are missing
    """
    if df is None or df.height == 0:
        raise ValueError(f"{stage_name}: Metadata DataFrame is empty or None")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{stage_name}: Missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )


def validate_video_file(video_path: str | Path, check_decode: bool = True) -> Tuple[bool, str]:
    """
    Validate that a video file exists and is not corrupted.
    
    Checks:
    1. File exists
    2. File is not empty (size > 0)
    3. Can open video container
    4. Has at least one video stream
    5. (Optional) Can decode at least one frame
    
    Args:
        video_path: Path to video file
        check_decode: If True, try to decode at least one frame (default: True)
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if video is valid, False otherwise
        - error_message: Error description if invalid, "OK" if valid
    """
    import logging
    logger = logging.getLogger(__name__)
    
    video_path = Path(video_path)
    
    # Check if file exists
    if not video_path.exists():
        return False, "File does not exist"
    
    # Check if file is empty
    try:
        if video_path.stat().st_size == 0:
            return False, "File is empty (0 bytes)"
    except OSError as e:
        return False, f"Cannot access file: {e}"
    
    # Try to open video container
    try:
        import av
        container = av.open(str(video_path))
        
        # Check if video has at least one video stream
        if len(container.streams.video) == 0:
            container.close()
            return False, "No video stream found"
        
        # Optionally try to decode first frame to verify file integrity
        if check_decode:
            try:
                stream = container.streams.video[0]
                # Try to decode at least one frame
                for frame in container.decode(video=0):
                    # If we can decode at least one frame, file is likely valid
                    break
            except Exception as e:
                container.close()
                return False, f"Cannot decode video: {e}"
        
        container.close()
        return True, "OK"
        
    except ImportError:
        # PyAV not available, skip validation
        logger.warning("PyAV not available, skipping video validation")
        return True, "OK (validation skipped)"
    except Exception as e:
        error_msg = str(e)
        # Check for specific corruption errors
        if "Invalid data" in error_msg or "Invalid argument" in error_msg:
            return False, f"Corrupted video file: {e}"
        return False, f"Cannot open video file: {e}"


def calculate_adaptive_num_frames(
    total_frames: int,
    frame_percentage: float = 0.10,
    min_frames: int = 5,
    max_frames: int = 50
) -> int:
    """
    Calculate number of frames to sample based on percentage with min/max bounds.
    
    Args:
        total_frames: Total frames in video
        frame_percentage: Percentage of frames to sample (default: 0.10 = 10%)
        min_frames: Minimum frames to sample (for very short videos)
        max_frames: Maximum frames to sample (for memory efficiency)
    
    Returns:
        Number of frames to sample
    """
    if total_frames <= 0:
        return min_frames
    
    calculated_frames = int(total_frames * frame_percentage)
    return max(min_frames, min(max_frames, calculated_frames))


def write_metadata_atomic(
    df: pl.DataFrame,
    output_path: Path,
    max_retries: int = 5,
    retry_delay: float = 0.5,
    lock_timeout: float = 30.0,
    append: bool = False
) -> bool:
    """
    Write metadata DataFrame to file atomically with file locking.
    
    Uses atomic write pattern: write to temp file, then rename.
    Uses file locking to prevent concurrent writes.
    Supports append mode: reads existing data, merges with new, writes back.
    
    Args:
        df: Polars DataFrame to write
        output_path: Target output path (will try .arrow, .parquet, .csv)
        max_retries: Maximum retries for file locking (default: 5)
        retry_delay: Delay between retry attempts (default: 0.5s)
        lock_timeout: Maximum time to wait for lock (default: 30s)
        append: If True, read existing file, merge with new data, write back (default: False)
    
    Returns:
        True if successful, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if df is None:
        logger.warning(f"Cannot write None DataFrame to {output_path}")
        return False
    
    if df.height == 0:
        if not append:
            logger.warning(f"Cannot write empty DataFrame to {output_path}")
            return False
        # In append mode, empty DataFrame is okay if we're just reading existing
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format based on extension, default to Arrow
    if output_path.suffix == '.parquet':
        use_arrow = False
        use_csv = False
    elif output_path.suffix == '.csv':
        use_arrow = False
        use_csv = True
    else:
        # Default to Arrow, fallback to Parquet
        use_arrow = True
        use_csv = False
    
    # In append mode, read existing data and merge (before acquiring lock)
    if append and output_path.exists() and output_path.stat().st_size > 0:
        try:
            existing_df = None
            if use_arrow:
                existing_df = load_metadata_flexible(str(output_path), max_retries=3, retry_delay=0.5)
            elif use_csv:
                try:
                    existing_df = pl.read_csv(str(output_path))
                except Exception:
                    pass
            else:
                try:
                    existing_df = pl.read_parquet(str(output_path))
                except Exception:
                    pass
            
            if existing_df is not None and existing_df.height > 0:
                # Merge: combine existing and new, remove duplicates by video_path (or first column if no video_path)
                merge_key = "video_path" if "video_path" in existing_df.columns and "video_path" in df.columns else None
                if merge_key:
                    # Remove duplicates, keep last (newest) entry
                    combined_df = pl.concat([existing_df, df]).unique(subset=[merge_key], keep="last")
                    logger.debug(f"Appended {df.height} rows to existing {existing_df.height} rows (merged to {combined_df.height} unique rows)")
                else:
                    # No merge key, just concatenate
                    combined_df = pl.concat([existing_df, df])
                    logger.debug(f"Appended {df.height} rows to existing {existing_df.height} rows (total: {combined_df.height} rows)")
                df = combined_df
            else:
                logger.debug(f"Existing file was empty or couldn't be read, writing new data only")
        except Exception as e:
            logger.warning(f"Could not read existing metadata for append (will overwrite): {e}")
            # Continue with write (will overwrite)
    
    # Create lock file path
    lock_file = output_path.with_suffix(output_path.suffix + '.lock')
    
    # Try to acquire lock and write
    lock_fd = None
    for attempt in range(max_retries):
        try:
            # Try to acquire exclusive lock (Unix only)
            if HAS_FCNTL:
                lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                # Windows or system without fcntl - use file existence as lock
                if lock_file.exists():
                    raise BlockingIOError("Lock file exists")
                lock_file.touch()
                lock_fd = None
            
            # Lock acquired, proceed with atomic write
            try:
                # Create temp file in same directory (for atomic rename)
                temp_file = output_path.parent / f".{output_path.name}.tmp.{os.getpid()}"
                
                # Write to temp file
                try:
                    if use_arrow:
                        try:
                            df.write_ipc(str(temp_file))
                        except Exception as e:
                            logger.warning(f"Arrow write failed, falling back to Parquet: {e}")
                            use_arrow = False
                            temp_file = output_path.with_suffix('.parquet').parent / f".{output_path.with_suffix('.parquet').name}.tmp.{os.getpid()}"
                            df.write_parquet(str(temp_file))
                            output_path = output_path.with_suffix('.parquet')
                    elif use_csv:
                        df.write_csv(str(temp_file))
                    else:
                        df.write_parquet(str(temp_file))
                    
                    # Atomic rename (works on most Unix filesystems)
                    temp_file.replace(output_path)
                    
                    logger.debug(f"Atomically wrote metadata to {output_path}")
                    return True
                    
                except Exception as e:
                    # Clean up temp file on error
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception:
                        pass
                    raise
                    
            finally:
                # Release lock
                try:
                    if HAS_FCNTL and lock_fd is not None:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        os.close(lock_fd)
                    lock_fd = None
                    # Remove lock file
                    try:
                        if lock_file.exists():
                            lock_file.unlink()
                    except Exception:
                        pass
                except Exception:
                    pass
                    
        except (OSError, BlockingIOError, FileExistsError) as e:
            # Lock acquisition failed, retry
            if lock_fd is not None:
                try:
                    if HAS_FCNTL:
                        os.close(lock_fd)
                except Exception:
                    pass
                lock_fd = None
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                logger.debug(f"Could not acquire lock for {output_path} (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Could not acquire lock for {output_path} after {max_retries} attempts, writing without lock")
                # Last resort: write without lock (not ideal but better than failing)
                try:
                    if use_arrow:
                        try:
                            df.write_ipc(str(output_path))
                        except Exception:
                            df.write_parquet(str(output_path.with_suffix('.parquet')))
                    elif use_csv:
                        df.write_csv(str(output_path))
                    else:
                        df.write_parquet(str(output_path))
                    return True
                except Exception as e:
                    logger.error(f"Failed to write metadata to {output_path} even without lock: {e}")
                    return False
        except Exception as e:
            # Other errors
            if lock_fd is not None:
                try:
                    if HAS_FCNTL:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        os.close(lock_fd)
                except Exception:
                    pass
                try:
                    if lock_file.exists():
                        lock_file.unlink()
                except Exception:
                    pass
            logger.error(f"Unexpected error writing metadata to {output_path}: {e}")
            return False
    
    return False


def get_video_metadata_cache_path(project_root: Optional[Path] = None) -> Optional[Path]:
    """
    Get the path to the persistent video metadata cache file.
    
    Args:
        project_root: Project root directory. If None, returns None (no persistent cache).
    
    Returns:
        Path to cache file, or None if project_root is None
    """
    if project_root is None:
        return None
    project_root = Path(project_root)
    return project_root / "data" / ".video_metadata_cache.json"


__all__ = [
    "resolve_video_path",
    "get_video_path_candidates",
    "check_video_path_exists",
    "find_metadata_file",
    "load_metadata_flexible",
    "write_metadata_atomic",
    "validate_metadata_columns",
    "validate_video_file",
    "calculate_adaptive_num_frames",
    "get_video_metadata_cache_path",
]
