"""
Extract features from scaled videos.

Extracts features that are detectable after scaling (downscaling or upscaling), focusing on:
- Edge preservation metrics
- Texture uniformity
- Compression artifact visibility
- Color consistency
- Scaling direction indicators (is_upscaled, is_downscaled)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import av
import cv2

from lib.data import load_metadata
from lib.utils.paths import resolve_video_path, validate_video_file, get_video_metadata_cache_path, load_metadata_flexible, write_metadata_atomic
from lib.utils.memory import aggressive_gc, log_memory_stats, safe_execute
from lib.features.handcrafted import HandcraftedFeatureExtractor

logger = logging.getLogger(__name__)


def extract_scaled_features(
    video_path: str,
    num_frames: Optional[int] = None,
    frame_percentage: Optional[float] = None,
    min_frames: int = 5,
    max_frames: int = 50,
    project_root: Optional[Path] = None
) -> dict:
    """
    Extract ALL features from scaled videos (base handcrafted + scaled-specific).
    
    Returns 23 features total:
    - 15 base handcrafted features (noise, DCT, blur, boundary, codec)
    - 6 scaled-specific features (edge, texture, color, compression)
    - 2 scaling indicators (is_upscaled, is_downscaled) - added separately
    
    Args:
        video_path: Path to scaled video file
        num_frames: Number of frames to sample (if provided, overrides percentage-based calculation)
        frame_percentage: Percentage of frames to sample (default: 0.10 = 10% if num_frames not provided)
        min_frames: Minimum frames to sample (for percentage-based sampling, default: 5)
        max_frames: Maximum frames to sample (for percentage-based sampling, default: 50)
    
    Returns:
        Dictionary of ALL features (15 base + 6 scaled-specific = 21 features)
        Note: is_upscaled and is_downscaled are added separately in stage4_extract_scaled_features
    """
    container = None
    try:
        container = av.open(video_path)
        if len(container.streams.video) == 0:
            logger.warning(f"Video has no video stream: {video_path}")
            return {}
        
        stream = container.streams.video[0]
        total_frames = stream.frames if stream.frames > 0 else 0
        
        if total_frames == 0:
            logger.warning(f"Video has no frames: {video_path}")
            return {}
        
        # Calculate number of frames to sample
        from lib.utils.paths import calculate_adaptive_num_frames
        
        if num_frames is not None:
            # Use fixed number of frames (backward compatible)
            frames_to_sample = num_frames
        else:
            # Use percentage-based adaptive sampling
            if frame_percentage is None:
                frame_percentage = 0.10  # Default 10%
            frames_to_sample = calculate_adaptive_num_frames(
                total_frames, frame_percentage, min_frames, max_frames
            )
            logger.debug(f"Adaptive sampling: {total_frames} total frames -> {frames_to_sample} frames ({frame_percentage*100:.1f}%, bounded [{min_frames}, {max_frames}])")
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)
        logger.debug(f"Stage 4: Sampling {frames_to_sample} frames from {total_frames} total frames")
        logger.debug(f"Stage 4: Frame indices: {frame_indices[:5]}..." if len(frame_indices) > 5 else f"Stage 4: Frame indices: {frame_indices}")
        
        all_base_features = []  # For base handcrafted features
        all_scaled_features = []  # For scaled-specific features
        frame_count = 0
        
        # Import base feature extractor
        from lib.features.handcrafted import extract_all_features
        logger.debug(f"Stage 4: Starting frame extraction loop...")
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_count in frame_indices:
                    logger.debug(f"Stage 4: Processing frame {frame_count}/{total_frames} (index {len(all_base_features)+1}/{frames_to_sample})")
                    frame_array = frame.to_ndarray(format='rgb24')
                    logger.debug(f"Stage 4: Frame shape: {frame_array.shape}, dtype: {frame_array.dtype}")
                    
                    # Extract BASE handcrafted features (15 features)
                    logger.debug(f"Stage 4: Extracting base handcrafted features from frame {frame_count}...")
                    base_features = extract_all_features(frame_array, str(video_path))
                    logger.debug(f"Stage 4: Extracted {len(base_features)} base features: {list(base_features.keys())[:5]}..." if len(base_features) > 5 else f"Stage 4: Extracted {len(base_features)} base features: {list(base_features.keys())}")
                    all_base_features.append(base_features)
                    
                    # Extract scaled-video-specific features (6 features)
                    logger.debug(f"Stage 4: Extracting scaled-specific features from frame {frame_count}...")
                    scaled_features = {}
                    
                    # Edge preservation (Canny edges)
                    gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    scaled_features["edge_density"] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
                    
                    # Texture uniformity (variance of local means)
                    kernel = np.ones((5, 5), np.float32) / 25
                    local_means = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                    scaled_features["texture_uniformity"] = float(1.0 / (1.0 + np.std(local_means)))
                    
                    # Color consistency (variance across channels)
                    scaled_features["color_consistency_r"] = float(np.std(frame_array[:, :, 0]))
                    scaled_features["color_consistency_g"] = float(np.std(frame_array[:, :, 1]))
                    scaled_features["color_consistency_b"] = float(np.std(frame_array[:, :, 2]))
                    
                    # Compression artifacts (blockiness)
                    h, w = gray.shape
                    block_size = 8
                    blockiness = 0.0
                    for i in range(0, h - block_size, block_size):
                        for j in range(0, w - block_size, block_size):
                            block = gray[i:i+block_size, j:j+block_size]
                            # Measure horizontal and vertical discontinuities
                            h_diff = np.mean(np.abs(np.diff(block, axis=1)))
                            v_diff = np.mean(np.abs(np.diff(block, axis=0)))
                            blockiness += h_diff + v_diff
                    scaled_features["compression_artifacts"] = float(blockiness / ((h // block_size) * (w // block_size)))
                    
                    all_scaled_features.append(scaled_features)
                    
                    # Aggressive GC after each frame extraction
                    del frame_array, gray, edges, local_means
                    aggressive_gc(clear_cuda=False)
                
                frame_count += 1
                if frame_count >= total_frames or len(all_base_features) >= frames_to_sample:
                    break
            
            if frame_count >= total_frames or len(all_base_features) >= frames_to_sample:
                break
        
        # Aggregate features across frames (mean)
        if not all_base_features:
            return {}
        
        # Aggregate base features
        aggregated = {}
        logger.debug(f"Stage 4: Aggregating {len(all_base_features)} base feature frames")
        for key in all_base_features[0].keys():
            values = [f[key] for f in all_base_features if key in f]
            aggregated[key] = float(np.mean(values)) if values else 0.0
            logger.debug(f"Stage 4: Base feature '{key}': {len(values)} values, mean={aggregated[key]:.6f}")
        
        # Aggregate scaled-specific features
        logger.debug(f"Stage 4: Aggregating {len(all_scaled_features)} scaled-specific feature frames")
        for key in all_scaled_features[0].keys():
            values = [f[key] for f in all_scaled_features if key in f]
            aggregated[key] = float(np.mean(values)) if values else 0.0
            logger.debug(f"Stage 4: Scaled feature '{key}': {len(values)} values, mean={aggregated[key]:.6f}")
        
        logger.debug(f"Stage 4: Total aggregated features: {len(aggregated)}")
        logger.debug(f"Stage 4: Feature names: {list(aggregated.keys())}")
        return aggregated
        
    except Exception as e:
        error_str = str(e)
        # Check for specific corruption errors
        if "Invalid data" in error_str or "Invalid argument" in error_str or "os error" in error_str.lower():
            logger.error(f"Corrupted video file detected: {video_path} - {e}")
        else:
            logger.error(f"Failed to extract scaled video features from {video_path}: {e}")
        return {}
    finally:
        if container is not None:
            try:
                container.close()
            except (OSError, RuntimeError, AttributeError):
                pass
        aggressive_gc(clear_cuda=False)


def stage4_extract_scaled_features(
    project_root: str,
    scaled_metadata_path: str,
    output_dir: str = "data/features_stage4",
    num_frames: Optional[int] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    delete_existing: bool = False,
    resume: bool = True,
    frame_percentage: Optional[float] = None,
    min_frames: int = 5,
    max_frames: int = 50,
    execution_order: str = "forward"
) -> pl.DataFrame:
    """
    Stage 4: Extract additional features from scaled videos.
    
    Extracts features specific to scaled videos and includes binary features:
    - is_upscaled: 1 if video was upscaled, 0 otherwise
    - is_downscaled: 1 if video was downscaled, 0 otherwise
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to scaled metadata (from Stage 3) - supports CSV/Arrow/Parquet
        output_dir: Directory to save features
        num_frames: Number of frames to sample per video (if provided, overrides percentage-based calculation)
        start_idx: Start index for video range (0-based, inclusive). If None, starts from 0.
        end_idx: End index for video range (0-based, exclusive). If None, processes all videos.
        delete_existing: If True, delete existing feature files before regenerating (clean mode)
        resume: If True, skip videos where feature files already exist (resume mode)
        frame_percentage: Percentage of frames to sample (default: 0.10 = 10% if num_frames not provided)
        min_frames: Minimum frames to sample (for percentage-based sampling, default: 5)
        max_frames: Maximum frames to sample (for percentage-based sampling, default: 50)
    
    Returns:
        DataFrame with feature metadata (includes is_upscaled and is_downscaled features)
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scaled metadata (support CSV, Arrow, and Parquet)
    logger.info("Stage 4: Loading scaled metadata...")
    from lib.utils.paths import validate_metadata_columns
    
    df = load_metadata_flexible(scaled_metadata_path)
    if df is None:
        logger.error(f"Scaled metadata not found: {scaled_metadata_path} (checked .arrow, .parquet, .csv)")
        return pl.DataFrame()
    
    # Validate required columns
    try:
        validate_metadata_columns(df, ["video_path", "label"], "Stage 4")
    except ValueError as e:
        logger.error(f"{e}")
        return pl.DataFrame()
    
    # Apply range filtering if specified
    total_videos = df.height
    if total_videos == 0:
        logger.warning("Stage 4: No videos found in metadata")
        return pl.DataFrame()
    
    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else total_videos
        if start < 0:
            start = 0
        if end > total_videos:
            end = total_videos
        if start >= end:
            logger.warning(f"Invalid range: start_idx={start}, end_idx={end}, total_videos={total_videos}. Skipping.")
            return pl.DataFrame()
        df = df.slice(start, end - start)
        logger.info(f"Stage 4: Processing video range [{start}, {end}) of {total_videos} total videos")
        
        # Check if range resulted in empty DataFrame
        if df.height == 0:
            logger.warning(f"Stage 4: Range [{start}, {end}) resulted in empty DataFrame")
            return pl.DataFrame()
    else:
        logger.info(f"Stage 4: Processing all {total_videos} videos")
    
    logger.info(f"Stage 4: Processing {df.height} scaled videos in this range")
    
    # Load existing metadata if it exists (for resume mode)
    existing_metadata = None
    existing_feature_paths = set()
    
    if resume and not delete_existing:
        # Try to load existing metadata (check all formats)
        # Use retry logic to handle race conditions and corrupted files
        existing_metadata_path = output_dir / "features_scaled_metadata"
        try:
            existing_metadata = load_metadata_flexible(str(existing_metadata_path), max_retries=5, retry_delay=1.0)
            if existing_metadata is not None and existing_metadata.height > 0:
                existing_feature_paths = set(existing_metadata["feature_path"].to_list())
                logger.info(f"Stage 4: Found {len(existing_feature_paths)} existing feature files (resume mode)")
            else:
                logger.info("Stage 4: No existing metadata found or metadata is empty, starting fresh")
                existing_metadata = None
                existing_feature_paths = set()
        except Exception as e:
            logger.warning(f"Stage 4: Could not load existing metadata (will start fresh): {e}")
            existing_metadata = None
            existing_feature_paths = set()
    
    # Delete existing feature files if clean mode
    if delete_existing:
        logger.info("Stage 4: Deleting existing feature files (clean mode)...")
        deleted_count = 0
        for feature_file in output_dir.glob("*_scaled_features.*"):
            if feature_file.name not in ["features_scaled_metadata.arrow", "features_scaled_metadata.parquet"]:
                try:
                    feature_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {feature_file}: {e}")
        logger.info(f"Stage 4: Deleted {deleted_count} existing feature files")
        existing_feature_paths = set()  # Clear after deletion
    
    feature_rows = []
    skipped_count = 0
    processed_count = 0
    corrupted_count = 0
    corrupted_files = []
    
    # Determine iteration order
    if execution_order == "reverse":
        indices = range(df.height - 1, -1, -1)  # Reverse: from end to start
        logger.info("Stage 4: Processing videos in REVERSE order (from end to start)")
    else:
        indices = range(df.height)  # Forward: from start to end (default)
        logger.info("Stage 4: Processing videos in FORWARD order (from start to end)")
    
    iteration_count = 0
    for idx in indices:
        iteration_count += 1
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        
        logger.info(f"Stage 4: [{iteration_count}/{df.height}] Processing video {idx}: {Path(video_rel).name}")
        logger.debug(f"Stage 4: Row data: video_path={video_rel}, label={label}")
        
        # Get original dimensions if available
        original_width = row.get("original_width")
        original_height = row.get("original_height")
        logger.debug(f"Stage 4: Original dimensions: {original_width}x{original_height}")
        
        try:
            logger.debug(f"Stage 4: Resolving video path for: {video_rel}")
            video_path = resolve_video_path(video_rel, project_root)
            logger.debug(f"Stage 4: Resolved path: {video_path}")
            
            if not Path(video_path).exists():
                logger.warning(f"Stage 4: [{iteration_count}/{df.height}] Video not found: {video_path}")
                continue
            
            logger.debug(f"Stage 4: Video file exists: {Path(video_path).exists()}, size: {Path(video_path).stat().st_size if Path(video_path).exists() else 'N/A'} bytes")
            
            # Validate video file before processing
            logger.debug(f"Stage 4: Validating video file: {video_path}")
            is_valid, error_msg = validate_video_file(video_path, check_decode=True)
            if not is_valid:
                logger.warning(f"Stage 4: [{iteration_count}/{df.height}] Skipping corrupted/invalid video: {video_path} - {error_msg}")
                corrupted_count += 1
                corrupted_files.append({
                    "video_path": video_rel,
                    "absolute_path": str(video_path),
                    "error": error_msg
                })
                continue
            logger.debug(f"Stage 4: Video file validation passed")
            
            if iteration_count % 10 == 0:
                log_memory_stats(f"Stage 4: processing video {iteration_count}/{df.height} (index {idx})")
            
            # Check if feature file already exists (resume mode) - CHECK BEFORE EXTRACTING FEATURES
            video_id = Path(video_path).stem
            feature_path_parquet = output_dir / f"{video_id}_scaled_features.parquet"
            feature_path_npy = output_dir / f"{video_id}_scaled_features.npy"
            feature_path_rel = str(feature_path_parquet.relative_to(project_root))
            
            if resume and not delete_existing:
                # Check if feature file exists BEFORE extracting features
                if feature_path_parquet.exists() or feature_path_npy.exists():
                    logger.debug(f"Stage 4: Feature file already exists: {feature_path_parquet} or {feature_path_npy}")
                    if feature_path_rel in existing_feature_paths or feature_path_parquet.exists() or feature_path_npy.exists():
                        logger.info(f"Stage 4: [{iteration_count}/{df.height}] Skipping {Path(video_path).name} - feature file already exists")
                        skipped_count += 1
                        # Still add to metadata if not already present
                        if existing_metadata is None or feature_path_rel not in existing_feature_paths:
                            logger.debug(f"Stage 4: Feature file exists but not in metadata, will try to load and add")
                            # Try to load existing features to add to metadata
                            try:
                                if feature_path_parquet.exists():
                                    import pyarrow.parquet as pq
                                    table = pq.read_table(str(feature_path_parquet))
                                    features_dict = {col: table[col][0].as_py() for col in table.column_names}
                                else:
                                    features_dict = np.load(str(feature_path_npy), allow_pickle=True).item()
                                
                                feature_row = {
                                    "video_path": video_rel,
                                    "label": label,
                                    "feature_path": feature_path_rel,
                                }
                                feature_row.update(features_dict)
                                feature_rows.append(feature_row)
                            except Exception as e:
                                logger.warning(f"Could not load existing features from {feature_path_parquet}: {e}")
                        continue
            
            # Get scaled video dimensions
            scaled_width = None
            scaled_height = None
            container = None
            try:
                container = av.open(video_path)
                stream = container.streams.video[0]
                scaled_width = stream.width
                scaled_height = stream.height
            except Exception as e:
                logger.debug(f"Could not get scaled dimensions: {e}")
            finally:
                if container is not None:
                    try:
                        container.close()
                    except (OSError, RuntimeError, AttributeError):
                        pass
            
            # Calculate scaling direction features
            is_upscaled = 0
            is_downscaled = 0
            if original_width is not None and original_height is not None and scaled_width is not None and scaled_height is not None:
                original_max_dim = max(original_width, original_height)
                scaled_max_dim = max(scaled_width, scaled_height)
                
                logger.debug(f"Stage 4: Scaling info - original: {original_width}x{original_height} (max={original_max_dim}), scaled: {scaled_width}x{scaled_height} (max={scaled_max_dim})")
                
                if scaled_max_dim > original_max_dim:
                    is_upscaled = 1
                    logger.debug(f"Stage 4: Video is UPSCALED")
                elif scaled_max_dim < original_max_dim:
                    is_downscaled = 1
                    logger.debug(f"Stage 4: Video is DOWNSCALED")
                else:
                    logger.debug(f"Stage 4: Video has NO SCALING (dimensions equal)")
            else:
                logger.debug(f"Stage 4: Scaling info unavailable - original: {original_width}x{original_height}, scaled: {scaled_width}x{scaled_height}")
            
            # Extract scaled-video-specific features with OOM handling
            logger.info(f"Stage 4: [{iteration_count}/{df.height}] Extracting features from {Path(video_path).name}...")
            logger.debug(f"Stage 4: Feature extraction parameters: num_frames={num_frames}, frame_percentage={frame_percentage}, min_frames={min_frames}, max_frames={max_frames}")
            
            features = safe_execute(
                extract_scaled_features,
                video_path,
                num_frames=num_frames,
                frame_percentage=frame_percentage,
                min_frames=min_frames,
                max_frames=max_frames,
                oom_retry=True,
                max_retries=1,
                context=f"Stage 4: extracting scaled features from {Path(video_path).name}"
            )
            
            if not features:
                logger.warning(f"Stage 4: [{iteration_count}/{df.height}] No features extracted from {video_path}")
                continue
            
            logger.debug(f"Stage 4: Extracted {len(features)} features: {list(features.keys())[:10]}..." if len(features) > 10 else f"Stage 4: Extracted {len(features)} features: {list(features.keys())}")
            logger.debug(f"Stage 4: Feature values sample: {dict(list(features.items())[:5])}")
            
            # Add scaling direction features
            features["is_upscaled"] = float(is_upscaled)
            features["is_downscaled"] = float(is_downscaled)
            logger.debug(f"Stage 4: Added scaling indicators: is_upscaled={is_upscaled}, is_downscaled={is_downscaled}")
            logger.debug(f"Stage 4: Total features after adding indicators: {len(features)}")
            
            # Save features as Arrow/Parquet (better than .npy)
            feature_path = feature_path_parquet
            
            logger.debug(f"Stage 4: Saving features to: {feature_path}")
            logger.debug(f"Stage 4: Feature dict keys: {list(features.keys())}")
            logger.debug(f"Stage 4: Feature dict values sample: {dict(list(features.items())[:5])}")
            
            # Convert features dict to Arrow table and save as Parquet
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                # Convert dict to columnar format (each key becomes a column with single value)
                feature_dict = {k: [v] for k, v in features.items()}
                logger.debug(f"Stage 4: Creating Arrow table with {len(feature_dict)} columns")
                table = pa.Table.from_pydict(feature_dict)
                logger.debug(f"Stage 4: Arrow table created: {table.num_rows} rows, {table.num_columns} columns")
                logger.debug(f"Stage 4: Writing Parquet file...")
                pq.write_table(table, str(feature_path), compression='snappy')
                logger.debug(f"Stage 4: Parquet file written successfully: {feature_path.exists()}, size: {feature_path.stat().st_size if feature_path.exists() else 'N/A'} bytes")
            except ImportError:
                # Fallback to numpy if pyarrow not available
                logger.warning("Stage 4: PyArrow not available, falling back to .npy format")
                feature_path = output_dir / f"{video_id}_scaled_features.npy"
                logger.debug(f"Stage 4: Saving as .npy to: {feature_path}")
                np.save(str(feature_path), features)
                logger.debug(f"Stage 4: .npy file written: {feature_path.exists()}")
            
            # Create metadata row
            feature_row = {
                "video_path": video_rel,
                "label": label,
                "feature_path": str(feature_path.relative_to(project_root)),
            }
            feature_row.update(features)  # Add all feature values
            logger.debug(f"Stage 4: Created metadata row with {len(feature_row)} columns")
            logger.debug(f"Stage 4: Metadata row columns: {list(feature_row.keys())[:10]}..." if len(feature_row) > 10 else f"Stage 4: Metadata row columns: {list(feature_row.keys())}")
            feature_rows.append(feature_row)
            processed_count += 1
            
            logger.info(f"Stage 4: [{iteration_count}/{df.height}] ✓ Completed {Path(video_path).name} - {len(features)} features extracted")
            if iteration_count % 10 == 0:
                logger.info(f"Stage 4: Progress: {idx+1}/{df.height} videos processed ({processed_count} new, {skipped_count} skipped)")
            
            aggressive_gc(clear_cuda=False)
            
        except Exception as e:
            logger.error(f"Error processing {video_rel}: {e}", exc_info=True)
            continue
    
    logger.info(f"Stage 4: Processed {processed_count} videos, skipped {skipped_count} videos, corrupted {corrupted_count} videos")
    
    # Log corrupted files summary if any and write to file
    if corrupted_count > 0:
        logger.warning(f"Stage 4: Found {corrupted_count} corrupted/invalid video files:")
        for corrupted in corrupted_files[:10]:  # Show first 10
            logger.warning(f"  - {corrupted['video_path']}: {corrupted['error']}")
        if corrupted_count > 10:
            logger.warning(f"  ... and {corrupted_count - 10} more corrupted files")
        
        # Write corrupted files to data folder for repair
        corrupted_log_path = project_root / "data" / "corrupted_scaled_videos.txt"
        corrupted_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Append mode: open file and append each corrupted file (one per line)
            with open(corrupted_log_path, 'a', encoding='utf-8') as f:
                for corrupted in corrupted_files:
                    # Write relative path (from metadata) for easier lookup
                    f.write(f"{corrupted['video_path']}\n")
            logger.info(f"Stage 4: Appended {corrupted_count} corrupted file(s) to {corrupted_log_path}")
        except Exception as e:
            logger.error(f"Stage 4: Failed to write corrupted files log: {e}")
    
    if not feature_rows:
        logger.warning("Stage 4: No new features extracted! (may have all been skipped)")
        # Return existing metadata if available
        if existing_metadata is not None:
            logger.info("Stage 4: Returning existing metadata")
            return existing_metadata
        return pl.DataFrame()
    
    # Merge new metadata_rows with existing metadata and write final file
    logger.info("=" * 80)
    logger.info("Stage 4: Merging new scaled features with existing metadata...")
    logger.info("=" * 80)
    
    # Create DataFrame from new feature_rows
    new_features_df = pl.DataFrame(feature_rows) if feature_rows else pl.DataFrame()
    
    if new_features_df.height > 0:
        logger.info(f"New scaled features to add: {new_features_df.height} entries")
        
        # Merge with existing metadata (avoid duplicates)
        if existing_metadata is not None and existing_metadata.height > 0 and not delete_existing:
            # Create a set of existing entries to avoid duplicates
            existing_keys = set()
            for row in existing_metadata.iter_rows(named=True):
                key = row.get("video_path", "")
                existing_keys.add(key)
            
            # Filter out duplicates from new entries
            new_entries_filtered = []
            for row in feature_rows:
                key = row.get("video_path", "")
                if key not in existing_keys:
                    new_entries_filtered.append(row)
            
            if new_entries_filtered:
                new_features_df = pl.DataFrame(new_entries_filtered)
                logger.info(f"After deduplication: {new_features_df.height} new entries to add")
                combined_features_df = pl.concat([existing_metadata, new_features_df])
            else:
                logger.info("All new entries already exist in metadata, no merge needed")
                combined_features_df = existing_metadata
        else:
            combined_features_df = new_features_df
        
        # Write final metadata file (prefer Arrow, fallback to Parquet, then CSV)
        logger.info(f"Writing final metadata file with {combined_features_df.height} total entries...")
        
        # Try Arrow first
        final_metadata_path = output_dir / "features_scaled_metadata.arrow"
        success = write_metadata_atomic(combined_features_df, final_metadata_path, append=False)
        
        if not success:
            # Fallback to Parquet
            final_metadata_path = output_dir / "features_scaled_metadata.parquet"
            success = write_metadata_atomic(combined_features_df, final_metadata_path, append=False)
            if success:
                logger.info(f"✓ Saved metadata as Parquet: {final_metadata_path}")
        else:
            logger.info(f"✓ Saved metadata as Arrow IPC: {final_metadata_path}")
        
        if not success:
            # Final fallback to CSV
            final_metadata_path = output_dir / "features_scaled_metadata.csv"
            try:
                combined_features_df.write_csv(final_metadata_path)
                logger.info(f"✓ Saved metadata as CSV: {final_metadata_path}")
                success = True
            except Exception as e:
                logger.error(f"Failed to save metadata as CSV: {e}")
                success = False
        
        # Remove old metadata files if format changed
        metadata_paths_to_check = [
            output_dir / "features_scaled_metadata.arrow",
            output_dir / "features_scaled_metadata.parquet",
            output_dir / "features_scaled_metadata.csv"
        ]
        for old_path in metadata_paths_to_check:
            if old_path != final_metadata_path and old_path.exists():
                try:
                    old_path.unlink()
                    logger.debug(f"Removed old metadata file: {old_path}")
                except (OSError, PermissionError, FileNotFoundError):
                    pass
        
        if success:
            logger.info(f"✓ Final metadata file written: {final_metadata_path}")
            logger.info(f"  Total entries: {combined_features_df.height}")
            original_count = existing_metadata.height if existing_metadata is not None else 0
            new_count = len(new_entries_filtered) if 'new_entries_filtered' in locals() and new_entries_filtered else new_features_df.height
            logger.info(f"  Original entries: {original_count}")
            logger.info(f"  New entries added: {new_count}")
            logger.info(f"✓ Stage 4: Extracted features from {processed_count} videos, skipped {skipped_count} videos")
            return combined_features_df
        else:
            logger.error("Failed to write final metadata file!")
            # Try to return existing metadata
            if existing_metadata is not None:
                return existing_metadata
            return pl.DataFrame()
    else:
        logger.info("✓ Stage 4 complete: No new features to save (all may have been skipped)")
        # Reload final metadata file to return complete dataset
        metadata_paths_to_check = [
            output_dir / "features_scaled_metadata.arrow",
            output_dir / "features_scaled_metadata.parquet",
            output_dir / "features_scaled_metadata.csv"
        ]
        for metadata_path in metadata_paths_to_check:
            if metadata_path.exists():
                try:
                    final_metadata = load_metadata_flexible(str(metadata_path), max_retries=3, retry_delay=0.5)
                    if final_metadata is not None:
                        logger.info(f"Stage 4: Returning complete metadata from {metadata_path} ({final_metadata.height} entries)")
                        return final_metadata
                except Exception as e:
                    logger.debug(f"Could not load metadata from {metadata_path}: {e}")
        
        # Fallback to existing_metadata if available
        if existing_metadata is not None:
            logger.info("Stage 4: Returning existing metadata")
            return existing_metadata
        return pl.DataFrame()

