"""Build video index manifest by joining metadata with scanned videos"""
import os
import logging
from typing import Optional
import pandas as pd
from .config import FVCConfig
from .metadata import parse_metadata
from .scan import scan_videos

logger = logging.getLogger(__name__)


def build_video_index(
    cfg: FVCConfig,
    drop_duplicates: bool = True,
    compute_stats: bool = True
) -> pd.DataFrame:
    """
    Build video index manifest.
    
    Args:
        cfg: FVCConfig instance
        drop_duplicates: Whether to drop duplicate videos (keep first per group)
        compute_stats: Whether to compute video statistics
    
    Returns:
        DataFrame with video index
    """
    logger.info("Parsing metadata...")
    main_df, dup_df = parse_metadata(cfg)
    logger.info(f"Found {len(main_df)} videos in metadata")
    
    logger.info("Scanning video folders...")
    records = scan_videos(cfg, compute_stats=compute_stats)
    vids_df = pd.DataFrame.from_records(records)
    logger.info(f"Found {len(vids_df)} videos in folders")
    
    # Join on video_id - first with main metadata
    logger.info("Joining metadata with video paths...")
    merged = vids_df.merge(
        main_df,
        on="video_id",
        how="left",
        suffixes=("", "_main")
    )
    
    # If metadata subset/platform differs from folder structure, prefer folder structure
    if "subset_main" in merged.columns:
        merged["subset"] = merged["subset"].fillna(merged["subset_main"])
        merged = merged.drop(columns=["subset_main"])
    
    if "platform_main" in merged.columns:
        merged["platform"] = merged["platform"].fillna(merged["platform_main"])
        merged = merged.drop(columns=["platform_main"])
    
    # Merge duplicates - this also has labels!
    if not dup_df.empty:
        logger.info("Merging duplicate information and labels...")
        # Merge on video_id, but handle label conflicts
        merged = merged.merge(
            dup_df[["video_id", "dup_group", "label", "platform"]],
            on="video_id",
            how="left",
            suffixes=("", "_dup")
        )
        
        # Use label from main if available, otherwise use label from dup
        if "label_dup" in merged.columns:
            merged["label"] = merged["label"].fillna(merged["label_dup"])
            merged = merged.drop(columns=["label_dup"])
        
        # Use platform from dup if main doesn't have it
        if "platform_dup" in merged.columns:
            merged["platform"] = merged["platform"].fillna(merged["platform_dup"])
            merged = merged.drop(columns=["platform_dup"])
    
    # Decide how to handle duplicates: simplest is to drop all but the first per dup_group
    if drop_duplicates and "dup_group" in merged.columns:
        logger.info("Dropping duplicates...")
        initial_count = len(merged)
        merged = merged.sort_values(["dup_group", "video_id"])
        merged = merged.drop_duplicates(subset=["dup_group"], keep="first")
        dropped = initial_count - len(merged)
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate videos")
    
    # Sanity: drop rows with missing labels
    before_drop = len(merged)
    merged = merged.dropna(subset=["label"])
    after_drop = len(merged)
    if before_drop > after_drop:
        logger.warning(f"Dropped {before_drop - after_drop} videos with missing labels")
    
    # Ensure label is int
    merged["label"] = merged["label"].astype(int)
    
    # Reorder columns for readability
    col_order = [
        "subset",
        "platform",
        "video_id",
        "video_path",
        "label",
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_sec",
        "codec_name",
        "codec_long_name",
        "bitrate",
        "total_bitrate",
        "file_size_bytes",
        "pixel_format",
        "aspect_ratio",
        "dup_group",
    ]
    
    cols = [c for c in col_order if c in merged.columns] + \
           [c for c in merged.columns if c not in col_order]
    merged = merged[cols]
    
    # Write CSV and JSON
    csv_path = os.path.join(cfg.data_dir, "video_index_input.csv")
    json_path = os.path.join(cfg.data_dir, "video_index_input.json")
    
    logger.info(f"Writing manifest to {csv_path}...")
    merged.to_csv(csv_path, index=False)
    
    logger.info(f"Writing JSON manifest to {json_path}...")
    merged.to_json(json_path, orient="records", lines=False, indent=2)
    
    logger.info(f"\n✓ Successfully created manifest with {len(merged)} entries")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  JSON: {json_path}")
    
    # Log summary statistics
    logger.info("\nDataset Summary:")
    logger.info(f"  Total videos: {len(merged)}")
    if "label" in merged.columns:
        label_counts = merged["label"].value_counts().sort_index()
        logger.info(f"  Real (0): {label_counts.get(0, 0)}")
        logger.info(f"  Fake (1): {label_counts.get(1, 0)}")
    if "subset" in merged.columns:
        subset_counts = merged["subset"].value_counts()
        logger.info(f"  By subset: {dict(subset_counts)}")
    if "platform" in merged.columns:
        platform_counts = merged["platform"].value_counts()
        logger.info(f"  By platform: {dict(platform_counts)}")
    
    return merged

