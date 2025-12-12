"""Parse FVC metadata CSVs and map to binary labels"""
import os
import re
import logging
from typing import Tuple, Optional
import pandas as pd
from .config import FVCConfig

logger = logging.getLogger(__name__)

LABEL_COLUMN_CANDIDATES = ["label", "class", "truth_label", "is_fake", "Label", "Class"]
VIDEO_URL_COLUMN_CANDIDATES = ["video_url", "url", "videoUrl", "Video_URL", "URL"]
CASCADE_ID_COLUMN_CANDIDATES = ["cascade_id", "cascadeId", "Cascade_ID", "group_id"]


def _find_column(columns, candidates):
    """Find first matching column from candidates list"""
    cols = [c for c in candidates if c in columns]
    if not cols:
        raise ValueError(f"None of {candidates} found in columns {list(columns)}")
    return cols[0]


def _extract_video_id_from_url(url: str) -> Optional[Tuple[str, str]]:
    """
    Extract video ID and platform from URL.
    
    Returns:
        Tuple of (video_id, platform) or None if cannot extract
    """
    if not isinstance(url, str):
        return None
    
    url = url.strip()
    
    # YouTube: https://www.youtube.com/watch?v=VIDEO_ID
    youtube_match = re.search(r'youtube\.com/watch\?v=([^&\s]+)', url)
    if youtube_match:
        return (youtube_match.group(1), "youtube")
    
    # YouTube short: https://youtu.be/VIDEO_ID
    youtube_short_match = re.search(r'youtu\.be/([^?\s]+)', url)
    if youtube_short_match:
        return (youtube_short_match.group(1), "youtube")
    
    # Twitter: extract tweet ID from various Twitter URL formats
    # Format: https://twitter.com/username/status/TWEET_ID
    twitter_match = re.search(r'twitter\.com/[^/]+/status/(\d+)', url)
    if twitter_match:
        return (twitter_match.group(1), "twitter")
    
    # Facebook: extract video ID from Facebook URLs
    # Format: https://www.facebook.com/page/videos/VIDEO_ID/
    facebook_match = re.search(r'facebook\.com/[^/]+/videos/(\d+)', url)
    if facebook_match:
        return (facebook_match.group(1), "facebook")
    
    return None


def _normalize_label(value) -> int:
    """Map textual labels to binary 0/1. Adjust mapping once you inspect the CSV."""
    if isinstance(value, str):
        v = value.strip().lower()
        # adjust these based on actual metadata
        if v in {"real", "authentic", "genuine", "0", "false"}:
            return 0
        if v in {"fake", "manipulated", "tampered", "1", "true"}:
            return 1
        # Handle additional labels from FVC_dup.csv
        # "uncertain", "debunked", "parody" - we'll skip these for now (return None)
        # or you could map them: uncertain/debunked/parody -> None (exclude from training)
    if isinstance(value, (int, float)):
        return int(value)
    # Return None for labels we can't map (will be filtered out)
    return None


def load_main_metadata(cfg: FVCConfig) -> pd.DataFrame:
    """Load and parse the main metadata CSV file"""
    path = os.path.join(cfg.metadata_dir, cfg.main_metadata_filename)
    
    # Try alternative names if main file doesn't exist
    if not os.path.exists(path):
        # Try common alternative names
        alternatives = [
            "FVC.csv",
            "metadata.csv",
            "FVC_metadata.csv",
            "labels.csv",
            "FVC_labels.csv"
        ]
        found = False
        for alt in alternatives:
            alt_path = os.path.join(cfg.metadata_dir, alt)
            if os.path.exists(alt_path):
                path = alt_path
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"Main metadata CSV not found at {path}. "
                f"Tried alternatives: {alternatives}. "
                f"Please place metadata CSV in {cfg.metadata_dir}/"
            )
    
    df = pd.read_csv(path)
    
    # Find video_url and label columns (FVC.csv has: cascade_id, video_url, label)
    url_col = _find_column(df.columns, VIDEO_URL_COLUMN_CANDIDATES)
    label_col = _find_column(df.columns, LABEL_COLUMN_CANDIDATES)
    
    df = df.rename(columns={url_col: "video_url", label_col: "raw_label"})
    df["label"] = df["raw_label"].apply(_normalize_label)
    
    # Extract video_id and platform from URLs
    extracted = df["video_url"].apply(_extract_video_id_from_url)
    df["video_id"] = extracted.apply(lambda x: x[0] if x else None)
    df["platform"] = extracted.apply(lambda x: x[1] if x else None)
    
    # Optional: detect subset from metadata if present
    if "subset" in df.columns:
        df["subset"] = df["subset"]
    else:
        df["subset"] = None
    
    # Drop rows where we couldn't extract video_id
    before_drop = len(df)
    df = df.dropna(subset=["video_id"])
    after_drop = len(df)
    if before_drop > after_drop:
        logger.warning(f"Dropped {before_drop - after_drop} rows where video_id could not be extracted from URL")
    
    return df[["video_id", "label", "subset", "platform"]]


def load_duplicates(cfg: FVCConfig) -> pd.DataFrame:
    """Load duplicates CSV if available. Also extracts labels from duplicates."""
    path = os.path.join(cfg.metadata_dir, cfg.dup_metadata_filename)
    
    if not os.path.exists(path):
        # no duplicates file, just return empty
        return pd.DataFrame(columns=["video_id", "dup_group", "label"])
    
    df = pd.read_csv(path)
    
    # FVC_dup.csv has: cascade_id, video_url, label
    # Extract video_id from URLs
    url_col = _find_column(df.columns, VIDEO_URL_COLUMN_CANDIDATES)
    cascade_col = _find_column(df.columns, CASCADE_ID_COLUMN_CANDIDATES)
    label_col = _find_column(df.columns, LABEL_COLUMN_CANDIDATES)
    
    df = df.rename(columns={url_col: "video_url", cascade_col: "cascade_id", label_col: "raw_label"})
    
    # Extract video_id and platform from URLs
    extracted = df["video_url"].apply(_extract_video_id_from_url)
    df["video_id"] = extracted.apply(lambda x: x[0] if x else None)
    df["platform"] = extracted.apply(lambda x: x[1] if x else None)
    
    # Normalize labels (will return None for uncertain/debunked/parody)
    df["label"] = df["raw_label"].apply(_normalize_label)
    
    # Use cascade_id as dup_group
    df["dup_group"] = df["cascade_id"]
    
    # Drop rows where we couldn't extract video_id
    df = df.dropna(subset=["video_id"])
    
    return df[["video_id", "dup_group", "label", "platform"]]


def parse_metadata(cfg: FVCConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse metadata CSVs and return DataFrames.
    
    Returns:
        main_df: video_id, label, subset, platform (from FVC.csv)
        dup_df: video_id, dup_group, label, platform (from FVC_dup.csv, also has labels)
    """
    main_df = load_main_metadata(cfg)
    dup_df = load_duplicates(cfg)
    return main_df, dup_df

