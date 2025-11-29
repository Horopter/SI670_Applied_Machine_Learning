"""
Data preparation and loading module.

Provides:
- Video index building
- Metadata parsing
- Video scanning
- Configuration
- CLI interface
- Data loading and splitting
"""

from .config import FVCConfig
from .index import build_video_index
from .metadata import parse_metadata
from .scan import scan_videos
from .cli import run_default_prep
from .loading import (
    load_metadata,
    filter_existing_videos,
    train_val_test_split,
    SplitConfig,
    stratified_kfold,
    maybe_limit_to_small_test_subset,
    make_balanced_batch_sampler,
)

__all__ = [
    # Data preparation
    "FVCConfig",
    "build_video_index",
    "parse_metadata",
    "scan_videos",
    "run_default_prep",
    # Data loading
    "load_metadata",
    "filter_existing_videos",
    "train_val_test_split",
    "SplitConfig",
    "stratified_kfold",
    "maybe_limit_to_small_test_subset",
    "make_balanced_batch_sampler",
]
