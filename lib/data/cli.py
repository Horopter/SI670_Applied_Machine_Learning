"""CLI entry point for FVC data preparation"""
import logging
from .config import FVCConfig
from .index import build_video_index

logger = logging.getLogger(__name__)


def run_default_prep():
    """Run default data preparation pipeline"""
    cfg = FVCConfig()
    
    logger.info("=" * 60)
    logger.info("FVC Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Root directory: {cfg.root_dir}")
    logger.info(f"Metadata directory: {cfg.metadata_dir}")
    logger.info(f"Data directory: {cfg.data_dir}")
    logger.info("")
    
    build_video_index(
        cfg,
        drop_duplicates=False,  # Keep all duplicate videos (grouped by dup_group for split awareness)
        compute_stats=True  # compute comprehensive video stats using ffprobe
    )


if __name__ == "__main__":
    run_default_prep()

