#!/usr/bin/env python3
"""
New 5-Stage Pipeline Runner

Stage 1: Augment videos (10 augmentations per video) → 11N videos
Stage 2: Extract handcrafted features from all 11N videos → M features
Stage 3: Downscale videos → 11N downscaled videos
Stage 4: Extract additional features from downscaled videos → P features
Stage 5: Train models using downscaled videos + M + P features
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.augmentation import stage1_augment_videos
from lib.features import stage2_extract_features, stage4_extract_downscaled_features
from lib.downscaling import stage3_downscale_videos
from lib.training import stage5_train_models

# Setup extensive logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for extensive logs
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger("lib").setLevel(logging.DEBUG)
logging.getLogger("lib.augmentation").setLevel(logging.DEBUG)
logging.getLogger("lib.features").setLevel(logging.DEBUG)
logging.getLogger("lib.downscaling").setLevel(logging.DEBUG)
logging.getLogger("lib.training").setLevel(logging.DEBUG)
logging.getLogger("lib.data").setLevel(logging.DEBUG)
logging.getLogger("lib.models").setLevel(logging.DEBUG)
logging.getLogger("lib.utils").setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description="Run 5-stage pipeline")
    parser.add_argument("--project-root", type=str, default=os.getcwd(), help="Project root directory")
    parser.add_argument("--num-augmentations", type=int, default=10, help="Number of augmentations per video")
    parser.add_argument("--skip-stage", type=int, nargs="+", default=[], help="Stages to skip (1-5)")
    parser.add_argument("--only-stage", type=int, nargs="+", default=[], help="Only run these stages (1-5)")
    parser.add_argument("--model-types", type=str, nargs="+", default=["logistic_regression", "svm"], help="Models to train in Stage 5")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of k-fold splits")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root)
    
    logger.info("="*80)
    logger.info("NEW 5-STAGE PIPELINE")
    logger.info("="*80)
    logger.info("Project root: %s", project_root)
    logger.info("Number of augmentations: %d", args.num_augmentations)
    logger.info("Skip stages: %s", args.skip_stage if args.skip_stage else "None")
    logger.info("Only stages: %s", args.only_stage if args.only_stage else "All")
    logger.info("Model types: %s", args.model_types)
    logger.info("K-fold splits: %d", args.n_splits)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Working directory: %s", os.getcwd())
    logger.info("="*80)
    
    # Determine which stages to run
    stages_to_run = []
    if args.only_stage:
        stages_to_run = args.only_stage
    else:
        stages_to_run = [1, 2, 3, 4, 5]
        stages_to_run = [s for s in stages_to_run if s not in args.skip_stage]
    
    # Stage 1: Augmentation
    if 1 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: VIDEO AUGMENTATION")
        logger.info("="*80)
        logger.debug("Starting Stage 1 with %d augmentations per video", args.num_augmentations)
        logger.debug("Output directory: %s", project_root / "data" / "augmented_videos")
        import time
        stage1_start = time.time()
        stage1_df = stage1_augment_videos(
            project_root=str(project_root),
            num_augmentations=args.num_augmentations,
            output_dir="data/augmented_videos"
        )
        stage1_duration = time.time() - stage1_start
        logger.info("Stage 1 completed in %.2f seconds", stage1_duration)
        logger.debug("Stage 1 output shape: %s", stage1_df.shape if hasattr(stage1_df, 'shape') else "N/A")
        stage1_metadata_path = project_root / "data" / "augmented_videos" / "augmented_metadata.csv"
        logger.debug("Stage 1 metadata saved to: %s", stage1_metadata_path)
    else:
        logger.info("Skipping Stage 1")
        stage1_metadata_path = project_root / "data" / "augmented_videos" / "augmented_metadata.csv"
        if not stage1_metadata_path.exists():
            logger.error("Stage 1 metadata not found: %s", stage1_metadata_path)
            return
        logger.debug("Using existing Stage 1 metadata: %s", stage1_metadata_path)
    
    # Stage 2: Extract features from original videos
    if 2 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: EXTRACT HANDCRAFTED FEATURES (M features)")
        logger.info("="*80)
        logger.debug("Input metadata: %s", stage1_metadata_path)
        num_frames = 6  # Optimized for 80GB RAM
        logger.debug("Number of frames: %d", num_frames)
        logger.debug("Output directory: %s", project_root / "data" / "features_stage2")
        import time
        stage2_start = time.time()
        stage2_df = stage2_extract_features(
            project_root=str(project_root),
            augmented_metadata_path=str(stage1_metadata_path),
            num_frames=num_frames,
            output_dir="data/features_stage2"
        )
        stage2_duration = time.time() - stage2_start
        logger.info("Stage 2 completed in %.2f seconds", stage2_duration)
        logger.debug("Stage 2 output shape: %s", stage2_df.shape if hasattr(stage2_df, 'shape') else "N/A")
        stage2_metadata_path = project_root / "data" / "features_stage2" / "features_metadata.csv"
        logger.debug("Stage 2 metadata saved to: %s", stage2_metadata_path)
    else:
        logger.info("Skipping Stage 2")
        stage2_metadata_path = project_root / "data" / "features_stage2" / "features_metadata.csv"
        if not stage2_metadata_path.exists():
            logger.error("Stage 2 metadata not found: %s", stage2_metadata_path)
            return
        logger.debug("Using existing Stage 2 metadata: %s", stage2_metadata_path)
    
    # Stage 3: Downscale videos
    if 3 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: DOWNSCALE VIDEOS")
        logger.info("="*80)
        logger.debug("Input metadata: %s", stage1_metadata_path)
        logger.debug("Downscaling method: resolution")
        logger.debug("Target size: (224, 224)")
        logger.debug("Output directory: %s", project_root / "data" / "downscaled_videos")
        import time
        stage3_start = time.time()
        stage3_df = stage3_downscale_videos(
            project_root=str(project_root),
            augmented_metadata_path=str(stage1_metadata_path),
            output_dir="data/downscaled_videos",
            method="resolution",  # or "autoencoder"
            target_size=(224, 224)
        )
        stage3_duration = time.time() - stage3_start
        logger.info("Stage 3 completed in %.2f seconds", stage3_duration)
        logger.debug("Stage 3 output shape: %s", stage3_df.shape if hasattr(stage3_df, 'shape') else "N/A")
        stage3_metadata_path = project_root / "data" / "downscaled_videos" / "downscaled_metadata.csv"
        logger.debug("Stage 3 metadata saved to: %s", stage3_metadata_path)
    else:
        logger.info("Skipping Stage 3")
        stage3_metadata_path = project_root / "data" / "downscaled_videos" / "downscaled_metadata.csv"
        if not stage3_metadata_path.exists():
            logger.error("Stage 3 metadata not found: %s", stage3_metadata_path)
            return
        logger.debug("Using existing Stage 3 metadata: %s", stage3_metadata_path)
    
    # Stage 4: Extract features from downscaled videos
    if 4 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: EXTRACT FEATURES FROM DOWNSCALED VIDEOS (P features)")
        logger.info("="*80)
        logger.debug("Input metadata: %s", stage3_metadata_path)
        num_frames = 6  # Optimized for 80GB RAM
        logger.debug("Number of frames: %d", num_frames)
        logger.debug("Output directory: %s", project_root / "data" / "features_stage4")
        import time
        stage4_start = time.time()
        stage4_df = stage4_extract_downscaled_features(
            project_root=str(project_root),
            downscaled_metadata_path=str(stage3_metadata_path),
            num_frames=num_frames,
            output_dir="data/features_stage4"
        )
        stage4_duration = time.time() - stage4_start
        logger.info("Stage 4 completed in %.2f seconds", stage4_duration)
        logger.debug("Stage 4 output shape: %s", stage4_df.shape if hasattr(stage4_df, 'shape') else "N/A")
        stage4_metadata_path = project_root / "data" / "features_stage4" / "features_downscaled_metadata.csv"
        logger.debug("Stage 4 metadata saved to: %s", stage4_metadata_path)
    else:
        logger.info("Skipping Stage 4")
        stage4_metadata_path = project_root / "data" / "features_stage4" / "features_downscaled_metadata.csv"
        if not stage4_metadata_path.exists():
            logger.error("Stage 4 metadata not found: %s", stage4_metadata_path)
            return
        logger.debug("Using existing Stage 4 metadata: %s", stage4_metadata_path)
    
    # Stage 5: Training
    if 5 in stages_to_run:
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: TRAIN MODELS")
        logger.info("="*80)
        logger.debug("Downscaled metadata: %s", stage3_metadata_path)
        logger.debug("Features Stage 2: %s", stage2_metadata_path)
        logger.debug("Features Stage 4: %s", stage4_metadata_path)
        num_frames = 6  # Optimized for 80GB RAM
        logger.debug("Model types: %s", args.model_types)
        logger.debug("K-fold splits: %d", args.n_splits)
        logger.debug("Number of frames: %d", num_frames)
        logger.debug("Output directory: data/training_results")
        import time
        stage5_start = time.time()
        results = stage5_train_models(
            project_root=str(project_root),
            downscaled_metadata_path=str(stage3_metadata_path),
            features_stage2_path=str(stage2_metadata_path),
            features_stage4_path=str(stage4_metadata_path),
            model_types=args.model_types,
            n_splits=args.n_splits,
            num_frames=num_frames,
            output_dir="data/training_results",
            use_tracking=True
        )
        stage5_duration = time.time() - stage5_start
        logger.info("Stage 5 completed in %.2f seconds", stage5_duration)
        logger.debug("Training results saved to: %s", project_root / "data" / "training_results")
        logger.info("✓ Stage 5 complete: Training finished")
    else:
        logger.info("Skipping Stage 5")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info("Stages executed: %s", stages_to_run)
    logger.debug("All intermediate files and results saved to data/ directory")


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        logger.info("Intermediate results may be available in data/ directory")
        sys.exit(130)
    except Exception as e:
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", str(e))
        logger.error("Exception type: %s", type(e).__name__)
        logger.error("Full traceback:", exc_info=True)
        logger.error("=" * 80)
        sys.exit(1)

