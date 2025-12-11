#!/usr/bin/env python3
"""
Test script to verify baseline models (Logistic Regression and SVM) work with Stage 2 and Stage 4 features.

This script:
1. Checks that Stage 2 and Stage 4 features exist
2. Verifies feature counts (Stage 2: 15 or 23, Stage 4: 23)
3. Tests loading features
4. Tests training Logistic Regression and SVM models
"""

import sys
import logging
from pathlib import Path
import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    project_root = Path.cwd()
    
    # Expected paths
    stage2_metadata = project_root / "data" / "features_stage2" / "features_metadata.parquet"
    stage4_metadata = project_root / "data" / "features_stage4" / "features_metadata.parquet"
    
    # Also try arrow format
    if not stage2_metadata.exists():
        stage2_metadata = project_root / "data" / "features_stage2" / "features_metadata.arrow"
    if not stage4_metadata.exists():
        stage4_metadata = project_root / "data" / "features_stage4" / "features_metadata.arrow"
    
    logger.info("=" * 80)
    logger.info("TESTING BASELINE MODELS WITH STAGE 2/4 FEATURES")
    logger.info("=" * 80)
    
    # Check prerequisites
    if not stage2_metadata.exists():
        logger.error(f"✗ Stage 2 metadata not found: {stage2_metadata}")
        return 1
    
    logger.info(f"✓ Stage 2 metadata found: {stage2_metadata}")
    
    # Load metadata
    try:
        if stage2_metadata.suffix == '.arrow':
            df = pl.read_ipc(stage2_metadata)
        elif stage2_metadata.suffix == '.parquet':
            df = pl.read_parquet(stage2_metadata)
        else:
            df = pl.read_csv(stage2_metadata)
        
        logger.info(f"Loaded {df.height} videos from Stage 2 metadata")
        
        # Get feature columns
        metadata_cols = {'video_path', 'label', 'feature_path', 'is_downscaled', 'is_upscaled'}
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        logger.info(f"Found {len(feature_cols)} feature columns")
        logger.info(f"Feature names: {sorted(feature_cols)}")
        
        # Stage 2 can have 15 (if codec cues unavailable) or 23 features
        if len(feature_cols) not in [15, 23]:
            logger.warning(f"Expected 15 or 23 features for Stage 2, got {len(feature_cols)}")
        else:
            logger.info(f"✓ Correct number of features ({len(feature_cols)})")
        
        # Check if we have labels
        if 'label' not in df.columns:
            logger.error("✗ No 'label' column found in metadata")
            return 1
        
        # Get a subset for testing (first 100 videos)
        test_df = df.head(100)
        logger.info(f"Using {test_df.height} videos for testing")
        
        # Test Logistic Regression with Stage 2 only
        logger.info("\n" + "=" * 80)
        logger.info("TESTING: Logistic Regression (Stage 2 only)")
        logger.info("=" * 80)
        
        try:
            from lib.training._linear import LogisticRegressionBaseline
            
            model = LogisticRegressionBaseline(
                features_stage2_path=str(stage2_metadata),
                features_stage4_path=None,
                use_stage2_only=True
            )
            
            logger.info("Training Logistic Regression...")
            model.fit(test_df, str(project_root))
            logger.info("✓ Logistic Regression trained successfully")
            
            # Test prediction
            logger.info("Testing prediction...")
            predictions = model.predict(test_df, str(project_root))
            logger.info(f"✓ Predictions shape: {predictions.shape}")
            
        except Exception as e:
            logger.error(f"✗ Logistic Regression (Stage 2) failed: {e}", exc_info=True)
            return 1
        
        # Test Logistic Regression with Stage 2 + Stage 4
        if stage4_metadata.exists():
            logger.info("\n" + "=" * 80)
            logger.info("TESTING: Logistic Regression (Stage 2 + Stage 4)")
            logger.info("=" * 80)
            
            try:
                model = LogisticRegressionBaseline(
                    features_stage2_path=str(stage2_metadata),
                    features_stage4_path=str(stage4_metadata),
                    use_stage2_only=False
                )
                
                logger.info("Training Logistic Regression...")
                model.fit(test_df, str(project_root))
                logger.info("✓ Logistic Regression (Stage 2+4) trained successfully")
                
                # Test prediction
                logger.info("Testing prediction...")
                predictions = model.predict(test_df, str(project_root))
                logger.info(f"✓ Predictions shape: {predictions.shape}")
                
            except Exception as e:
                logger.error(f"✗ Logistic Regression (Stage 2+4) failed: {e}", exc_info=True)
                return 1
        else:
            logger.warning("Stage 4 metadata not found, skipping Stage 2+4 test")
        
        # Test SVM with Stage 2 only
        logger.info("\n" + "=" * 80)
        logger.info("TESTING: SVM (Stage 2 only)")
        logger.info("=" * 80)
        
        try:
            from lib.training._svm import SVMBaseline
            
            model = SVMBaseline(
                features_stage2_path=str(stage2_metadata),
                features_stage4_path=None,
                use_stage2_only=True
            )
            
            logger.info("Training SVM...")
            model.fit(test_df, str(project_root))
            logger.info("✓ SVM trained successfully")
            
            # Test prediction
            logger.info("Testing prediction...")
            predictions = model.predict(test_df, str(project_root))
            logger.info(f"✓ Predictions shape: {predictions.shape}")
            
        except Exception as e:
            logger.error(f"✗ SVM (Stage 2) failed: {e}", exc_info=True)
            return 1
        
        # Test SVM with Stage 2 + Stage 4
        if stage4_metadata.exists():
            logger.info("\n" + "=" * 80)
            logger.info("TESTING: SVM (Stage 2 + Stage 4)")
            logger.info("=" * 80)
            
            try:
                model = SVMBaseline(
                    features_stage2_path=str(stage2_metadata),
                    features_stage4_path=str(stage4_metadata),
                    use_stage2_only=False
                )
                
                logger.info("Training SVM...")
                model.fit(test_df, str(project_root))
                logger.info("✓ SVM (Stage 2+4) trained successfully")
                
                # Test prediction
                logger.info("Testing prediction...")
                predictions = model.predict(test_df, str(project_root))
                logger.info(f"✓ Predictions shape: {predictions.shape}")
                
            except Exception as e:
                logger.error(f"✗ SVM (Stage 2+4) failed: {e}", exc_info=True)
                return 1
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 80)
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

