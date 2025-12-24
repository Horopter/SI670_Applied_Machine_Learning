#!/usr/bin/env python3
"""
Sanity check for Stage 2 and Stage 4 features before Stage 5 training.

Validates that:
- Stage 2 feature metadata exists and has valid data
- Stage 4 feature metadata exists and has valid data (if needed)
- Feature files are accessible
- Minimum data requirements are met
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path before importing lib
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared get_project_root function
from src.notebooks.notebook_utils import get_project_root

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_features_metadata(metadata_path: Optional[str], stage_name: str, min_rows: int = 100) -> bool:
    """
    Check if feature metadata exists and has valid data.
    
    Args:
        metadata_path: Path to feature metadata file (can be None)
        stage_name: Name of stage (e.g., "Stage 2", "Stage 4")
        min_rows: Minimum number of rows required
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if metadata_path:
            logger.info(f"Checking {stage_name} features metadata: {metadata_path}")
        else:
            logger.info(f"Checking {stage_name} features metadata...")
        
        if not metadata_path or not Path(metadata_path).exists():
            logger.warning(f"  ⚠ {stage_name} metadata file not found: {metadata_path}")
            return False
        
        # Try to load metadata using polars directly (simpler, no torch dependency)
        try:
            import polars as pl
            
            # Try different formats
            df = None
            path_obj = Path(metadata_path)
            
            if path_obj.suffix == '.parquet':
                df = pl.read_parquet(metadata_path)
            elif path_obj.suffix == '.arrow':
                df = pl.read_ipc(metadata_path)
            elif path_obj.suffix == '.csv':
                df = pl.read_csv(metadata_path)
            else:
                # Try all formats
                for ext in ['.parquet', '.arrow', '.csv']:
                    alt_path = path_obj.with_suffix(ext)
                    if alt_path.exists():
                        if ext == '.parquet':
                            df = pl.read_parquet(str(alt_path))
                        elif ext == '.arrow':
                            df = pl.read_ipc(str(alt_path))
                        elif ext == '.csv':
                            df = pl.read_csv(str(alt_path))
                        break
            
            if df is None:
                logger.warning(f"  ⚠ {stage_name} metadata could not be loaded: {metadata_path}")
                return False
            
            if df.height == 0:
                logger.warning(f"  ⚠ {stage_name} metadata is empty (0 rows): {metadata_path}")
                return False
            
            if df.height < min_rows:
                logger.warning(
                    f"  ⚠ {stage_name} metadata has only {df.height} rows "
                    f"(minimum {min_rows} recommended): {metadata_path}"
                )
                # Don't fail, just warn
            
            logger.info(f"  ✓ {stage_name} metadata valid: {df.height} feature rows")
            logger.info(f"    Columns: {len(df.columns)}")
            logger.info(f"    Sample columns: {list(df.columns[:5])}")
            
            # Explicitly clear DataFrame to avoid cleanup issues
            del df
            import gc
            gc.collect()
            
            return True
            
        except ImportError:
            # Fallback to lib.utils.paths if polars import fails
            from lib.utils.paths import load_metadata_flexible
            df = load_metadata_flexible(metadata_path)
            
            if df is None:
                logger.warning(f"  ⚠ {stage_name} metadata could not be loaded: {metadata_path}")
                return False
            
            if df.height == 0:
                logger.warning(f"  ⚠ {stage_name} metadata is empty (0 rows): {metadata_path}")
                return False
            
            if df.height < min_rows:
                logger.warning(
                    f"  ⚠ {stage_name} metadata has only {df.height} rows "
                    f"(minimum {min_rows} recommended): {metadata_path}"
                )
            
            logger.info(f"  ✓ {stage_name} metadata valid: {df.height} feature rows")
            logger.info(f"    Columns: {len(df.columns)}")
            logger.info(f"    Sample columns: {list(df.columns[:5])}")
            
            # Explicitly clear DataFrame to avoid cleanup issues
            del df
            import gc
            gc.collect()
            
            return True
        
    except Exception as e:
        logger.error(f"  ✗ Error checking {stage_name} metadata: {e}", exc_info=True)
        return False


def main() -> int:
    """Run sanity check for Stage 2 and Stage 4 features."""
    logger.info("=" * 80)
    logger.info("Feature Sanity Check")
    logger.info("=" * 80)
    
    # Get project root using shared function
    project_root = get_project_root()
    logger.info(f"Project root: {project_root}")
    
    # Default paths - use correct metadata file names
    # Stage 2: features_metadata.arrow/parquet/csv
    # Stage 4: features_scaled_metadata.arrow/parquet/csv
    features_stage2_dir = Path(project_root) / "data" / "features_stage2"
    features_stage4_dir = Path(project_root) / "data" / "features_stage4"
    
    # Try to use find_metadata_file utility if available
    try:
        from lib.utils.paths import find_metadata_file
        # Stage 2: look for features_metadata
        features_stage2_path = find_metadata_file(features_stage2_dir, "features_metadata")
        # Stage 4: look for features_scaled_metadata
        features_stage4_path = find_metadata_file(features_stage4_dir, "features_scaled_metadata")
    except ImportError:
        # Fallback: manually check for files
        features_stage2_path = None
        features_stage4_path = None
        
        # Stage 2: Check for features_metadata in various formats
        for ext in ['.arrow', '.parquet', '.csv']:
            test_path = features_stage2_dir / f"features_metadata{ext}"
            if test_path.exists():
                features_stage2_path = test_path
                break
        
        # Stage 4: Check for features_scaled_metadata in various formats
        for ext in ['.arrow', '.parquet', '.csv']:
            test_path = features_stage4_dir / f"features_scaled_metadata{ext}"
            if test_path.exists():
                features_stage4_path = test_path
                break
        
        # If not found, set default paths for error reporting
        if features_stage2_path is None:
            features_stage2_path = features_stage2_dir / "features_metadata.arrow"
        if features_stage4_path is None:
            features_stage4_path = features_stage4_dir / "features_scaled_metadata.arrow"
    
    logger.info("")
    logger.info("Checking Stage 2 features (required for baseline models)...")
    # check_features_metadata handles None paths gracefully
    stage2_valid = check_features_metadata(
        str(features_stage2_path) if features_stage2_path else None,
        "Stage 2",
        min_rows=100
    )
    
    logger.info("")
    logger.info("Checking Stage 4 features (required for stage2_stage4 models)...")
    # check_features_metadata handles None paths gracefully
    stage4_valid = check_features_metadata(
        str(features_stage4_path) if features_stage4_path else None,
        "Stage 4",
        min_rows=100
    )
    
    logger.info("")
    logger.info("=" * 80)
    
    # Determine overall status
    if stage2_valid and stage4_valid:
        logger.info("✅ Feature sanity check PASSED")
        logger.info("  - Stage 2 features: Valid")
        logger.info("  - Stage 4 features: Valid")
        return 0
    elif stage2_valid:
        logger.info("⚠️  Feature sanity check PASSED (with warnings)")
        logger.info("  - Stage 2 features: Valid")
        logger.info("  - Stage 4 features: Missing or invalid (may be OK for stage2-only models)")
        return 0  # Don't fail if Stage 4 is missing - some models don't need it
    else:
        logger.error("❌ Feature sanity check FAILED")
        logger.error("  - Stage 2 features: Missing or invalid (REQUIRED)")
        logger.error("  - Stage 4 features: Missing or invalid")
        logger.error("")
        logger.error("Please ensure Stage 2 feature extraction completed successfully.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        # Flush all output before exit
        sys.stdout.flush()
        sys.stderr.flush()
        # Use os._exit to bypass Python cleanup that might cause crashes
        import os
        os._exit(exit_code)
    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        sys.stdout.flush()
        sys.stderr.flush()
        import os
        os._exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.stdout.flush()
        sys.stderr.flush()
        import os
        os._exit(1)
