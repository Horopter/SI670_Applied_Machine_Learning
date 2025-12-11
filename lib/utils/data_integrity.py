"""
Data Integrity Validation System

Provides comprehensive data validation at pipeline boundaries to ensure:
- File existence and accessibility
- Data format correctness
- Schema validation
- Count consistency
- Referential integrity
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import polars as pl
import numpy as np

from .guardrails import validate_file_integrity, validate_directory, DataIntegrityError

logger = logging.getLogger(__name__)


class DataIntegrityChecker:
    """Comprehensive data integrity checker."""
    
    @staticmethod
    def validate_metadata_file(
        metadata_path: Path,
        required_columns: Optional[Set[str]] = None,
        min_rows: int = 1,
        max_rows: Optional[int] = None,
        allow_empty: bool = False
    ) -> Tuple[bool, str, Optional[pl.DataFrame]]:
        """
        Validate a metadata file.
        
        Returns:
            Tuple of (is_valid, error_message, dataframe)
        """
        # Check file exists and is readable
        is_valid, error = validate_file_integrity(metadata_path, must_exist=True, check_readable=True)
        if not is_valid:
            return False, error, None
        
        # Try to load
        try:
            if metadata_path.suffix == '.arrow':
                df = pl.read_ipc(metadata_path)
            elif metadata_path.suffix == '.parquet':
                df = pl.read_parquet(metadata_path)
            elif metadata_path.suffix == '.csv':
                df = pl.read_csv(metadata_path)
            else:
                return False, f"Unsupported file format: {metadata_path.suffix}", None
        except Exception as e:
            return False, f"Failed to load metadata file: {e}", None
        
        # Check row count
        num_rows = df.height
        if not allow_empty and num_rows == 0:
            return False, f"Metadata file is empty: {metadata_path}", None
        
        if num_rows < min_rows:
            return False, f"Insufficient rows: {num_rows} < {min_rows}", None
        
        if max_rows is not None and num_rows > max_rows:
            return False, f"Too many rows: {num_rows} > {max_rows}", None
        
        # Check required columns
        if required_columns:
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}", None
        
        return True, "OK", df
    
    @staticmethod
    def validate_feature_file(
        feature_path: Path,
        expected_feature_count: Optional[int] = None,
        min_size_bytes: int = 100
    ) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Validate a feature file.
        
        Returns:
            Tuple of (is_valid, error_message, features_array)
        """
        # Check file exists
        is_valid, error = validate_file_integrity(
            feature_path,
            min_size_bytes=min_size_bytes,
            must_exist=True,
            check_readable=True
        )
        if not is_valid:
            return False, error, None
        
        # Try to load
        try:
            if feature_path.suffix == '.npy':
                features = np.load(feature_path)
            elif feature_path.suffix == '.parquet':
                df = pl.read_parquet(feature_path)
                # Convert to numpy array
                features = df.to_numpy()
            else:
                return False, f"Unsupported feature file format: {feature_path.suffix}", None
        except Exception as e:
            return False, f"Failed to load feature file: {e}", None
        
        # Validate shape
        if features.size == 0:
            return False, f"Feature file is empty: {feature_path}", None
        
        # Check feature count if expected
        if expected_feature_count is not None:
            if len(features.shape) == 1:
                actual_count = features.shape[0]
            elif len(features.shape) == 2:
                actual_count = features.shape[1]
            else:
                return False, f"Unexpected feature shape: {features.shape}", None
            
            if actual_count != expected_feature_count:
                return False, f"Feature count mismatch: {actual_count} != {expected_feature_count}", None
        
        return True, "OK", features
    
    @staticmethod
    def validate_video_file(
        video_path: Path,
        min_size_bytes: int = 1000,
        check_readable: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate a video file.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file exists and is readable
        is_valid, error = validate_file_integrity(
            video_path,
            min_size_bytes=min_size_bytes,
            must_exist=True,
            check_readable=check_readable
        )
        if not is_valid:
            return False, error
        
        # Try to open with av (if available)
        try:
            import av
            container = av.open(str(video_path))
            if len(container.streams.video) == 0:
                container.close()
                return False, "No video stream found"
            container.close()
        except ImportError:
            # av not available - skip deep validation
            pass
        except Exception as e:
            return False, f"Video file validation failed: {e}"
        
        return True, "OK"
    
    @staticmethod
    def validate_stage_prerequisites(
        stage: int,
        project_root: Path,
        scaled_metadata_path: Optional[Path] = None,
        features_stage2_path: Optional[Path] = None,
        features_stage4_path: Optional[Path] = None,
        min_rows: int = 3000
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate prerequisites for a pipeline stage.
        
        Returns:
            Tuple of (all_valid, errors, validation_info)
        """
        errors = []
        validation_info = {}
        
        if stage >= 3:
            # Stage 3+ requires scaled metadata
            if scaled_metadata_path:
                is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
                    scaled_metadata_path,
                    required_columns={'video_path', 'label'},
                    min_rows=min_rows
                )
                if not is_valid:
                    errors.append(f"Stage 3 metadata: {error}")
                else:
                    validation_info['stage3_rows'] = df.height if df else 0
                    validation_info['stage3_valid'] = True
            else:
                errors.append("Stage 3 metadata path not provided")
        
        if stage >= 5:
            # Stage 5 may require Stage 2 features
            if features_stage2_path:
                is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
                    features_stage2_path,
                    min_rows=1,
                    allow_empty=False
                )
                if not is_valid:
                    errors.append(f"Stage 2 features: {error}")
                else:
                    validation_info['stage2_rows'] = df.height if df else 0
                    validation_info['stage2_valid'] = True
            
            # Stage 5 may require Stage 4 features
            if features_stage4_path:
                is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
                    features_stage4_path,
                    min_rows=1,
                    allow_empty=False
                )
                if not is_valid:
                    errors.append(f"Stage 4 features: {error}")
                else:
                    validation_info['stage4_rows'] = df.height if df else 0
                    validation_info['stage4_valid'] = True
        
        return len(errors) == 0, errors, validation_info
    
    @staticmethod
    def compute_file_hash(file_path: Path, algorithm: str = 'sha256') -> Optional[str]:
        """Compute hash of a file for integrity checking."""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return None
    
    @staticmethod
    def validate_data_consistency(
        metadata_df: pl.DataFrame,
        file_column: str,
        base_path: Path,
        file_suffix: str = '.mp4',
        check_existence: bool = True
    ) -> Tuple[bool, List[str], Dict[str, int]]:
        """
        Validate that files referenced in metadata actually exist.
        
        Returns:
            Tuple of (is_consistent, missing_files, stats)
        """
        if file_column not in metadata_df.columns:
            return False, [f"Column '{file_column}' not found in metadata"], {}
        
        missing_files = []
        stats = {
            'total_rows': metadata_df.height,
            'files_checked': 0,
            'files_found': 0,
            'files_missing': 0,
        }
        
        for row in metadata_df.iter_rows(named=True):
            file_path_str = row[file_column]
            stats['files_checked'] += 1
            
            if check_existence:
                # Try to resolve path
                file_path = base_path / file_path_str
                if not file_path.exists():
                    # Try with suffix
                    if not file_path_str.endswith(file_suffix):
                        file_path = base_path / f"{file_path_str}{file_suffix}"
                    
                    if not file_path.exists():
                        missing_files.append(file_path_str)
                        stats['files_missing'] += 1
                    else:
                        stats['files_found'] += 1
                else:
                    stats['files_found'] += 1
            else:
                stats['files_found'] += 1
        
        is_consistent = len(missing_files) == 0
        return is_consistent, missing_files, stats

