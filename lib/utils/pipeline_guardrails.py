"""
Pipeline-specific guardrails and validation.

Provides stage-specific validation and guardrails for the 5-stage pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import polars as pl

from .guardrails import (
    ResourceMonitor,
    ResourceLimits,
    RetryConfig,
    HealthCheckStatus,
    DataIntegrityError,
    ResourceExhaustedError,
    validate_file_integrity,
    validate_directory,
    guarded_execution,
)
from .data_integrity import DataIntegrityChecker

logger = logging.getLogger(__name__)


class PipelineGuardrails:
    """Guardrails specific to the 5-stage pipeline."""
    
    def __init__(
        self,
        project_root: Path,
        resource_limits: Optional[ResourceLimits] = None,
        strict_mode: bool = True
    ):
        self.project_root = Path(project_root).resolve()
        self.monitor = ResourceMonitor(resource_limits)
        self.strict_mode = strict_mode
    
    def validate_stage1_output(self) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate Stage 1 (augmentation) output."""
        errors = []
        info = {}
        
        # Check output directory
        aug_dir = self.project_root / "data" / "augmented_videos"
        is_valid, error = validate_directory(aug_dir, must_exist=True, must_be_writable=False)
        if not is_valid:
            errors.append(f"Augmentation directory: {error}")
        
        # Check metadata file
        metadata_paths = [
            aug_dir / "augmented_metadata.arrow",
            aug_dir / "augmented_metadata.parquet",
            aug_dir / "augmented_metadata.csv",
        ]
        
        metadata_found = False
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
                    metadata_path,
                    required_columns={'video_path', 'label'},
                    min_rows=1
                )
                if is_valid:
                    info['stage1_rows'] = df.height if df else 0
                    info['stage1_metadata_path'] = str(metadata_path)
                    metadata_found = True
                    break
                else:
                    errors.append(f"Stage 1 metadata: {error}")
        
        if not metadata_found:
            errors.append("Stage 1 metadata file not found")
        
        return len(errors) == 0, errors, info
    
    def validate_stage2_output(self) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate Stage 2 (features) output."""
        errors = []
        info = {}
        
        # Check output directory
        features_dir = self.project_root / "data" / "features_stage2"
        is_valid, error = validate_directory(features_dir, must_exist=True, must_be_writable=False)
        if not is_valid:
            errors.append(f"Features directory: {error}")
        
        # Check metadata file
        metadata_paths = [
            features_dir / "features_metadata.arrow",
            features_dir / "features_metadata.parquet",
            features_dir / "features_metadata.csv",
        ]
        
        metadata_found = False
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
                    metadata_path,
                    required_columns={'video_path', 'label'},
                    min_rows=1
                )
                if is_valid:
                    info['stage2_rows'] = df.height if df else 0
                    info['stage2_metadata_path'] = str(metadata_path)
                    
                    # Check feature count
                    metadata_cols = {'video_path', 'label', 'feature_path'}
                    feature_cols = [col for col in df.columns if col not in metadata_cols]
                    info['stage2_feature_count'] = len(feature_cols)
                    
                    metadata_found = True
                    break
                else:
                    errors.append(f"Stage 2 metadata: {error}")
        
        if not metadata_found:
            errors.append("Stage 2 metadata file not found")
        
        return len(errors) == 0, errors, info
    
    def validate_stage3_output(self) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate Stage 3 (scaling) output."""
        errors = []
        info = {}
        
        # Check output directory
        scaled_dir = self.project_root / "data" / "scaled_videos"
        is_valid, error = validate_directory(scaled_dir, must_exist=True, must_be_writable=False)
        if not is_valid:
            errors.append(f"Scaled videos directory: {error}")
        
        # Check metadata file
        metadata_paths = [
            scaled_dir / "scaled_metadata.arrow",
            scaled_dir / "scaled_metadata.parquet",
            scaled_dir / "scaled_metadata.csv",
        ]
        
        metadata_found = False
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
                    metadata_path,
                    required_columns={'video_path', 'label'},
                    min_rows=3000  # CRITICAL: Must have > 3000 rows
                )
                if is_valid:
                    info['stage3_rows'] = df.height if df else 0
                    info['stage3_metadata_path'] = str(metadata_path)
                    metadata_found = True
                    break
                else:
                    errors.append(f"Stage 3 metadata: {error}")
        
        if not metadata_found:
            errors.append("Stage 3 metadata file not found")
        
        return len(errors) == 0, errors, info
    
    def validate_stage4_output(self) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate Stage 4 (scaled features) output."""
        errors = []
        info = {}
        
        # Check output directory
        features_dir = self.project_root / "data" / "features_stage4"
        is_valid, error = validate_directory(features_dir, must_exist=True, must_be_writable=False)
        if not is_valid:
            errors.append(f"Scaled features directory: {error}")
        
        # Check metadata file
        metadata_paths = [
            features_dir / "features_scaled_metadata.arrow",
            features_dir / "features_scaled_metadata.parquet",
            features_dir / "features_scaled_metadata.csv",
        ]
        
        metadata_found = False
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
                    metadata_path,
                    required_columns={'video_path', 'label'},
                    min_rows=1
                )
                if is_valid:
                    info['stage4_rows'] = df.height if df else 0
                    info['stage4_metadata_path'] = str(metadata_path)
                    
                    # Check feature count
                    metadata_cols = {'video_path', 'label', 'feature_path', 'is_downscaled', 'is_upscaled'}
                    feature_cols = [col for col in df.columns if col not in metadata_cols]
                    info['stage4_feature_count'] = len(feature_cols)
                    
                    metadata_found = True
                    break
                else:
                    errors.append(f"Stage 4 metadata: {error}")
        
        if not metadata_found:
            errors.append("Stage 4 metadata file not found")
        
        return len(errors) == 0, errors, info
    
    def validate_stage5_prerequisites(
        self,
        model_types: List[str],
        scaled_metadata_path: Optional[str] = None,
        features_stage2_path: Optional[str] = None,
        features_stage4_path: Optional[str] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate all prerequisites for Stage 5."""
        errors = []
        info = {}
        
        # Stage 3 is REQUIRED for all models
        stage3_ok, stage3_errors, stage3_info = self.validate_stage3_output()
        if not stage3_ok:
            errors.extend([f"Stage 3: {e}" for e in stage3_errors])
        else:
            info.update(stage3_info)
        
        # Stage 2 is required for *_stage2 models
        if any("stage2" in m for m in model_types):
            stage2_ok, stage2_errors, stage2_info = self.validate_stage2_output()
            if not stage2_ok:
                errors.extend([f"Stage 2: {e}" for e in stage2_errors])
            else:
                info.update(stage2_info)
        
        # Stage 4 is required for *_stage2_stage4 models
        if any("stage2_stage4" in m for m in model_types):
            stage4_ok, stage4_errors, stage4_info = self.validate_stage4_output()
            if not stage4_ok:
                errors.extend([f"Stage 4: {e}" for e in stage4_errors])
            else:
                info.update(stage4_info)
        
        # Resource health check
        health = self.monitor.full_health_check(self.project_root)
        info['system_health'] = health.status.value
        info['system_health_message'] = health.message
        
        if health.status == HealthCheckStatus.CRITICAL:
            errors.append(f"System health: {health.message}")
        
        return len(errors) == 0, errors, info
    
    def preflight_check(self, stage: int, **kwargs) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Perform preflight check for a specific stage."""
        if stage == 1:
            return self.validate_stage1_output()
        elif stage == 2:
            return self.validate_stage2_output()
        elif stage == 3:
            return self.validate_stage3_output()
        elif stage == 4:
            return self.validate_stage4_output()
        elif stage == 5:
            return self.validate_stage5_prerequisites(
                model_types=kwargs.get('model_types', []),
                scaled_metadata_path=kwargs.get('scaled_metadata_path'),
                features_stage2_path=kwargs.get('features_stage2_path'),
                features_stage4_path=kwargs.get('features_stage4_path'),
            )
        else:
            return False, [f"Unknown stage: {stage}"], {}

