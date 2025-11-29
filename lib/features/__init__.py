"""
Feature extraction module.

Provides:
- Handcrafted feature extractors (noise, DCT, blur, boundary, codec)
- Stage 2: Extract features from original videos
- Stage 4: Extract features from downscaled videos
"""

from .handcrafted import (
    extract_noise_residual,
    extract_dct_statistics,
    extract_blur_sharpness,
    extract_boundary_inconsistency,
    extract_codec_cues,
    extract_all_features,
    HandcraftedFeatureExtractor,
)
from .pipeline import stage2_extract_features
from .downscaled import stage4_extract_downscaled_features


__all__ = [
    # Extractors
    "extract_noise_residual",
    "extract_dct_statistics",
    "extract_blur_sharpness",
    "extract_boundary_inconsistency",
    "extract_codec_cues",
    "extract_all_features",
    "HandcraftedFeatureExtractor",
    # Stage pipelines
    "stage2_extract_features",
    "stage4_extract_downscaled_features",
]

