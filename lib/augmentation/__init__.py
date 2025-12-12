"""
Video augmentation module.

Provides:
- Spatial and temporal augmentation transforms
- Video I/O utilities (frame loading/saving)
- Pre-generation pipeline for augmented clips
- Stage 1 augmentation pipeline
"""

from .io import load_frames, save_frames
from .transforms import (
    RandomRotation,
    RandomAffine,
    RandomGaussianNoise,
    RandomGaussianBlur,
    RandomCutout,
    LetterboxResize,
    apply_simple_augmentation,
    temporal_frame_drop,
    temporal_frame_duplicate,
    temporal_reverse,
)
# Stage 1 pipeline - always available (doesn't require pregenerate)
from .pipeline import stage1_augment_videos

# Pregenerate imports are optional - only import if needed
# These are used for pre-generation pipeline, not Stage 1 augmentation
# Import lazily to avoid breaking Stage 1 if lib.models is unavailable
# NOTE: apply_temporal_augmentations is defined in .transforms, not .pregenerate
# pregenerate.py imports it from .transforms and re-exports it
try:
    from .pregenerate import (
        generate_augmented_clips,
        pregenerate_augmented_dataset,
        load_precomputed_clip,
        build_comprehensive_frame_transforms,
    )
    # Import apply_temporal_augmentations directly from transforms (source of truth)
    from .transforms import apply_temporal_augmentations
except (ImportError, ModuleNotFoundError):
    # If pregenerate can't be imported (e.g., missing lib.models), 
    # Stage 1 will still work. These functions will be None.
    generate_augmented_clips = None
    pregenerate_augmented_dataset = None
    load_precomputed_clip = None
    build_comprehensive_frame_transforms = None
    # Try to import apply_temporal_augmentations directly even if pregenerate fails
    try:
        from .transforms import apply_temporal_augmentations
    except ImportError:
        apply_temporal_augmentations = None

__all__ = [
    # Transforms
    "build_comprehensive_frame_transforms",
    "apply_temporal_augmentations",
    "RandomRotation",
    "RandomAffine",
    "RandomGaussianNoise",
    "RandomGaussianBlur",
    "RandomCutout",
    "LetterboxResize",
    "apply_simple_augmentation",
    "temporal_frame_drop",
    "temporal_frame_duplicate",
    "temporal_reverse",
    # I/O
    "load_frames",
    "save_frames",
    # Pre-generation
    "generate_augmented_clips",
    "pregenerate_augmented_dataset",
    "load_precomputed_clip",
    # Stage 1
    "stage1_augment_videos",
]

