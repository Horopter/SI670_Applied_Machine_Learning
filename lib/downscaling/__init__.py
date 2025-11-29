"""
Video downscaling module.

Provides:
- Resolution-based downscaling (letterbox resize)
- Autoencoder-based downscaling (optional)
- Stage 3: Downscale all videos
"""

from .methods import (
    letterbox_resize,
    downscale_video_frames,
)
from .pipeline import (
    downscale_video,
    stage3_downscale_videos,
)


__all__ = [
    # Methods
    "letterbox_resize",
    "downscale_video_frames",
    # Stage 3
    "downscale_video",
    "stage3_downscale_videos",
]

