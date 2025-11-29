"""
Utility functions module.

Provides:
- Memory management (aggressive GC, memory profiling)
- OOM handling and safe execution
- Video path resolution
- Video data loading and splitting
- Video metrics
"""

# Memory management
from .memory import (
    aggressive_gc,
    log_memory_stats,
    get_memory_stats,
    check_oom_error,
    handle_oom_error,
    safe_execute,
)

# Path utilities
from .paths import (
    resolve_video_path,
    get_video_path_candidates,
    check_video_path_exists,
)

# Metrics utilities
from .metrics import (
    collect_logits_and_labels,
    basic_classification_metrics,
    confusion_matrix,
    roc_auc,
)


__all__ = [
    # Memory
    "aggressive_gc",
    "log_memory_stats",
    "get_memory_stats",
    "check_oom_error",
    "handle_oom_error",
    "safe_execute",
    # Paths
    "resolve_video_path",
    "get_video_path_candidates",
    "check_video_path_exists",
    # Metrics
    "collect_logits_and_labels",
    "basic_classification_metrics",
    "confusion_matrix",
    "roc_auc",
]

