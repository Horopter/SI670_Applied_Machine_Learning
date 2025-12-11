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
    find_metadata_file,
    load_metadata_flexible,
    write_metadata_atomic,
)

# Video metadata cache
from .video_cache import (
    get_video_metadata,
    get_video_metadata_hash,
    clear_cache,
)

# Metrics utilities
from .metrics import (
    collect_logits_and_labels,
    basic_classification_metrics,
    confusion_matrix,
    roc_auc,
)

# Guardrails and data integrity
from .guardrails import (
    ResourceMonitor,
    ResourceLimits,
    RetryConfig,
    TimeoutHandler,
    HealthCheckStatus,
    HealthCheckResult,
    GuardrailError,
    ResourceExhaustedError,
    TimeoutError as GuardrailTimeoutError,
    DataIntegrityError,
    retry_with_backoff,
    guarded_execution,
    guarded_decorator,
    resource_guard,
    validate_file_integrity,
    validate_directory,
)

from .data_integrity import (
    DataIntegrityChecker,
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
    "find_metadata_file",
    "load_metadata_flexible",
    "write_metadata_atomic",
    # Video cache
    "get_video_metadata",
    "get_video_metadata_hash",
    "clear_cache",
    # Metrics
    "collect_logits_and_labels",
    "basic_classification_metrics",
    "confusion_matrix",
    "roc_auc",
    # Guardrails
    "ResourceMonitor",
    "ResourceLimits",
    "RetryConfig",
    "TimeoutHandler",
    "HealthCheckStatus",
    "HealthCheckResult",
    "GuardrailError",
    "ResourceExhaustedError",
    "GuardrailTimeoutError",
    "DataIntegrityError",
    "retry_with_backoff",
    "guarded_execution",
    "guarded_decorator",
    "resource_guard",
    "validate_file_integrity",
    "validate_directory",
    # Data integrity
    "DataIntegrityChecker",
]

