"""
Memory management utilities.

Provides:
- Aggressive garbage collection
- Memory profiling and logging
- OOM detection and handling
"""

from __future__ import annotations

import logging
import gc
from typing import Dict, Any
import torch
import psutil
import os

logger = logging.getLogger(__name__)


def aggressive_gc(clear_cuda: bool = True, threshold: int = 0) -> None:
    """
    Perform aggressive garbage collection with multiple passes and CUDA cache clearing.
    
    Args:
        clear_cuda: Whether to clear CUDA cache
        threshold: GC threshold (0 = collect all generations)
    """
    # HIGHLY AGGRESSIVE: Multiple passes of GC (doubled from 5 to 10 for maximum cleanup)
    # Note: gc.collect() doesn't accept threshold in Python < 3.10, so we don't use it
    for _ in range(10):
        collected = gc.collect()
        if collected == 0:
            break
    
    # Clear CUDA cache if requested (doubled passes for maximum cleanup)
    if clear_cuda and torch.cuda.is_available():
        # HIGHLY AGGRESSIVE: Multiple passes of cache clearing (doubled from 2 to 4) to ensure all memory is freed
        for _ in range(4):
            torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations complete before returning


def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory statistics.
    
    Returns:
        Dictionary with memory stats
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    stats = {
        "cpu_memory_mb": mem_info.rss / 1024 / 1024,
        "cpu_memory_gb": mem_info.rss / 1024 / 1024 / 1024,
        "cpu_vms_mb": mem_info.vms / 1024 / 1024,
    }
    
    # GPU stats if available
    if torch.cuda.is_available():
        stats.update({
            "gpu_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
            "gpu_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "gpu_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9,
        })
    else:
        stats.update({
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_total_gb": 0.0,
            "gpu_free_gb": 0.0,
        })
    
    return stats


def log_memory_stats(context: str = "", detailed: bool = False) -> None:
    """
    Log current memory statistics.
    
    Args:
        context: Context string for logging
        detailed: Whether to log detailed stats
    """
    stats = get_memory_stats()
    
    if context:
        logger.info(f"Memory stats ({context}): {stats}")
    else:
        logger.info(f"Memory stats: {stats}")
    
    if detailed:
        process = psutil.Process(os.getpid())
        logger.debug(f"CPU percent: {process.cpu_percent()}%")
        logger.debug(f"Num threads: {process.num_threads()}")


def check_oom_error(error: Exception) -> bool:
    """
    Check if error is an OOM error.
    
    Args:
        error: Exception to check
    
    Returns:
        True if OOM error, False otherwise
    """
    error_str = str(error).lower()
    oom_indicators = [
        "out of memory",
        "cuda out of memory",
        "oom",
        "allocation failed",
        "memory allocation",
    ]
    return any(indicator in error_str for indicator in oom_indicators)


def handle_oom_error(error: Exception, context: str = "") -> None:
    """
    Handle OOM error by performing cleanup.
    
    Args:
        error: OOM exception
        context: Context string for logging
    """
    logger.error(f"OOM error {context}: {error}")
    
    # Aggressive cleanup
    aggressive_gc(clear_cuda=True)
    
    # Log memory stats
    log_memory_stats(f"after OOM {context}", detailed=True)


def safe_execute(
    func,
    *args,
    oom_retry: bool = True,
    max_retries: int = 1,
    context: str = "",
    **kwargs
) -> Any:
    """
    Execute function with OOM error handling and retry logic.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        oom_retry: Whether to retry on OOM
        max_retries: Maximum number of retries
        context: Context string for logging
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if check_oom_error(e) and oom_retry and attempt < max_retries:
                handle_oom_error(e, context)
                logger.info(f"Retrying {context} (attempt {attempt + 1}/{max_retries})")
                continue
            else:
                raise
    
    raise RuntimeError(f"Failed after {max_retries} retries")

