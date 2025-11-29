"""
MLOps Utilities: Memory management, OOM handling, and checkpointing helpers.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def aggressive_gc(clear_cuda: bool = True) -> None:
    """
    Perform ULTRA aggressive garbage collection for maximum memory cleanup.
    
    Args:
        clear_cuda: If True, also clear CUDA cache
    """
    # Ultra aggressive: Multiple GC passes with different thresholds
    # First pass: collect everything
    for _ in range(5):
        gc.collect()
    
    # Second pass: collect with lower threshold (more aggressive)
    import gc as gc_module
    old_thresholds = gc_module.get_threshold()
    gc_module.set_threshold(1, 1, 1)  # Most aggressive: collect immediately
    for _ in range(3):
        gc.collect()
    gc_module.set_threshold(*old_thresholds)  # Restore original thresholds
    
    # Final pass: standard collection
    for _ in range(2):
        gc.collect()
    
    # Clear CUDA cache if available (multiple times for thorough cleanup)
    if clear_cuda and torch.cuda.is_available():
        for _ in range(3):
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Final cache clear
        torch.cuda.empty_cache()
    
    logger.debug("Ultra aggressive GC completed (CUDA cache cleared: %s)", clear_cuda)


def check_oom_error(error: Exception) -> bool:
    """Check if an error is an OOM (Out of Memory) error."""
    error_str = str(error).lower()
    oom_indicators = [
        "out of memory",
        "cuda out of memory",
        "oom",
        "memory allocation failed",
        "allocation failed",
    ]
    return any(indicator in error_str for indicator in oom_indicators)


def handle_oom_error(error: Exception, context: str = "") -> None:
    """
    Handle OOM error with aggressive cleanup.
    
    Args:
        error: The OOM exception
        context: Context string for logging
    """
    logger.error("OOM ERROR detected%s: %s", f" ({context})" if context else "", str(error))
    
    # Aggressive cleanup
    aggressive_gc(clear_cuda=True)
    
    # Log memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.error(
            "GPU Memory after cleanup: %.2f GB allocated, %.2f GB reserved, %.2f GB total",
            allocated, reserved, total
        )


def safe_execute(func, *args, oom_retry: bool = True, max_retries: int = 1, 
                 context: str = "", **kwargs) -> Any:
    """
    Safely execute a function with OOM error handling and retry.
    
    Args:
        func: Function to execute
        oom_retry: If True, retry once after OOM with aggressive cleanup
        max_retries: Maximum number of retries
        context: Context string for logging
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result of func execution
    
    Raises:
        RuntimeError: If execution fails after retries
    """
    retries = 0
    
    while retries <= max_retries:
        try:
            result = func(*args, **kwargs)
            return result
        
        except RuntimeError as e:
            if check_oom_error(e):
                handle_oom_error(e, context)
                
                if oom_retry and retries < max_retries:
                    retries += 1
                    logger.warning("Retrying after OOM cleanup (attempt %d/%d)", 
                                 retries, max_retries + 1)
                    aggressive_gc(clear_cuda=True)
                    continue
                else:
                    raise RuntimeError(
                        f"OOM error after {retries} retries. "
                        f"Consider reducing batch_size, num_frames, or model size."
                    ) from e
            else:
                raise
        
        except Exception as e:
            # For non-OOM errors, log and re-raise
            logger.error("Error in %s: %s", context or "safe_execute", str(e))
            raise
    
    raise RuntimeError(f"Failed after {max_retries} retries")


def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics."""
    stats = {}
    
    # CPU memory (if psutil available)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        stats['cpu_memory_mb'] = mem_info.rss / 1024 / 1024
        stats['cpu_memory_gb'] = mem_info.rss / 1024 / 1024 / 1024
        stats['cpu_vms_mb'] = mem_info.vms / 1024 / 1024
    except ImportError:
        pass
    
    # GPU memory
    if torch.cuda.is_available():
        stats['gpu_allocated_gb'] = torch.cuda.memory_allocated(0) / 1e9
        stats['gpu_reserved_gb'] = torch.cuda.memory_reserved(0) / 1e9
        stats['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        stats['gpu_free_gb'] = stats['gpu_total_gb'] - stats['gpu_reserved_gb']
    
    return stats


def get_detailed_memory_breakdown() -> Dict[str, Any]:
    """Get detailed memory breakdown by object type."""
    breakdown = {}
    
    try:
        import sys
        import gc
        
        # Count objects by type
        type_counts = {}
        type_sizes = {}
        
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
            # Estimate size
            try:
                size = sys.getsizeof(obj)
                type_sizes[obj_type] = type_sizes.get(obj_type, 0) + size
            except Exception:
                pass
        
        # Get top memory consumers
        top_types = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)[:20]
        
        breakdown['top_memory_consumers'] = {
            name: {
                'count': type_counts.get(name, 0),
                'total_size_mb': size / 1024 / 1024
            }
            for name, size in top_types
        }
        
        # Count Polars DataFrames
        try:
            import polars as pl
            df_count = sum(1 for obj in gc.get_objects() if isinstance(obj, pl.DataFrame))
            breakdown['polars_dataframes'] = df_count
        except Exception:
            pass
        
        # Count PyTorch tensors
        tensor_count = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))
        tensor_memory = sum(obj.numel() * obj.element_size() for obj in gc.get_objects() 
                          if isinstance(obj, torch.Tensor)) / 1024 / 1024
        breakdown['torch_tensors'] = {
            'count': tensor_count,
            'total_memory_mb': tensor_memory
        }
        
        # Count models
        model_count = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.nn.Module))
        breakdown['torch_models'] = model_count
        
    except Exception as e:
        breakdown['error'] = str(e)
    
    return breakdown


def log_memory_stats(context: str = "", detailed: bool = False) -> None:
    """Log current memory statistics."""
    stats = get_memory_stats()
    logger.info("Memory stats%s: %s", f" ({context})" if context else "", stats)
    
    if detailed:
        breakdown = get_detailed_memory_breakdown()
        logger.info("Detailed memory breakdown%s:", f" ({context})" if context else "")
        for key, value in breakdown.items():
            if key != 'error':
                logger.info("  %s: %s", key, value)
            else:
                logger.warning("  Error getting breakdown: %s", value)


__all__ = [
    "aggressive_gc",
    "check_oom_error",
    "handle_oom_error",
    "safe_execute",
    "get_memory_stats",
    "log_memory_stats",
    "get_detailed_memory_breakdown",
]

