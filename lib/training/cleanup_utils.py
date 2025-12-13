"""
Shared utility functions for cleaning up models and freeing memory.

Eliminates duplicate cleanup code across training scripts.
"""

from __future__ import annotations

from typing import Any
import torch


def cleanup_model_and_memory(
    model: Any | None = None,
    device: torch.device | None = None,
    clear_cuda: bool = True
) -> None:
    """
    Clean up model and free memory.
    
    Args:
        model: Model object to delete (optional, will check locals() if None)
        device: PyTorch device (optional, for CUDA cache clearing)
        clear_cuda: Whether to clear CUDA cache if available
    """
    # Delete model if provided
    # NOTE: Cannot use locals() to delete variables - it doesn't work in Python
    # The caller should pass the model explicitly if they want it deleted
    if model is not None:
        try:
            del model
        except (AttributeError, RuntimeError):
            pass
    
    # Clear GPU cache if using CUDA
    if clear_cuda:
        try:
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (RuntimeError, AttributeError):
            pass


def cleanup_resources(
    model: Any | None = None,
    device: torch.device | None = None,
    mlflow_tracker: Any | None = None,
    clear_cuda: bool = True,
    aggressive_gc_func: callable | None = None
) -> None:
    """
    Comprehensive resource cleanup for training loops.
    
    Args:
        model: Model object to delete
        device: PyTorch device
        mlflow_tracker: MLflow tracker to end (optional)
        clear_cuda: Whether to clear CUDA cache
        aggressive_gc_func: Function to call for aggressive garbage collection
    """
    # End MLflow run if active
    if mlflow_tracker is not None:
        try:
            mlflow_tracker.end_run()
        except Exception:
            pass
    
    # Clean up model and memory
    cleanup_model_and_memory(model=model, device=device, clear_cuda=clear_cuda)
    
    # Aggressive garbage collection if function provided
    if aggressive_gc_func is not None:
        try:
            aggressive_gc_func(clear_cuda=clear_cuda)
        except Exception:
            pass
