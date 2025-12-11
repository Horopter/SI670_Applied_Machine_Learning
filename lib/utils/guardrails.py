"""
Comprehensive Guardrail System for Production Pipeline

This module provides production-grade guardrails including:
- Input validation
- Resource monitoring
- Timeout management
- Retry logic with exponential backoff
- Data integrity checks
- Health checks
- Error recovery
- Circuit breakers
"""

from __future__ import annotations

import os
import sys
import time
import signal
import logging
import functools
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple, TypeVar, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import psutil
import gc

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

T = TypeVar('T')


class GuardrailError(Exception):
    """Base exception for guardrail violations."""
    pass


class ResourceExhaustedError(GuardrailError):
    """Raised when system resources are exhausted."""
    pass


class TimeoutError(GuardrailError):
    """Raised when an operation exceeds timeout."""
    pass


class DataIntegrityError(GuardrailError):
    """Raised when data integrity checks fail."""
    pass


class HealthCheckStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ResourceLimits:
    """Resource limits for guardrails."""
    max_memory_gb: float = 200.0  # Maximum memory usage in GB
    max_disk_gb: float = 1000.0  # Maximum disk usage in GB
    max_cpu_percent: float = 95.0  # Maximum CPU usage percentage
    max_file_handles: int = 10000  # Maximum open file handles
    min_free_disk_gb: float = 50.0  # Minimum free disk space required
    max_gpu_memory_gb: float = 20.0  # Maximum GPU memory usage in GB


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    retryable_exceptions: Tuple[type, ...] = (
        OSError,
        IOError,
        ConnectionError,
        TimeoutError,
    )


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthCheckStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.process = psutil.Process(os.getpid())
    
    def check_memory(self) -> Tuple[bool, Dict[str, float]]:
        """Check memory usage against limits."""
        mem_info = self.process.memory_info()
        mem_gb = mem_info.rss / (1024 ** 3)
        
        system_mem = psutil.virtual_memory()
        system_mem_gb = system_mem.total / (1024 ** 3)
        system_mem_used_gb = system_mem.used / (1024 ** 3)
        system_mem_available_gb = system_mem.available / (1024 ** 3)
        
        metrics = {
            'process_memory_gb': mem_gb,
            'system_memory_total_gb': system_mem_gb,
            'system_memory_used_gb': system_mem_used_gb,
            'system_memory_available_gb': system_mem_available_gb,
            'system_memory_percent': system_mem.percent,
        }
        
        is_ok = (
            mem_gb < self.limits.max_memory_gb and
            system_mem_available_gb > self.limits.min_free_disk_gb
        )
        
        if not is_ok:
            logger.warning(
                f"Memory usage high: process={mem_gb:.2f}GB, "
                f"system_available={system_mem_available_gb:.2f}GB"
            )
        
        return is_ok, metrics
    
    def check_disk(self, path: Union[str, Path]) -> Tuple[bool, Dict[str, float]]:
        """Check disk usage for a given path."""
        path_obj = Path(path)
        if not path_obj.exists():
            return False, {'error': 'Path does not exist'}
        
        # Get disk usage for the partition containing this path
        disk_usage = psutil.disk_usage(str(path_obj))
        total_gb = disk_usage.total / (1024 ** 3)
        used_gb = disk_usage.used / (1024 ** 3)
        free_gb = disk_usage.free / (1024 ** 3)
        percent_used = (disk_usage.used / disk_usage.total) * 100
        
        metrics = {
            'total_gb': total_gb,
            'used_gb': used_gb,
            'free_gb': free_gb,
            'percent_used': percent_used,
        }
        
        is_ok = (
            free_gb > self.limits.min_free_disk_gb and
            used_gb < self.limits.max_disk_gb
        )
        
        if not is_ok:
            logger.warning(
                f"Disk usage high: free={free_gb:.2f}GB, "
                f"used={used_gb:.2f}GB ({percent_used:.1f}%)"
            )
        
        return is_ok, metrics
    
    def check_cpu(self) -> Tuple[bool, Dict[str, float]]:
        """Check CPU usage."""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        system_cpu_percent = psutil.cpu_percent(interval=0.1)
        
        metrics = {
            'process_cpu_percent': cpu_percent,
            'system_cpu_percent': system_cpu_percent,
        }
        
        is_ok = cpu_percent < self.limits.max_cpu_percent
        
        if not is_ok:
            logger.warning(f"CPU usage high: {cpu_percent:.1f}%")
        
        return is_ok, metrics
    
    def check_file_handles(self) -> Tuple[bool, Dict[str, int]]:
        """Check open file handles."""
        try:
            num_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else len(self.process.open_files())
        except (psutil.AccessDenied, AttributeError):
            # Fallback: can't get exact count
            num_fds = len(self.process.open_files())
        
        metrics = {
            'open_file_handles': num_fds,
        }
        
        is_ok = num_fds < self.limits.max_file_handles
        
        if not is_ok:
            logger.warning(f"Too many open file handles: {num_fds}")
        
        return is_ok, metrics
    
    def check_gpu(self) -> Tuple[bool, Dict[str, float]]:
        """Check GPU memory usage."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return True, {'gpu_available': False}
        
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            metrics = {
                'gpu_memory_allocated_gb': gpu_memory_allocated,
                'gpu_memory_reserved_gb': gpu_memory_reserved,
                'gpu_memory_total_gb': gpu_memory_total,
                'gpu_memory_percent': (gpu_memory_allocated / gpu_memory_total) * 100,
            }
            
            is_ok = gpu_memory_allocated < self.limits.max_gpu_memory_gb
            
            if not is_ok:
                logger.warning(
                    f"GPU memory usage high: {gpu_memory_allocated:.2f}GB / {gpu_memory_total:.2f}GB"
                )
            
            return is_ok, metrics
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
            return True, {'gpu_check_error': str(e)}
    
    def full_health_check(self, check_path: Optional[Union[str, Path]] = None) -> HealthCheckResult:
        """Perform a full health check."""
        all_metrics = {}
        issues = []
        
        # Check memory
        mem_ok, mem_metrics = self.check_memory()
        all_metrics.update(mem_metrics)
        if not mem_ok:
            issues.append("High memory usage")
        
        # Check disk
        if check_path:
            disk_ok, disk_metrics = self.check_disk(check_path)
            all_metrics.update(disk_metrics)
            if not disk_ok:
                issues.append("Low disk space")
        
        # Check CPU
        cpu_ok, cpu_metrics = self.check_cpu()
        all_metrics.update(cpu_metrics)
        if not cpu_ok:
            issues.append("High CPU usage")
        
        # Check file handles
        fh_ok, fh_metrics = self.check_file_handles()
        all_metrics.update(fh_metrics)
        if not fh_ok:
            issues.append("Too many open file handles")
        
        # Check GPU
        gpu_ok, gpu_metrics = self.check_gpu()
        all_metrics.update(gpu_metrics)
        if not gpu_ok:
            issues.append("High GPU memory usage")
        
        # Determine status
        if len(issues) == 0:
            status = HealthCheckStatus.HEALTHY
            message = "All systems healthy"
        elif len(issues) == 1:
            status = HealthCheckStatus.DEGRADED
            message = f"Degraded: {issues[0]}"
        elif len(issues) <= 2:
            status = HealthCheckStatus.UNHEALTHY
            message = f"Unhealthy: {', '.join(issues)}"
        else:
            status = HealthCheckStatus.CRITICAL
            message = f"Critical: {', '.join(issues)}"
        
        return HealthCheckResult(
            status=status,
            message=message,
            metrics=all_metrics
        )


class TimeoutHandler:
    """Handle timeouts for long-running operations."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        
        # Set up signal handler for timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout_seconds))
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
        
        elapsed = time.time() - (self.start_time or time.time())
        if elapsed > self.timeout_seconds:
            raise TimeoutError(f"Operation exceeded timeout: {elapsed:.2f}s > {self.timeout_seconds}s")
        
        return False
    
    def check_elapsed(self) -> float:
        """Check elapsed time."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def time_remaining(self) -> float:
        """Get remaining time."""
        elapsed = self.check_elapsed()
        return max(0.0, self.timeout_seconds - elapsed)


def retry_with_backoff(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> T:
    """Retry a function with exponential backoff."""
    config = config or RetryConfig()
    delay = config.initial_delay
    
    last_exception = None
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt < config.max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay = min(delay * config.exponential_base, config.max_delay)
            else:
                logger.error(f"All {config.max_retries + 1} attempts failed")
                raise
        except Exception as e:
            # Non-retryable exception - raise immediately
            logger.error(f"Non-retryable error: {e}")
            raise
    
    # Should never reach here, but for type checking
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def validate_file_integrity(
    file_path: Union[str, Path],
    min_size_bytes: int = 0,
    must_exist: bool = True,
    check_readable: bool = True,
    check_writable: bool = False
) -> Tuple[bool, str]:
    """
    Validate file integrity.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(file_path)
    
    if must_exist and not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    if file_path.exists():
        # Check size
        size = file_path.stat().st_size
        if size < min_size_bytes:
            return False, f"File too small: {size} bytes < {min_size_bytes} bytes"
        
        if size == 0:
            return False, f"File is empty: {file_path}"
        
        # Check readable
        if check_readable and not os.access(file_path, os.R_OK):
            return False, f"File is not readable: {file_path}"
        
        # Check writable
        if check_writable and not os.access(file_path, os.W_OK):
            return False, f"File is not writable: {file_path}"
    
    return True, "OK"


def validate_directory(
    dir_path: Union[str, Path],
    must_exist: bool = True,
    must_be_writable: bool = False,
    min_free_space_gb: float = 1.0
) -> Tuple[bool, str]:
    """
    Validate directory.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    dir_path = Path(dir_path)
    
    if must_exist and not dir_path.exists():
        return False, f"Directory does not exist: {dir_path}"
    
    if dir_path.exists():
        if not dir_path.is_dir():
            return False, f"Path is not a directory: {dir_path}"
        
        # Check writable
        if must_be_writable and not os.access(dir_path, os.W_OK):
            return False, f"Directory is not writable: {dir_path}"
        
        # Check free space
        try:
            disk_usage = psutil.disk_usage(str(dir_path))
            free_gb = disk_usage.free / (1024 ** 3)
            if free_gb < min_free_space_gb:
                return False, f"Insufficient disk space: {free_gb:.2f}GB < {min_free_space_gb}GB"
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
    
    return True, "OK"


@contextmanager
def resource_guard(
    monitor: ResourceMonitor,
    check_path: Optional[Union[str, Path]] = None,
    fail_on_unhealthy: bool = True
):
    """Context manager for resource monitoring."""
    health = monitor.full_health_check(check_path)
    
    if health.status == HealthCheckStatus.CRITICAL:
        if fail_on_unhealthy:
            raise ResourceExhaustedError(f"Critical resource exhaustion: {health.message}")
        else:
            logger.error(f"Critical resource exhaustion: {health.message}")
    
    if health.status == HealthCheckStatus.UNHEALTHY and fail_on_unhealthy:
        logger.warning(f"Unhealthy system state: {health.message}")
    
    try:
        yield health
    finally:
        # Final health check
        final_health = monitor.full_health_check(check_path)
        if final_health.status.value < health.status.value:
            logger.warning(
                f"Resource usage increased during operation: "
                f"{health.status.value} -> {final_health.status.value}"
            )


def guarded_execution(
    func: Callable[..., T],
    timeout_seconds: Optional[float] = None,
    retry_config: Optional[RetryConfig] = None,
    resource_limits: Optional[ResourceLimits] = None,
    check_path: Optional[Union[str, Path]] = None,
    validate_inputs: Optional[Callable[[Any], Tuple[bool, str]]] = None,
    validate_outputs: Optional[Callable[[T], Tuple[bool, str]]] = None,
    *args,
    **kwargs
) -> T:
    """
    Execute a function with comprehensive guardrails.
    
    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time
        retry_config: Retry configuration
        resource_limits: Resource limits
        check_path: Path for disk space checks
        validate_inputs: Function to validate inputs (args, kwargs) -> (is_valid, error)
        validate_outputs: Function to validate outputs -> (is_valid, error)
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result of func(*args, **kwargs)
    """
    # Validate inputs
    if validate_inputs:
        is_valid, error = validate_inputs((args, kwargs))
        if not is_valid:
            raise ValueError(f"Input validation failed: {error}")
    
    # Set up resource monitoring
    monitor = ResourceMonitor(resource_limits)
    
    # Wrapper function with timeout
    def execute_with_timeout():
        if timeout_seconds:
            with TimeoutHandler(timeout_seconds):
                with resource_guard(monitor, check_path, fail_on_unhealthy=True):
                    return func(*args, **kwargs)
        else:
            with resource_guard(monitor, check_path, fail_on_unhealthy=True):
                return func(*args, **kwargs)
    
    # Execute with retries
    if retry_config:
        result = retry_with_backoff(execute_with_timeout, retry_config)
    else:
        result = execute_with_timeout()
    
    # Validate outputs
    if validate_outputs:
        is_valid, error = validate_outputs(result)
        if not is_valid:
            raise DataIntegrityError(f"Output validation failed: {error}")
    
    return result


def guarded_decorator(
    timeout_seconds: Optional[float] = None,
    retry_config: Optional[RetryConfig] = None,
    resource_limits: Optional[ResourceLimits] = None,
    check_path: Optional[Union[str, Path]] = None,
    validate_inputs: Optional[Callable[[Any], Tuple[bool, str]]] = None,
    validate_outputs: Optional[Callable[[T], Tuple[bool, str]]] = None,
):
    """Decorator version of guarded_execution."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return guarded_execution(
                func,
                timeout_seconds=timeout_seconds,
                retry_config=retry_config,
                resource_limits=resource_limits,
                check_path=check_path,
                validate_inputs=validate_inputs,
                validate_outputs=validate_outputs,
                *args,
                **kwargs
            )
        return wrapper
    return decorator

