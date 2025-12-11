# Production Guardrails System

## Overview

This document describes the comprehensive guardrail system implemented to ensure production reliability and prevent failures.

## System Architecture

The guardrail system operates at multiple levels:

1. **Resource Monitoring**: Real-time monitoring of CPU, memory, disk, GPU, and file handles
2. **Data Integrity**: Validation of all data files, metadata, and feature files
3. **Timeout Management**: Prevents operations from hanging indefinitely
4. **Retry Logic**: Automatic retry with exponential backoff for transient failures
5. **Health Checks**: System-wide health monitoring with status levels
6. **Input/Output Validation**: Boundary validation at all pipeline stages
7. **Error Recovery**: Graceful handling and recovery from failures

## Components

### 1. Resource Monitoring (`lib/utils/guardrails.py`)

**ResourceMonitor**: Monitors system resources and enforces limits
- Memory usage (process and system)
- Disk space (total, used, free)
- CPU usage (process and system)
- File handles (open file descriptors)
- GPU memory (if available)

**Resource Limits**:
- Max memory: 200 GB (configurable)
- Max disk: 1000 GB (configurable)
- Max CPU: 95% (configurable)
- Max file handles: 10,000 (configurable)
- Min free disk: 50 GB (configurable)
- Max GPU memory: 20 GB (configurable)

**Health Check Status Levels**:
- `HEALTHY`: All systems operating normally
- `DEGRADED`: One resource issue detected
- `UNHEALTHY`: Multiple resource issues
- `CRITICAL`: Severe resource exhaustion

### 2. Data Integrity (`lib/utils/data_integrity.py`)

**DataIntegrityChecker**: Validates data files and metadata
- File existence and accessibility
- File size validation
- Metadata schema validation
- Feature file format validation
- Video file validation
- Data consistency checks
- Referential integrity

**Validation Functions**:
- `validate_metadata_file()`: Validates metadata files (Arrow/Parquet/CSV)
- `validate_feature_file()`: Validates feature files (NPY/Parquet)
- `validate_video_file()`: Validates video files
- `validate_stage_prerequisites()`: Validates prerequisites for pipeline stages
- `validate_data_consistency()`: Validates file references in metadata

### 3. Timeout Management

**TimeoutHandler**: Context manager for operation timeouts
- Signal-based timeout (Unix)
- Elapsed time tracking
- Automatic timeout enforcement
- Graceful timeout handling

### 4. Retry Logic

**RetryConfig**: Configuration for retry behavior
- Max retries: 3 (configurable)
- Initial delay: 1 second
- Max delay: 60 seconds
- Exponential backoff: base 2.0
- Retryable exceptions: OSError, IOError, ConnectionError, TimeoutError

**retry_with_backoff()**: Executes function with automatic retries

### 5. Pipeline Guardrails (`lib/utils/pipeline_guardrails.py`)

**PipelineGuardrails**: Stage-specific validation
- `validate_stage1_output()`: Validates augmentation output
- `validate_stage2_output()`: Validates feature extraction output
- `validate_stage3_output()`: Validates video scaling output (requires > 3000 rows)
- `validate_stage4_output()`: Validates scaled feature extraction output
- `validate_stage5_prerequisites()`: Validates all prerequisites for training
- `preflight_check()`: Performs preflight check for any stage

## Integration Points

### Stage 5 Training Pipeline

**Location**: `lib/training/pipeline.py`

**Guardrails Applied**:
1. **Pre-flight Validation**: Validates all prerequisites before training starts
2. **Metadata Integrity**: Validates metadata file before loading
3. **Row Count Validation**: Ensures > 3000 rows (critical requirement)
4. **Resource Health Check**: Checks system health before proceeding
5. **Data Consistency**: Validates file references in metadata

**Code Example**:
```python
from lib.utils.data_integrity import DataIntegrityChecker
from lib.utils.guardrails import ResourceMonitor, HealthCheckStatus, ResourceExhaustedError

# Validate metadata integrity
is_valid, error_msg, scaled_df = DataIntegrityChecker.validate_metadata_file(
    metadata_path_obj,
    required_columns={'video_path', 'label'},
    min_rows=3000,
    allow_empty=False
)

# Resource health check
monitor = ResourceMonitor()
health = monitor.full_health_check(project_root_path)
if health.status == HealthCheckStatus.CRITICAL:
    raise ResourceExhaustedError(f"Critical system state: {health.message}")
```

## Critical Validations

### 1. Stage 3 Metadata Validation
- **Requirement**: Must have > 3000 rows
- **Validation**: Enforced at multiple points
- **Failure Action**: Raises `ValueError` with detailed error message

### 2. Resource Exhaustion
- **Monitoring**: Continuous monitoring during operations
- **Thresholds**: Configurable limits for all resources
- **Failure Action**: Raises `ResourceExhaustedError` if critical

### 3. Data Integrity
- **File Validation**: All files validated before use
- **Schema Validation**: Metadata schemas validated
- **Consistency Checks**: File references validated
- **Failure Action**: Raises `DataIntegrityError` with details

### 4. Timeout Protection
- **Default Timeout**: Configurable per operation
- **Signal Handling**: Unix signal-based timeout
- **Failure Action**: Raises `TimeoutError` if exceeded

## Error Handling Strategy

### Error Hierarchy
1. **GuardrailError**: Base exception for guardrail violations
2. **ResourceExhaustedError**: System resources exhausted
3. **TimeoutError**: Operation exceeded timeout
4. **DataIntegrityError**: Data validation failed

### Error Recovery
- **Automatic Retry**: Transient errors retried automatically
- **Graceful Degradation**: Non-critical issues logged as warnings
- **Fail Fast**: Critical issues fail immediately with clear errors
- **Detailed Logging**: All errors logged with full context

## Usage Examples

### Basic Resource Monitoring
```python
from lib.utils.guardrails import ResourceMonitor, resource_guard

monitor = ResourceMonitor()
with resource_guard(monitor, check_path="/data"):
    # Your operation here
    pass
```

### Data Validation
```python
from lib.utils.data_integrity import DataIntegrityChecker

is_valid, error, df = DataIntegrityChecker.validate_metadata_file(
    metadata_path,
    required_columns={'video_path', 'label'},
    min_rows=3000
)
if not is_valid:
    raise DataIntegrityError(error)
```

### Guarded Execution
```python
from lib.utils.guardrails import guarded_execution, RetryConfig, ResourceLimits

result = guarded_execution(
    my_function,
    timeout_seconds=3600,
    retry_config=RetryConfig(max_retries=3),
    resource_limits=ResourceLimits(max_memory_gb=200),
    check_path="/data"
)
```

### Pipeline Preflight Check
```python
from lib.utils.pipeline_guardrails import PipelineGuardrails

guardrails = PipelineGuardrails(project_root="/path/to/project")
is_ok, errors, info = guardrails.validate_stage5_prerequisites(
    model_types=['logistic_regression', 'svm'],
    scaled_metadata_path="data/scaled_videos/scaled_metadata.parquet",
    features_stage2_path="data/features_stage2/features_metadata.parquet"
)
```

## Configuration

### Resource Limits
Customize resource limits based on your system:
```python
from lib.utils.guardrails import ResourceLimits

limits = ResourceLimits(
    max_memory_gb=200.0,
    max_disk_gb=1000.0,
    max_cpu_percent=95.0,
    min_free_disk_gb=50.0,
    max_gpu_memory_gb=20.0
)
```

### Retry Configuration
Customize retry behavior:
```python
from lib.utils.guardrails import RetryConfig

retry_config = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    max_delay=120.0,
    exponential_base=2.0
)
```

## Monitoring and Logging

All guardrail operations are logged:
- Resource checks: WARNING level for issues
- Data validation: ERROR level for failures
- Health checks: INFO level for status
- Timeouts: ERROR level with details
- Retries: WARNING level for retry attempts

## Best Practices

1. **Always validate inputs** at function boundaries
2. **Check resources** before long-running operations
3. **Use timeouts** for all external operations
4. **Enable retries** for transient failures
5. **Validate outputs** after critical operations
6. **Monitor health** continuously during processing
7. **Fail fast** on critical errors
8. **Log everything** for debugging

## Testing

Guardrails are tested through:
- Unit tests for individual components
- Integration tests for pipeline stages
- Stress tests for resource limits
- Failure injection tests

## Future Enhancements

1. **Circuit Breakers**: Prevent cascading failures
2. **Rate Limiting**: Prevent resource exhaustion
3. **Distributed Monitoring**: Multi-node resource tracking
4. **Predictive Alerts**: Warn before resource exhaustion
5. **Auto-scaling**: Automatic resource adjustment
6. **Metrics Export**: Export metrics to monitoring systems

