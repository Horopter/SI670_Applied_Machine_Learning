#!/bin/bash

#SBATCH --job-name=fvc_binary_classifier
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

# Suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02
module load cuda/12.1 || true

# Directory setup
mkdir -p logs .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"
export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export ORIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export VENV_DIR="$ORIG_DIR/venv"

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo "$@" >&1
    echo "$@" >&2
    sync 2>/dev/null || true
}

# ============================================================================
# Virtual Environment Setup
# ============================================================================

log "Activating virtual environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    log "✗ ERROR: Virtual environment not found: $VENV_DIR"
    exit 1
fi

# Check venv Python version BEFORE activating (venv is locked to the Python that created it)
if [ -f "$VENV_DIR/bin/python" ]; then
    VENV_PYTHON_VERSION=$("$VENV_DIR/bin/python" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    MODULE_PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2 || echo "3.11")
    
    if [ "$VENV_PYTHON_VERSION" != "3.11" ] && [ "$VENV_PYTHON_VERSION" != "$MODULE_PYTHON_VERSION" ]; then
        log "✗ ERROR: venv was created with Python $VENV_PYTHON_VERSION, but module provides Python $MODULE_PYTHON_VERSION"
        log "  The venv is 'locked' to the Python version that created it."
        log "  Recreate the venv with:"
        log "    rm -rf venv"
        log "    module load python3.11-anaconda/2024.02"
        log "    python3.11 -m venv venv"
        log "    source venv/bin/activate"
        log "    pip install -r requirements.txt"
        exit 1
    fi
fi

source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Verify venv Python matches module Python after activation
ACTIVE_PYTHON=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [ "$ACTIVE_PYTHON" != "3.11" ]; then
    log "⚠ WARNING: Active Python version is $ACTIVE_PYTHON, expected 3.11"
    log "  This may cause compatibility issues."
fi

mkdir -p "$WORK_DIR/runs" "$WORK_DIR/logs"
ln -snf "$WORK_DIR/runs" runs 2>/dev/null || true

# ============================================================================
# Environment Variables for Performance
# ============================================================================

export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ============================================================================
# System Information
# ============================================================================

set +u
log "=========================================="
log "JOB STARTUP INFORMATION"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python 2>/dev/null || echo 'not found')"
log "Python version: $(python --version 2>&1 || echo 'unknown')"
log "=========================================="
set -u

# Verify Python version (should be 3.11 from module, but check anyway)
PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))" 2>/dev/null || echo "unknown")
if [ "$PYTHON_VERSION" != "3.11" ] && [ "$PYTHON_VERSION" != "unknown" ]; then
    log "⚠ WARNING: Python version is $PYTHON_VERSION, expected 3.11"
    log "  This may cause compatibility issues. Ensure module python3.11-anaconda/2024.02 is loaded."
fi

# ============================================================================
# Verify Environment
# ============================================================================

log "Verifying Python environment..."
if python -c "import sys; import papermill; print('✓ papermill available', flush=True)" 2>/dev/null; then
    log "✓ papermill found"
else
    log "✗ ERROR: papermill not found. Install with: pip install papermill"
    exit 1
fi

# Check for ipykernel (required for papermill to execute notebooks)
if python -c "import ipykernel" 2>/dev/null; then
    log "✓ ipykernel found"
else
    log "✗ ERROR: ipykernel not found. Install with: pip install ipykernel"
    log "  Or install all requirements: pip install -r requirements.txt"
    exit 1
fi

# ============================================================================
# Jupyter Kernel Setup
# ============================================================================

KNAME="fvc-binary-classifier-${SLURM_JOB_ID:-$$}"
KERNEL_ARG="python3"  # Default fallback

log "Installing Jupyter kernel: ${KNAME}..."
# ipykernel should be installed by now (checked/installed in Verify Environment section)
if python -m ipykernel install --user --name "${KNAME}" --display-name "FVC Binary Classifier (${KNAME})" 2>&1; then
    log "✓ Kernel installed successfully"
    KERNEL_ARG="${KNAME}"
    trap 'jupyter kernelspec remove -y "${KNAME}" 2>/dev/null || true' EXIT
else
    log "⚠ WARNING: Kernel installation failed, using default python3"
    log "  This is usually fine - papermill can use the default python3 kernel"
fi

# ============================================================================
# Notebook Configuration
# ============================================================================

NOTEBOOK_IN="src/fvc_binary_classifier.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
NOTEBOOK_OUT="$WORK_DIR/runs/fvc_binary_classifier_executed_${STAMP}.ipynb"

# ============================================================================
# Pre-flight Validation
# ============================================================================

log "Running pre-flight validation..."
VALIDATION_ERRORS=0

# Check for notebook
if [ ! -f "$ORIG_DIR/$NOTEBOOK_IN" ]; then
    log "✗ ERROR: Notebook not found: $NOTEBOOK_IN"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
else
    log "✓ Notebook found: $NOTEBOOK_IN"
fi

if [ $VALIDATION_ERRORS -gt 0 ]; then
    log "❌ PRE-FLIGHT VALIDATION FAILED: $VALIDATION_ERRORS error(s) found"
    exit 1
fi

log "✅ Pre-flight validation passed!"

# ============================================================================
# File Copying (if using SLURM_TMPDIR)
# ============================================================================

if [ -n "${SLURM_TMPDIR:-}" ] && [ "$WORK_DIR" != "$ORIG_DIR" ]; then
    log "Using SLURM_TMPDIR: $SLURM_TMPDIR"
    mkdir -p "$WORK_DIR/src"
    
    cp -v "$ORIG_DIR/$NOTEBOOK_IN" "$WORK_DIR/$NOTEBOOK_IN" 2>/dev/null || exit 1
    cd "$WORK_DIR"
fi

# ============================================================================
# Cleanup Previous Instances
# ============================================================================

log "=== Cleaning Previous Runs ==="
# Kill any previous instances
pkill -f "papermill.*fvc_binary_classifier" 2>/dev/null || true
pkill -f "ipykernel.*fvc-binary-classifier" 2>/dev/null || true

# Clear logs folder in ORIG_DIR (the actual logs directory)
if [ -d "$ORIG_DIR/logs" ]; then
    # Count files before deletion for logging
    DELETED_COUNT=$(find "$ORIG_DIR/logs" -type f \( -name "fvc_binary_classifier*" -o -name "*.log" \) 2>/dev/null | wc -l || echo "0")
    # Delete all log files (including .out, .err, .log)
    find "$ORIG_DIR/logs" -type f \( -name "fvc_binary_classifier*" -o -name "*.log" \) -delete 2>/dev/null || true
    if [ "$DELETED_COUNT" -gt 0 ]; then
        log "✓ Cleared $DELETED_COUNT log file(s) from: $ORIG_DIR/logs"
    else
        log "✓ Logs folder checked: $ORIG_DIR/logs (no matching files to delete)"
    fi
fi

# Also clear logs in WORK_DIR if different
if [ "$WORK_DIR" != "$ORIG_DIR" ] && [ -d "$WORK_DIR/logs" ]; then
    DELETED_COUNT=$(find "$WORK_DIR/logs" -type f \( -name "fvc_binary_classifier*" -o -name "*.log" \) 2>/dev/null | wc -l || echo "0")
    find "$WORK_DIR/logs" -type f \( -name "fvc_binary_classifier*" -o -name "*.log" \) -delete 2>/dev/null || true
    if [ "$DELETED_COUNT" -gt 0 ]; then
        log "✓ Cleared $DELETED_COUNT log file(s) from: $WORK_DIR/logs"
    fi
fi

# Clean up runs folder (keep directory structure)
if [ -d "$WORK_DIR/runs" ]; then
    find "$WORK_DIR/runs" -type f -name "fvc_binary_classifier_executed_*.ipynb" -mtime +7 -delete 2>/dev/null || true
    log "✓ Cleaned old executed notebooks from runs folder"
fi
sleep 2
log "✓ Cleanup completed"

# ============================================================================
# Notebook Execution (or MLOps Pipeline)
# ============================================================================

# Check if we should use MLOps pipeline instead of notebook
# Default to MLOps pipeline for better OOM handling, K-fold CV, and checkpointing
USE_MLOPS_PIPELINE="${USE_MLOPS_PIPELINE:-true}"

if [ "$USE_MLOPS_PIPELINE" = "true" ]; then
    log "Using MLOps pipeline (run_mlops_pipeline.py)"
    log "Features: K-fold CV, aggressive GC, OOM handling, per-stage checkpointing"
    
    PIPELINE_START=$(date +%s)
    LOG_FILE="$WORK_DIR/logs/fvc_binary_classifier_run.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Change to project root for script execution
    cd "$ORIG_DIR" || exit 1
    
    # Ensure we're using the correct Python from venv
    PYTHON_CMD=$(which python || echo "python")
    
    if "$PYTHON_CMD" "$ORIG_DIR/src/run_mlops_pipeline.py" 2>&1 | tee "$LOG_FILE"; then
        PIPELINE_END=$(date +%s)
        PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
        log "✓ MLOps pipeline completed in ${PIPELINE_DURATION}s"
        NOTEBOOK_DURATION=$PIPELINE_DURATION
    else
        PIPELINE_END=$(date +%s)
        PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
        log "✗ ERROR: MLOps pipeline failed after ${PIPELINE_DURATION}s!"
        log "Check log file: $LOG_FILE"
        exit 1
    fi
else
    log "Running papermill -> ${NOTEBOOK_OUT}"
    log "Notebook: ${NOTEBOOK_IN}"
    log "Kernel: ${KERNEL_ARG}"
    NOTEBOOK_START=$(date +%s)

# Run with proper logging - capture both stdout and stderr
LOG_FILE="$WORK_DIR/logs/fvc_binary_classifier_run.log"
mkdir -p "$(dirname "$LOG_FILE")"

if ! papermill "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" \
    --kernel "${KERNEL_ARG}" \
    --log-output \
    --execution-timeout 14400 \
    2>&1 | tee "$LOG_FILE"; then
    NOTEBOOK_END=$(date +%s)
    NOTEBOOK_DURATION=$((NOTEBOOK_END - NOTEBOOK_START))
    log "✗ ERROR: Papermill failed after ${NOTEBOOK_DURATION}s!"
    log "Check log file: $LOG_FILE"
    exit 1
fi

NOTEBOOK_END=$(date +%s)
NOTEBOOK_DURATION=$((NOTEBOOK_END - NOTEBOOK_START))
log "✓ Notebook execution completed in ${NOTEBOOK_DURATION}s"
fi

# ============================================================================
# Copy Results Back
# ============================================================================

if [ -n "${SLURM_TMPDIR:-}" ] && [ "$WORK_DIR" != "$ORIG_DIR" ]; then
    log "Copying results back..."
    
    # Copy logs
    if [ -f "$LOG_FILE" ]; then
        mkdir -p "$ORIG_DIR/logs"
        cp -v "$LOG_FILE" "$ORIG_DIR/logs/" 2>/dev/null || true
    fi
    
    # Copy executed notebook
    if [ -n "${SLURM_JOB_ID:-}" ] && [ -f "${NOTEBOOK_OUT}" ]; then
        RUN_OUT_DIR="$ORIG_DIR/runs/run_${SLURM_JOB_ID}"
        mkdir -p "$RUN_OUT_DIR"
        cp -v "${NOTEBOOK_OUT}" "$RUN_OUT_DIR/" 2>/dev/null || true
    fi
fi

log ""
log "============================================================"
log "EXECUTION SUMMARY"
log "============================================================"
if [ "$USE_MLOPS_PIPELINE" = "true" ]; then
    log "MLOps pipeline execution time: ${NOTEBOOK_DURATION}s"
else
    log "Notebook execution time: ${NOTEBOOK_DURATION}s"
fi
log "============================================================"

