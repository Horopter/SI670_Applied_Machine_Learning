#!/bin/bash
#
# SLURM Batch Script for Stage 3: Video Downscaling
#
# Downscales videos to a target resolution using letterboxing.
#
# Usage:
#   sbatch src/scripts/slurm_stage3_downscaling.sh
#   sbatch --time=4:00:00 src/scripts/slurm_stage3_downscaling.sh
#   sbatch --mem=64G src/scripts/slurm_stage3_downscaling.sh
#

#SBATCH --job-name=fvc_stage3_down
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=0  # Downscaling doesn't need GPU
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage3_down-%j.out
#SBATCH --error=logs/stage3_down-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

# ============================================================================
# Environment Setup
# ============================================================================

unset MallocStackLogging || true
unset MallocStackLoggingNoCompact || true
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# Set extreme conservative memory settings
if [ -z "${FVC_FIXED_SIZE:-}" ]; then
    export FVC_FIXED_SIZE=112
    echo "Using extreme conservative resolution: FVC_FIXED_SIZE=112 (112x112)" >&2
fi

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02

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

source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_DISABLE_PROMPT=1

# ============================================================================
# Environment Variables
# ============================================================================

export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "STAGE 3: VIDEO DOWNSCALING JOB"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python 2>/dev/null || echo 'not found')"
log "Python version: $(python --version 2>&1 || echo 'unknown')"
log "=========================================="

# ============================================================================
# Verify Prerequisites
# ============================================================================

log "Verifying prerequisites..."

# Check critical Python packages
PREREQ_PACKAGES=("polars" "numpy" "opencv-python" "av")

MISSING_PACKAGES=()
for pkg in "${PREREQ_PACKAGES[@]}"; do
    case "$pkg" in
        "opencv-python")
            if ! python -c "import cv2" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg (cv2) found"
            fi
            ;;
        *)
            if ! python -c "import $pkg" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg found"
            fi
            ;;
    esac
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    log "✗ ERROR: Missing required packages: ${MISSING_PACKAGES[*]}"
    exit 1
fi

# Verify Stage 1 output
AUGMENTED_METADATA="${FVC_STAGE1_OUTPUT_DIR:-data/augmented_videos}/augmented_metadata.csv"
AUGMENTED_METADATA="$ORIG_DIR/$AUGMENTED_METADATA"
if [ ! -f "$AUGMENTED_METADATA" ]; then
    log "✗ ERROR: Stage 1 output not found: $AUGMENTED_METADATA"
    log "  Run Stage 1 first: sbatch src/scripts/slurm_stage1_augmentation.sh"
    exit 1
else
    log "✓ Stage 1 output found: $AUGMENTED_METADATA"
fi

log "✅ All prerequisites verified"

# ============================================================================
# Stage 3 Execution
# ============================================================================

log "=========================================="
log "Starting Stage 3: Video Downscaling"
log "=========================================="

TARGET_SIZE="${FVC_TARGET_SIZE:-224}"
METHOD="${FVC_DOWNSCALE_METHOD:-resolution}"
OUTPUT_DIR="${FVC_STAGE3_OUTPUT_DIR:-data/downscaled_videos}"
log "Target size: ${TARGET_SIZE}x${TARGET_SIZE}"
log "Method: $METHOD"
log "Output directory: $OUTPUT_DIR"

STAGE3_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage3_downscaling_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

log "Running Stage 3 downscaling script..."
log "Log file: $LOG_FILE"

if "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage3_downscaling.py" \
    --project-root "$ORIG_DIR" \
    --augmented-metadata "${FVC_STAGE1_OUTPUT_DIR:-data/augmented_videos}/augmented_metadata.csv" \
    --method "$METHOD" \
    --target-size "$TARGET_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"; then
    
    STAGE3_END=$(date +%s)
    STAGE3_DURATION=$((STAGE3_END - STAGE3_START))
    log "✓ Stage 3 completed successfully in ${STAGE3_DURATION}s ($(($STAGE3_DURATION / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Next step: Run Stage 4 with: sbatch src/scripts/slurm_stage4_downscaled_features.sh"
else
    STAGE3_END=$(date +%s)
    STAGE3_DURATION=$((STAGE3_END - STAGE3_START))
    log "✗ ERROR: Stage 3 failed after ${STAGE3_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 3 EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE3_DURATION}s ($(($STAGE3_DURATION / 60)) minutes)"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Log file: $LOG_FILE"
log "============================================================"

