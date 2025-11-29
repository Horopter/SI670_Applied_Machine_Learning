#!/bin/bash
#
# SLURM Batch Script for Stage 4: Downscaled Feature Extraction
#
# Extracts additional features from downscaled videos (P features).
#
# Usage:
#   sbatch src/scripts/slurm_stage4_downscaled_features.sh
#   sbatch --time=6:00:00 src/scripts/slurm_stage4_downscaled_features.sh
#   sbatch --mem=80G src/scripts/slurm_stage4_downscaled_features.sh
#

#SBATCH --job-name=fvc_stage4_feat
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=0  # Feature extraction doesn't need GPU
#SBATCH --time=6:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage4_feat-%j.out
#SBATCH --error=logs/stage4_feat-%j.err
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
log "STAGE 4: DOWNSCALED FEATURE EXTRACTION JOB"
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
PREREQ_PACKAGES=("polars" "numpy" "opencv-python" "av" "scipy" "scikit-learn")

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
        "scikit-learn")
            if ! python -c "import sklearn" 2>/dev/null; then
                MISSING_PACKAGES+=("$pkg")
            else
                log "✓ $pkg (sklearn) found"
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

# Verify Stage 3 output
DOWNSCALED_METADATA="${FVC_STAGE3_OUTPUT_DIR:-data/downscaled_videos}/downscaled_metadata.csv"
DOWNSCALED_METADATA="$ORIG_DIR/$DOWNSCALED_METADATA"
if [ ! -f "$DOWNSCALED_METADATA" ]; then
    log "✗ ERROR: Stage 3 output not found: $DOWNSCALED_METADATA"
    log "  Run Stage 3 first: sbatch src/scripts/slurm_stage3_downscaling.sh"
    exit 1
else
    log "✓ Stage 3 output found: $DOWNSCALED_METADATA"
fi

log "✅ All prerequisites verified"

# ============================================================================
# Stage 4 Execution
# ============================================================================

log "=========================================="
log "Starting Stage 4: Downscaled Feature Extraction"
log "=========================================="

NUM_FRAMES="${FVC_NUM_FRAMES:-6}"  # Optimized for 80GB RAM
OUTPUT_DIR="${FVC_STAGE4_OUTPUT_DIR:-data/features_stage4}"
log "Number of frames: $NUM_FRAMES"
log "Output directory: $OUTPUT_DIR"

STAGE4_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage4_downscaled_features_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

log "Running Stage 4 downscaled feature extraction script..."
log "Log file: $LOG_FILE"

if "$PYTHON_CMD" "$ORIG_DIR/src/scripts/run_stage4_downscaled_features.py" \
    --project-root "$ORIG_DIR" \
    --downscaled-metadata "${FVC_STAGE3_OUTPUT_DIR:-data/downscaled_videos}/downscaled_metadata.csv" \
    --num-frames "$NUM_FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"; then
    
    STAGE4_END=$(date +%s)
    STAGE4_DURATION=$((STAGE4_END - STAGE4_START))
    log "✓ Stage 4 completed successfully in ${STAGE4_DURATION}s ($(($STAGE4_DURATION / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Next step: Run Stage 5 with: sbatch src/scripts/slurm_stage5_training.sh"
else
    STAGE4_END=$(date +%s)
    STAGE4_DURATION=$((STAGE4_END - STAGE4_START))
    log "✗ ERROR: Stage 4 failed after ${STAGE4_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 4 EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE4_DURATION}s ($(($STAGE4_DURATION / 60)) minutes)"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Log file: $LOG_FILE"
log "============================================================"

