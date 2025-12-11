#!/bin/bash
#
# SLURM Batch Script for Repair Stage 3: Re-scale Corrupted Videos
#
# Re-scales videos identified as corrupted by Stage 4.
# Reads corrupted file list from data/corrupted_scaled_videos.txt
#
# Usage:
#   sbatch src/scripts/slurm_repair_stage3_scaling.sh

#SBATCH --job-name=fvc_repair_stage3
#SBATCH --account=si670f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/repair_stage3-%j.out
#SBATCH --error=logs/repair_stage3-%j.err
#SBATCH --mail-user=santoshd@umich.edu,urvim@umich.edu,suzanef@umich.edu
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

if [ -z "${FVC_TARGET_SIZE:-}" ]; then
    export FVC_TARGET_SIZE=256
    echo "Using optimized resolution: FVC_TARGET_SIZE=256 (256x256)" >&2
fi

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02
module load ffmpeg || module load ffmpeg/4.4 || true

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

export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "REPAIR STAGE 3: RE-SCALING CORRUPTED VIDEOS"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python 2>/dev/null || echo 'not found')"
log "Python version: $(python --version 2>&1 || echo 'unknown')"
log "=========================================="
log ""

# ============================================================================
# Check for Corrupted Files List
# ============================================================================

CORRUPTED_LIST="${FVC_CORRUPTED_LIST:-data/corrupted_scaled_videos.txt}"
CORRUPTED_LIST_PATH="$ORIG_DIR/$CORRUPTED_LIST"

log "Checking for corrupted files list: $CORRUPTED_LIST_PATH"

if [ ! -f "$CORRUPTED_LIST_PATH" ]; then
    log "⚠ WARNING: Corrupted files list not found: $CORRUPTED_LIST_PATH"
    log "  This is normal if no corrupted videos were detected by Stage 4."
    log "  Exiting successfully (nothing to repair)."
    exit 0
fi

# Count corrupted files
CORRUPTED_COUNT=$(grep -v '^#' "$CORRUPTED_LIST_PATH" | grep -v '^$' | wc -l || echo "0")
if [ "$CORRUPTED_COUNT" -eq 0 ]; then
    log "⚠ WARNING: Corrupted files list is empty or contains only comments/blank lines"
    log "  Exiting successfully (nothing to repair)."
    exit 0
fi

log "Found $CORRUPTED_COUNT corrupted video(s) to repair"
log ""

# ============================================================================
# Repair Stage 3 Execution
# ============================================================================

log "=========================================="
log "Starting Repair Stage 3: Re-scaling Corrupted Videos"
log "=========================================="

TARGET_SIZE="${FVC_TARGET_SIZE:-256}"
METHOD="${FVC_DOWNSCALE_METHOD:-autoencoder}"
OUTPUT_DIR="${FVC_STAGE3_OUTPUT_DIR:-data/scaled_videos}"
AUGMENTED_METADATA="${FVC_AUGMENTED_METADATA:-data/augmented_videos/augmented_metadata}"
CHUNK_SIZE="${FVC_CHUNK_SIZE:-400}"
MAX_FRAMES="${FVC_MAX_FRAMES:-500}"

log "Target size: ${TARGET_SIZE}x${TARGET_SIZE}"
log "Method: $METHOD"
log "Output directory: $OUTPUT_DIR"
log "Augmented metadata: $AUGMENTED_METADATA"
log "Chunk size: ${CHUNK_SIZE} frames"
log "Max frames: ${MAX_FRAMES} per video"
log ""

REPAIR_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/repair_stage3_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

# Validate Python script exists
PYTHON_SCRIPT="$ORIG_DIR/src/scripts/repair_stage3_scaling.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log "✗ ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

log "Running repair script: $PYTHON_SCRIPT"
log ""

# Run repair script
"$PYTHON_CMD" "$PYTHON_SCRIPT" \
    --project-root "$ORIG_DIR" \
    --corrupted-list "$CORRUPTED_LIST" \
    --augmented-metadata "$AUGMENTED_METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --target-size "$TARGET_SIZE" \
    --method "$METHOD" \
    --chunk-size "$CHUNK_SIZE" \
    --max-frames "$MAX_FRAMES" \
    --delete-corrupted \
    2>&1 | tee "$LOG_FILE"

REPAIR_EXIT_CODE=${PIPESTATUS[0]}
REPAIR_END=$(date +%s)
REPAIR_DURATION=$((REPAIR_END - REPAIR_START))

log ""
log "=========================================="
if [ "$REPAIR_EXIT_CODE" -eq 0 ]; then
    log "✓ REPAIR STAGE 3 COMPLETED SUCCESSFULLY"
else
    log "✗ REPAIR STAGE 3 FAILED (exit code: $REPAIR_EXIT_CODE)"
fi
log "Duration: ${REPAIR_DURATION}s ($(($REPAIR_DURATION / 60))m $(($REPAIR_DURATION % 60))s)"
log "=========================================="
log ""

# Check if corrupted files list still exists (should be archived if all repaired)
if [ -f "$CORRUPTED_LIST_PATH" ]; then
    REMAINING_COUNT=$(grep -v '^#' "$CORRUPTED_LIST_PATH" | grep -v '^$' | wc -l || echo "0")
    if [ "$REMAINING_COUNT" -gt 0 ]; then
        log "⚠ WARNING: $REMAINING_COUNT corrupted video(s) still remain in list"
        log "  Some videos may have failed to repair or were not found in metadata"
    else
        log "✓ All corrupted videos have been processed (list may be archived)"
    fi
fi

if [ "$REPAIR_EXIT_CODE" -eq 0 ]; then
    log "✓ Repair job completed successfully"
    exit 0
else
    log "✗ Repair job failed with exit code: $REPAIR_EXIT_CODE"
    exit "$REPAIR_EXIT_CODE"
fi

