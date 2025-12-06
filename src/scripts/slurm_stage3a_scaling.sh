#!/bin/bash
#
# SLURM Batch Script for Stage 3a: Video Scaling (Frame-based distribution)
#
# Processes videos based on frame count capacity: 720,000 frames (8 hours @ 4s/100frames)
# Account: eecs442f25_class
#
# Usage:
#   sbatch src/scripts/slurm_stage3a_scaling.sh

#SBATCH --job-name=fvc_stage3a_scaling
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/stage3a_scaling-%j.out
#SBATCH --error=logs/stage3a_scaling-%j.err
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

if [ -z "${FVC_FIXED_SIZE:-}" ]; then
    export FVC_FIXED_SIZE=256
    echo "Using optimized resolution: FVC_FIXED_SIZE=256 (256x256)" >&2
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
log "STAGE 3a: VIDEO SCALING JOB"
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

if ! command -v ffprobe &> /dev/null; then
    log "⚠ WARNING: ffprobe not found. FFmpeg module may not be loaded correctly."
    log "  Try: module load ffmpeg"
    log "  Continuing anyway (may use fallback methods)..."
else
    log "✓ ffprobe found: $(which ffprobe)"
fi

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
AUGMENTED_METADATA_DIR="${FVC_STAGE1_OUTPUT_DIR:-data/augmented_videos}"
AUGMENTED_METADATA=""
for ext in arrow parquet csv; do
    candidate="$ORIG_DIR/$AUGMENTED_METADATA_DIR/augmented_metadata.$ext"
    if [ -f "$candidate" ]; then
        AUGMENTED_METADATA="$candidate"
        log "✓ Stage 1 output found: $AUGMENTED_METADATA"
        break
    fi
done
if [ -z "$AUGMENTED_METADATA" ]; then
    log "✗ ERROR: Stage 1 output not found in $ORIG_DIR/$AUGMENTED_METADATA_DIR/"
    log "  Expected: augmented_metadata.arrow, augmented_metadata.parquet, or augmented_metadata.csv"
    log "  Run Stage 1 first: sbatch src/scripts/slurm_stage1_augmentation.sh"
    exit 1
fi

log "✅ All prerequisites verified"

# ============================================================================
# Calculate Video Range Based on Frame Capacity
# ============================================================================

# Processing capacity: 8 hours = 28800 seconds
# Processing rate: 4 seconds per 100 frames
# Capacity: (28800 / 4) * 100 = 720,000 frames
MAX_FRAMES_CAPACITY=720000

log "=========================================="
log "Calculating video range for Stage 3a"
log "=========================================="
log "Frame capacity: $MAX_FRAMES_CAPACITY frames (8 hours @ 4s/100frames)"
log "Substage: 3a (first of 8 substages)"

# Calculate cumulative frame ranges for all substages
# Substage capacities (in frames):
# 3a, 3b: 720,000 each (eecs442f25_class, 8 hours)
# 3c, 3d: 2,160,000 each (si670f25_class santoshd, 1 day)
# 3e, 3f: 2,160,000 each (si670f25_class urvim, 1 day)
# 3g, 3h: 2,160,000 each (si670f25_class suzanef, 1 day)

# Calculate start and end indices based on cumulative frame counts
# Substage index: 0 (3a is first substage)
SUBSTAGE_INDEX=0

RANGE_RESULT=$(python -c "
import polars as pl
import sys

try:
    metadata_path = '$AUGMENTED_METADATA'
    if metadata_path.endswith('.arrow'):
        df = pl.read_ipc(metadata_path)
    elif metadata_path.endswith('.parquet'):
        df = pl.read_parquet(metadata_path)
    else:
        df = pl.read_csv(metadata_path)
    
    # Check if DataFrame is empty
    if df.height == 0:
        print('ERROR: Metadata file is empty (0 videos)', file=sys.stderr)
        sys.exit(1)
    
    # Substage capacities (cumulative frame counts)
    # 3a: 0-720k, 3b: 720k-1440k, 3c: 1440k-3600k, 3d: 3600k-5760k
    # 3e: 5760k-7920k, 3f: 7920k-10080k, 3g: 10080k-12240k, 3h: 12240k-14400k
    cumulative_ends = [720000, 1440000, 3600000, 5760000, 7920000, 10080000, 12240000, 14400000]
    substage_idx = $SUBSTAGE_INDEX
    
    # Get frame counts - check if frame_count column exists
    if 'frame_count' not in df.columns:
        # Fallback: distribute by video count (equal distribution)
        total_videos = df.height
        videos_per_substage = (total_videos + 7) // 8  # Round up
        start_idx = substage_idx * videos_per_substage
        end_idx = (substage_idx + 1) * videos_per_substage
        if end_idx > total_videos:
            end_idx = total_videos
        print(f'{start_idx},{end_idx}')
    else:
        # Calculate cumulative frame counts
        df = df.with_columns([
            pl.col('frame_count').fill_null(0).cumsum().alias('cumulative_frames')
        ])
        
        total_frames = df['frame_count'].sum()
        
        # Determine target cumulative frame range for this substage
        if substage_idx == 0:
            target_cumulative_start = 0
        else:
            target_cumulative_start = cumulative_ends[substage_idx - 1]
        
        target_cumulative_end = cumulative_ends[substage_idx]
        
        # Find start index: first video where cumulative > start (or = start if start is 0)
        if target_cumulative_start == 0:
            start_idx = 0
        else:
            start_rows = df.filter(pl.col('cumulative_frames') <= target_cumulative_start)
            start_idx = start_rows.height
        
        # Find end index: first video where cumulative > end
        end_rows = df.filter(pl.col('cumulative_frames') <= target_cumulative_end)
        end_idx = end_rows.height
        
        # If we haven't reached the target, include one more video
        if end_idx < df.height:
            if end_idx == 0 or df['cumulative_frames'][end_idx - 1] < target_cumulative_end:
                # Check if adding one more video would exceed capacity
                if end_idx < df.height:
                    next_cumulative = df['cumulative_frames'][end_idx] if end_idx < df.height else total_frames
                    if next_cumulative <= target_cumulative_end:
                        end_idx = end_idx + 1
        
        # Ensure we don't exceed total videos
        if end_idx > df.height:
            end_idx = df.height
        
        print(f'{start_idx},{end_idx}')
        
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
" 2>&1)

if [[ "$RANGE_RESULT" == ERROR:* ]]; then
    log "✗ ERROR: Failed to calculate range: $RANGE_RESULT"
    exit 1
fi

IFS=',' read -r START_IDX END_IDX <<< "$RANGE_RESULT"

if [ -z "$START_IDX" ] || [ -z "$END_IDX" ]; then
    log "✗ ERROR: Could not determine video range"
    exit 1
fi

# Validate that START_IDX and END_IDX are numeric
if ! [[ "$START_IDX" =~ ^[0-9]+$ ]] || ! [[ "$END_IDX" =~ ^[0-9]+$ ]]; then
    log "✗ ERROR: Invalid range values (must be numeric): START_IDX='$START_IDX', END_IDX='$END_IDX'"
    exit 1
fi

# Validate range: START_IDX must be >= 0 and < END_IDX
if [ "$START_IDX" -lt 0 ]; then
    log "✗ ERROR: START_IDX must be >= 0, got: $START_IDX"
    exit 1
fi

if [ "$START_IDX" -ge "$END_IDX" ]; then
    log "✗ ERROR: Invalid range: START_IDX ($START_IDX) >= END_IDX ($END_IDX)"
    log "  This would result in an empty range (no videos to process)"
    exit 1
fi

log ""
log "=========================================="
log "STAGE 3a: VIDEO RANGE ASSIGNMENT"
log "=========================================="
log "This job will process videos in range: [$START_IDX, $END_IDX)"
log "Total videos in this range: $((END_IDX - START_IDX))"
log "Substage: 3a (1st of 8 substages)"
log "Account: eecs442f25_class"
log "Resources: 80GB RAM, 4 CPUs, 1 GPU, 8 hours"
log "=========================================="
log ""

# ============================================================================
# Stage 3a Execution
# ============================================================================

log "=========================================="
log "Starting Stage 3a: Video Scaling"
log "Processing video range: [$START_IDX, $END_IDX)"
log "=========================================="

TARGET_SIZE="${FVC_TARGET_SIZE:-256}"
METHOD="${FVC_DOWNSCALE_METHOD:-autoencoder}"
OUTPUT_DIR="${FVC_STAGE3_OUTPUT_DIR:-data/scaled_videos}"
DELETE_EXISTING="${FVC_DELETE_EXISTING:-0}"
RESUME="${FVC_RESUME:-1}"
CHUNK_SIZE="${FVC_CHUNK_SIZE:-400}"

log "Target size: ${TARGET_SIZE}x${TARGET_SIZE}"
log "Method: $METHOD"
log "Output directory: $OUTPUT_DIR"
log "Delete existing: $DELETE_EXISTING"
log "Resume mode: $RESUME"
log "Chunk size: ${CHUNK_SIZE} frames"

STAGE3_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage3a_scaling_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python || echo "python")

# Validate Python script exists
PYTHON_SCRIPT="$ORIG_DIR/src/scripts/run_stage3_scaling.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log "✗ ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

DELETE_FLAG=""
if [ "$DELETE_EXISTING" = "1" ] || [ "$DELETE_EXISTING" = "true" ] || [ "$DELETE_EXISTING" = "yes" ]; then
    DELETE_FLAG="--delete-existing"
fi

RESUME_FLAG=""
if [ "$RESUME" != "0" ] && [ "$RESUME" != "false" ] && [ "$RESUME" != "no" ]; then
    RESUME_FLAG="--resume"
fi

log "Running Stage 3a scaling script..."
log "Log file: $LOG_FILE"

if "$PYTHON_CMD" "$PYTHON_SCRIPT" \
    --project-root "$ORIG_DIR" \
    --augmented-metadata "$AUGMENTED_METADATA" \
    --method "$METHOD" \
    --target-size "$TARGET_SIZE" \
    --chunk-size "$CHUNK_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    $DELETE_FLAG \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"; then
    
    STAGE3_END=$(date +%s)
    STAGE3_DURATION=$((STAGE3_END - STAGE3_START))
    log "✓ Stage 3a completed successfully in ${STAGE3_DURATION}s ($(($STAGE3_DURATION / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Next step: Run remaining substages (3b-3h)"
else
    STAGE3_END=$(date +%s)
    STAGE3_DURATION=$((STAGE3_END - STAGE3_START))
    log "✗ ERROR: Stage 3a failed after ${STAGE3_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 3a EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE3_DURATION}s ($(($STAGE3_DURATION / 60)) minutes)"
log "Video range processed: [$START_IDX, $END_IDX)"
log "Total videos in range: $((END_IDX - START_IDX))"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Log file: $LOG_FILE"
log ""

# Report missing videos from log file
log "Checking for missing/failed videos..."
if [ -f "$LOG_FILE" ]; then
    MISSING_COUNT=$(grep -c "Video not found\|Could not resolve video path\|skipping" "$LOG_FILE" 2>/dev/null || echo "0")
    if [ "$MISSING_COUNT" -gt 0 ]; then
        log "⚠ WARNING: Found $MISSING_COUNT missing/failed video references in log"
        log "Missing videos (first 10):"
        grep "Video not found\|Could not resolve video path" "$LOG_FILE" 2>/dev/null | head -10 | while read -r line; do
            log "  - $line"
        done
        if [ "$MISSING_COUNT" -gt 10 ]; then
            log "  ... and $((MISSING_COUNT - 10)) more (see log file for details)"
        fi
    else
        log "✓ No missing videos detected"
    fi
fi

log "============================================================"

