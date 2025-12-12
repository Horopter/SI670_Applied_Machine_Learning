#!/bin/bash
#
# SLURM Batch Script for Stage 3h: Video Scaling (Frame-based distribution)
#
# Processes videos based on frame count capacity: 2160000 frames (1 day @ 4s/100frames)
# Account: stats_dept2
#
# Usage:
#   sbatch src/scripts/slurm_stage3h_scaling.sh

#SBATCH --job-name=fvc_stage3h_scaling
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/stage3h_scaling-%j.out
#SBATCH --error=logs/stage3h_scaling-%j.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ============================================================================
# System Information
# ============================================================================

log "=========================================="
log "STAGE 3h: VIDEO SCALING JOB"
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

# Processing capacity: 2160000 frames (1 day @ 4s/100frames)
MAX_FRAMES_CAPACITY=2160000

log "=========================================="
log "Calculating video range for Stage 3h"
log "=========================================="
log "Frame capacity: $MAX_FRAMES_CAPACITY frames (1 day @ 4s/100frames)"
log "Substage: 3h (eighth of 8 substages)"

# Calculate start and end indices based on cumulative frame counts
# Substage index: 7 (3h is eighth substage)
SUBSTAGE_INDEX=7

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
    cumulative_ends = [720000, 1440000, 3600000, 5760000, 7920000, 10080000, 12240000, 14400000]
    substage_idx = $SUBSTAGE_INDEX
    
    # Get frame counts - check if frame_count column exists
    if 'frame_count' not in df.columns:
        # Fallback: distribute by video count (equal distribution)
        total_videos = df.height
        videos_per_substage = (total_videos + 7) // 8
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
log "STAGE 3h: VIDEO RANGE ASSIGNMENT"
log "=========================================="
log "This job will process videos in range: [$START_IDX, $END_IDX)"
log "Total videos in this range: $((END_IDX - START_IDX))"
log "Substage: 3h (8th of 8 substages)"
log "Account: stats_dept2"
log "Resources: 64GB RAM, 1 CPU, 1 GPU, 1 day"
log "=========================================="
log ""

# ============================================================================
# Stage 3h Execution
# ============================================================================

log "=========================================="
log "Starting Stage 3h: Video Scaling"
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
LOG_FILE="$ORIG_DIR/logs/stage3h_scaling_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null || echo "python3")
# Use unbuffered Python for immediate output
PYTHON_CMD="$PYTHON_CMD -u"

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

# Execution order: forward (0) or reverse (1), default to reverse
EXECUTION_ORDER="${FVC_EXECUTION_ORDER:-reverse}"
if [ "$EXECUTION_ORDER" = "0" ] || [ "$EXECUTION_ORDER" = "forward" ]; then
    EXECUTION_ORDER="forward"
elif [ "$EXECUTION_ORDER" = "1" ] || [ "$EXECUTION_ORDER" = "reverse" ]; then
    EXECUTION_ORDER="reverse"
else
    EXECUTION_ORDER="reverse"  # Default
fi

log "Running Stage 3h scaling script..."
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
    --execution-order "$EXECUTION_ORDER" \
    $DELETE_FLAG \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"; then
    
    STAGE3_END=$(date +%s)
    STAGE3_DURATION=$((STAGE3_END - STAGE3_START))
    log "✓ Stage 3h completed successfully in ${STAGE3_DURATION}s ($((${STAGE3_DURATION} / 60)) minutes)"

    # Run sanity check to verify all scaled videos are in metadata
    log ""
    log "=========================================="
    log "STAGE 3H: SANITY CHECK"
    log "=========================================="
    SANITY_CHECK_SCRIPT="$ORIG_DIR/src/scripts/check_stage3_completion.py"
    if [ -f "$SANITY_CHECK_SCRIPT" ]; then
        log "Running sanity check: verifying all scaled videos are in metadata..."
        if "$PYTHON_CMD" "$SANITY_CHECK_SCRIPT" \
            --project-root "$ORIG_DIR" \
            --scaled-videos-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"; then
            log "✓ Sanity check passed"
        else
            log "⚠ WARNING: Sanity check found discrepancies (see log above)"
            log "  This may indicate missing metadata entries for some videos"
            log "  Consider running with --reconstruct flag to fix metadata"
        fi
    else
        log "⚠ WARNING: Sanity check script not found: $SANITY_CHECK_SCRIPT"
    fi

    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Next step: Run remaining substages if not all complete"
else
    STAGE3_END=$(date +%s)
    STAGE3_DURATION=$((STAGE3_END - STAGE3_START))
    log "✗ ERROR: Stage 3h failed after ${STAGE3_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 3h EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE3_DURATION}s ($((${STAGE3_DURATION} / 60)) minutes)"
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
