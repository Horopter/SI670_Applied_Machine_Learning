#!/bin/bash
#
# SLURM Batch Script for Stage 4c: Scaled Video Feature Extraction
#
# Processes last third of videos (equal distribution, includes remainder)
# Account: si670f25_class (suzanef)
#
# Usage:
#   sbatch src/scripts/slurm_stage4c_scaled_features.sh

#SBATCH --job-name=fvc_stage4c_feat
#SBATCH --account=si670f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/stage4c_feat-%j.out
#SBATCH --error=logs/stage4c_feat-%j.err
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
log "STAGE 4c: SCALED VIDEO FEATURE EXTRACTION JOB"
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

# Verify Stage 3 output (try Arrow first, then Parquet, then CSV)
SCALED_METADATA_DIR="${FVC_STAGE3_OUTPUT_DIR:-data/scaled_videos}"
SCALED_METADATA=""
for ext in arrow parquet csv; do
    candidate="$ORIG_DIR/$SCALED_METADATA_DIR/scaled_metadata.$ext"
    if [ -f "$candidate" ]; then
        SCALED_METADATA="$candidate"
        log "✓ Stage 3 output found: $SCALED_METADATA"
        break
    fi
done
if [ -z "$SCALED_METADATA" ]; then
    log "⚠ WARNING: Stage 3 output not found in $ORIG_DIR/$SCALED_METADATA_DIR/"
    log "  Expected: scaled_metadata.arrow, scaled_metadata.parquet, or scaled_metadata.csv"
    log "  Attempting to reconstruct metadata from scaled video files..."
    
    # Try to reconstruct metadata from scaled video files
    RECONSTRUCT_SCRIPT="$ORIG_DIR/src/scripts/reconstruct_scaled_metadata.py"
    if [ -f "$RECONSTRUCT_SCRIPT" ]; then
        log "Running reconstruction script..."
        if "$PYTHON_CMD" "$RECONSTRUCT_SCRIPT" \
            --project-root "$ORIG_DIR" \
            --scaled-videos-dir "$SCALED_METADATA_DIR" \
            2>&1 | tee -a "$LOG_FILE"; then
            # Check again for metadata after reconstruction
            for ext in arrow parquet csv; do
                candidate="$ORIG_DIR/$SCALED_METADATA_DIR/scaled_metadata.$ext"
                if [ -f "$candidate" ]; then
                    SCALED_METADATA="$candidate"
                    log "✓ Reconstructed metadata found: $SCALED_METADATA"
                    break
                fi
            done
            if [ -z "$SCALED_METADATA" ]; then
                log "✗ ERROR: Metadata reconstruction failed or produced no output"
                log "  Check reconstruction script output above"
                exit 1
            fi
        else
            log "✗ ERROR: Metadata reconstruction script failed"
            log "  Check reconstruction script output above"
            exit 1
        fi
    else
        log "✗ ERROR: Reconstruction script not found: $RECONSTRUCT_SCRIPT"
        log "  Cannot reconstruct metadata automatically"
        log "  Run Stage 3 first: sbatch src/scripts/slurm_stage3_scaling.sh"
        exit 1
    fi
fi

log "✅ All prerequisites verified"

# ============================================================================
# Calculate Video Range (Equal Distribution - Last Third, Includes Remainder)
# ============================================================================

log "=========================================="
log "Calculating video range for Stage 4c"
log "=========================================="
log "Distribution: Equal video count (last third, includes remainder)"
log "Substage: 4c (3rd of 3 substages)"

# Calculate start and end indices based on equal video distribution
# Substage index: 2 (4c is third substage, gets remainder)
SUBSTAGE_INDEX=2
NUM_SUBSTAGES=3

RANGE_RESULT=$(python -c "
import polars as pl
import sys

try:
    metadata_path = '$SCALED_METADATA'
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
    
    total_videos = df.height
    videos_per_substage = (total_videos + $NUM_SUBSTAGES - 1) // $NUM_SUBSTAGES
    
    start_idx = $SUBSTAGE_INDEX * videos_per_substage
    # Last substage gets all remaining videos (includes remainder)
    end_idx = total_videos
    
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
log "STAGE 4c: VIDEO RANGE ASSIGNMENT"
log "=========================================="
log "This job will process videos in range: [$START_IDX, $END_IDX)"
log "Total videos in this range: $((END_IDX - START_IDX))"
log "Substage: 4c (3rd of 3 substages, includes remainder)"
log "Account: si670f25_class (suzanef)"
log "Resources: 64GB RAM, 1 CPU, 1 GPU, 1 day"
log "=========================================="
log ""

# ============================================================================
# Stage 4c Execution
# ============================================================================

log "=========================================="
log "Starting Stage 4c: Scaled Video Feature Extraction"
log "Processing video range: [$START_IDX, $END_IDX)"
log "=========================================="

NUM_FRAMES="${FVC_NUM_FRAMES:-}"
OUTPUT_DIR="${FVC_STAGE4_OUTPUT_DIR:-data/features_stage4}"
DELETE_EXISTING="${FVC_DELETE_EXISTING:-0}"
RESUME="${FVC_RESUME:-1}"

log "Output directory: $OUTPUT_DIR"
if [ -n "$NUM_FRAMES" ]; then
    log "Number of frames: $NUM_FRAMES (fixed)"
else
    log "Frame sampling: Adaptive (10% of frames, min=5, max=50)"
fi
log "Delete existing: $DELETE_EXISTING"
log "Resume mode: $RESUME"

STAGE4_START=$(date +%s)
LOG_FILE="$ORIG_DIR/logs/stage4c_feat_${SLURM_JOB_ID:-$$}.log"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$ORIG_DIR" || exit 1
PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null || echo "python3")
# Use unbuffered Python for immediate output
PYTHON_CMD="$PYTHON_CMD -u"

# Validate Python script exists
PYTHON_SCRIPT="$ORIG_DIR/src/scripts/run_stage4_scaled_features.py"
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

# Execution order: forward (0) or reverse (1), default to forward
EXECUTION_ORDER="${FVC_EXECUTION_ORDER:-forward}"
if [ "$EXECUTION_ORDER" = "0" ] || [ "$EXECUTION_ORDER" = "forward" ]; then
    EXECUTION_ORDER="forward"
elif [ "$EXECUTION_ORDER" = "1" ] || [ "$EXECUTION_ORDER" = "reverse" ]; then
    EXECUTION_ORDER="reverse"
else
    EXECUTION_ORDER="forward"  # Default
fi

NUM_FRAMES_FLAG=""
if [ -n "$NUM_FRAMES" ]; then
    NUM_FRAMES_FLAG="--num-frames $NUM_FRAMES"
fi

log "Running Stage 4c feature extraction script..."
log "Log file: $LOG_FILE"

if "$PYTHON_CMD" "$PYTHON_SCRIPT" \
    --project-root "$ORIG_DIR" \
    --scaled-metadata "$SCALED_METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --execution-order "$EXECUTION_ORDER" \
    $NUM_FRAMES_FLAG \
    $DELETE_FLAG \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"; then
    
    STAGE4_END=$(date +%s)
    STAGE4_DURATION=$((STAGE4_END - STAGE4_START))
    log "✓ Stage 4c completed successfully in ${STAGE4_DURATION}s ($((${STAGE4_DURATION} / 60)) minutes)"
    log "Results saved to: $ORIG_DIR/$OUTPUT_DIR"
    log "Next step: Run remaining substages if not all complete"
else
    STAGE4_END=$(date +%s)
    STAGE4_DURATION=$((STAGE4_END - STAGE4_START))
    log "✗ ERROR: Stage 4c failed after ${STAGE4_DURATION}s"
    log "Check log file: $LOG_FILE"
    exit 1
fi

log ""
log "============================================================"
log "STAGE 4c EXECUTION SUMMARY"
log "============================================================"
log "Execution time: ${STAGE4_DURATION}s ($((${STAGE4_DURATION} / 60)) minutes)"
log "Video range processed: [$START_IDX, $END_IDX)"
log "Total videos in range: $((END_IDX - START_IDX))"
log "Output directory: $ORIG_DIR/$OUTPUT_DIR"
log "Log file: $LOG_FILE"
log "============================================================"

