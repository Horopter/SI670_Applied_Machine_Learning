#!/bin/bash
#
# SLURM Coordinator Script for Stage 3: Video Scaling
#
# This script delegates work to substages (3a, 3b, 3c, 3d, 3e, 3f, 3g, 3h)
# by submitting separate SLURM jobs for each substage.
#
# Usage:
#   sbatch src/scripts/slurm_stage3_scaling.sh
#
# Environment variables (passed to all substages):
#   FVC_TARGET_SIZE: Target max dimension (default: 256)
#   FVC_DOWNSCALE_METHOD: Scaling method (default: autoencoder)
#   FVC_STAGE3_OUTPUT_DIR: Output directory (default: data/scaled_videos)
#   FVC_DELETE_EXISTING: Set to 1/true/yes to delete existing scaled videos (default: 0)
#   FVC_RESUME: Set to 0/false/no to disable resume mode (default: 1)
#   FVC_CHUNK_SIZE: Chunk size for processing (default: 500)
#   FVC_MAX_FRAMES: Max frames per video (default: 500)

#SBATCH --job-name=fvc_stage3_coord
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --time=8:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage3_coord-%j.out
#SBATCH --error=logs/stage3_coord-%j.err
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

# ============================================================================
# Configuration
# ============================================================================

ORIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SUBSTAGE_SCRIPTS=(
    "src/scripts/slurm_stage3a_scaling.sh"
    "src/scripts/slurm_stage3b_scaling.sh"
    "src/scripts/slurm_stage3c_scaling.sh"
    "src/scripts/slurm_stage3d_scaling.sh"
    "src/scripts/slurm_stage3e_scaling.sh"
    "src/scripts/slurm_stage3f_scaling.sh"
    "src/scripts/slurm_stage3g_scaling.sh"
    "src/scripts/slurm_stage3h_scaling.sh"
)

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo "$@" >&1
    echo "$@" >&2
    sync 2>/dev/null || true
}

# ============================================================================
# Main Execution
# ============================================================================

log "=========================================="
log "STAGE 3 COORDINATOR: Video Scaling"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $ORIG_DIR"
log "=========================================="
log ""

# Verify substage scripts exist
log "Verifying substage scripts..."
MISSING_SCRIPTS=()
for script in "${SUBSTAGE_SCRIPTS[@]}"; do
    script_path="$ORIG_DIR/$script"
    if [ ! -f "$script_path" ]; then
        MISSING_SCRIPTS+=("$script")
        log "✗ Missing: $script"
    else
        log "✓ Found: $script"
    fi
done

if [ ${#MISSING_SCRIPTS[@]} -gt 0 ]; then
    log "✗ ERROR: Missing substage scripts: ${MISSING_SCRIPTS[*]}"
    exit 1
fi

log ""
log "=========================================="
log "Submitting substage jobs..."
log "=========================================="

# Change to project root
cd "$ORIG_DIR" || exit 1

# Verify sbatch command is available
if ! command -v sbatch &> /dev/null; then
    log "✗ ERROR: sbatch command not found. This script must be run on a SLURM cluster."
    exit 1
fi

# Collect environment variables to pass to substages
ENV_VARS=()
if [ -n "${FVC_TARGET_SIZE:-}" ]; then
    ENV_VARS+=("FVC_TARGET_SIZE=$FVC_TARGET_SIZE")
fi
if [ -n "${FVC_DOWNSCALE_METHOD:-}" ]; then
    ENV_VARS+=("FVC_DOWNSCALE_METHOD=$FVC_DOWNSCALE_METHOD")
fi
if [ -n "${FVC_STAGE3_OUTPUT_DIR:-}" ]; then
    ENV_VARS+=("FVC_STAGE3_OUTPUT_DIR=$FVC_STAGE3_OUTPUT_DIR")
fi
if [ -n "${FVC_DELETE_EXISTING:-}" ]; then
    ENV_VARS+=("FVC_DELETE_EXISTING=$FVC_DELETE_EXISTING")
fi
if [ -n "${FVC_RESUME:-}" ]; then
    ENV_VARS+=("FVC_RESUME=$FVC_RESUME")
fi
if [ -n "${FVC_CHUNK_SIZE:-}" ]; then
    ENV_VARS+=("FVC_CHUNK_SIZE=$FVC_CHUNK_SIZE")
fi
if [ -n "${FVC_MAX_FRAMES:-}" ]; then
    ENV_VARS+=("FVC_MAX_FRAMES=$FVC_MAX_FRAMES")
fi
if [ -n "${FVC_STAGE1_OUTPUT_DIR:-}" ]; then
    ENV_VARS+=("FVC_STAGE1_OUTPUT_DIR=$FVC_STAGE1_OUTPUT_DIR")
fi

# Submit all substage jobs
SUBMITTED_JOBS=()
FAILED_SUBMISSIONS=()

for script in "${SUBSTAGE_SCRIPTS[@]}"; do
    substage_name=$(basename "$script" .sh | sed 's/slurm_stage3/stage3/')
    log "Submitting $substage_name..."
    
    # Build sbatch command with environment variables
    # Combine all env vars into a single --export argument
    if [ ${#ENV_VARS[@]} -gt 0 ]; then
        EXPORT_VARS="ALL"
        for env_var in "${ENV_VARS[@]}"; do
            EXPORT_VARS="$EXPORT_VARS,$env_var"
        done
        SBATCH_CMD=("sbatch" "--export=$EXPORT_VARS" "$script")
    else
        SBATCH_CMD=("sbatch" "--export=ALL" "$script")
    fi
    
    if SBATCH_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1); then
        JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' || echo "")
        if [ -n "$JOB_ID" ]; then
            SUBMITTED_JOBS+=("$JOB_ID")
            log "  ✓ Submitted $substage_name as job $JOB_ID"
        else
            log "  ✗ Failed to submit $substage_name (could not parse job ID)"
            log "    Output: $SBATCH_OUTPUT"
            FAILED_SUBMISSIONS+=("$substage_name")
        fi
    else
        log "  ✗ Failed to submit $substage_name"
        FAILED_SUBMISSIONS+=("$substage_name")
    fi
done

log ""
log "=========================================="
log "Submission Summary"
log "=========================================="
log "Total substages: ${#SUBSTAGE_SCRIPTS[@]}"
log "Successfully submitted: ${#SUBMITTED_JOBS[@]}"
log "Failed submissions: ${#FAILED_SUBMISSIONS[@]}"

if [ ${#FAILED_SUBMISSIONS[@]} -gt 0 ]; then
    log ""
    log "✗ ERROR: Failed to submit the following substages:"
    for failed in "${FAILED_SUBMISSIONS[@]}"; do
        log "  - $failed"
    done
    exit 1
fi

if [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    log ""
    log "✓ Successfully submitted all substage jobs:"
    for i in "${!SUBMITTED_JOBS[@]}"; do
        substage_name=$(basename "${SUBSTAGE_SCRIPTS[$i]}" .sh | sed 's/slurm_stage3/stage3/')
        log "  - $substage_name: Job ID ${SUBMITTED_JOBS[$i]}"
    done
    log ""
    log "Monitor jobs with:"
    log "  squeue -u \$USER"
    log ""
    log "Check job status with:"
    for i in "${!SUBMITTED_JOBS[@]}"; do
        log "  squeue -j ${SUBMITTED_JOBS[$i]}"
    done
fi

log ""
log "=========================================="
log "STAGE 3 COORDINATOR COMPLETE"
log "=========================================="
log "All substage jobs have been submitted."
log "Stage 3 will complete when all substages (3a-3h) finish."
log "Next step: After Stage 3 completes, run Stage 4 with:"
log "  sbatch src/scripts/slurm_stage4_scaled_features.sh"
log "=========================================="
