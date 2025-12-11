#!/bin/bash
#
# SLURM Coordinator Script for Stage 4: Scaled Video Feature Extraction
#
# This script delegates work to substages (4a, 4b, 4c)
# by submitting separate SLURM jobs for each substage.
#
# Usage:
#   sbatch src/scripts/slurm_stage4_scaled_features.sh
#
# Environment variables (passed to all substages):
#   FVC_NUM_FRAMES: Number of frames to sample per video (default: 8)
#   FVC_STAGE4_OUTPUT_DIR: Output directory (default: data/features_stage4)
#   FVC_DELETE_EXISTING: Set to 1/true/yes to delete existing features (default: 0)
#   FVC_RESUME: Set to 0/false/no to disable resume mode (default: 1)
#   FVC_STAGE3_OUTPUT_DIR: Stage 3 output directory (default: data/scaled_videos)

#SBATCH --job-name=fvc_stage4_coord
#SBATCH --account=si670f25_class
#SBATCH --partition=standard
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/stage4_coord-%j.out
#SBATCH --error=logs/stage4_coord-%j.err
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
    "src/scripts/slurm_stage4a_scaled_features.sh"
    "src/scripts/slurm_stage4b_scaled_features.sh"
    "src/scripts/slurm_stage4c_scaled_features.sh"
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
log "STAGE 4 COORDINATOR: Scaled Video Feature Extraction"
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
if [ -n "${FVC_NUM_FRAMES:-}" ]; then
    ENV_VARS+=("FVC_NUM_FRAMES=$FVC_NUM_FRAMES")
fi
if [ -n "${FVC_STAGE4_OUTPUT_DIR:-}" ]; then
    ENV_VARS+=("FVC_STAGE4_OUTPUT_DIR=$FVC_STAGE4_OUTPUT_DIR")
fi
if [ -n "${FVC_DELETE_EXISTING:-}" ]; then
    ENV_VARS+=("FVC_DELETE_EXISTING=$FVC_DELETE_EXISTING")
fi
if [ -n "${FVC_RESUME:-}" ]; then
    ENV_VARS+=("FVC_RESUME=$FVC_RESUME")
fi
if [ -n "${FVC_STAGE3_OUTPUT_DIR:-}" ]; then
    ENV_VARS+=("FVC_STAGE3_OUTPUT_DIR=$FVC_STAGE3_OUTPUT_DIR")
fi

# Submit all substage jobs
SUBMITTED_JOBS=()
FAILED_SUBMISSIONS=()

for script in "${SUBSTAGE_SCRIPTS[@]}"; do
    substage_name=$(basename "$script" .sh | sed 's/slurm_stage4/stage4/')
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
        substage_name=$(basename "${SUBSTAGE_SCRIPTS[$i]}" .sh | sed 's/slurm_stage4/stage4/')
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
log "STAGE 4 COORDINATOR COMPLETE"
log "=========================================="
log "All substage jobs have been submitted."
log "Stage 4 will complete when all substages (4a-4c) finish."
log "Next step: After Stage 4 completes, run Stage 5 with:"
log "  sbatch src/scripts/slurm_stage5_training.sh"
log "=========================================="
