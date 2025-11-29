#!/bin/bash
#
# Simple one-video-at-a-time preprocessing script
#
# Usage:
#   sbatch src/scripts/preprocess_videos.sh
#   # Or with custom settings:
#   NUM_AUGMENTATIONS=5 sbatch src/scripts/preprocess_videos.sh
#

#SBATCH --job-name=fvc_preprocess
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail

# Unset macOS malloc warnings
unset MallocStackLogging || true
unset MallocStackLoggingNoCompact || true

# Suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02
module load cuda/12.1 || true

# Directory setup
mkdir -p logs
export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export ORIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export VENV_DIR="$ORIG_DIR/venv"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated virtual environment: $VENV_DIR"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

# ============================================================================
# Preprocessing Configuration
# ============================================================================

# Number of augmentations per video (default: 1, can be 1, 2, 5, or 10)
NUM_AUGMENTATIONS="${NUM_AUGMENTATIONS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-videos_augmented}"

echo "=========================================="
echo "VIDEO PREPROCESSING"
echo "=========================================="
echo "Project root: $ORIG_DIR"
echo "Number of augmentations per video: $NUM_AUGMENTATIONS"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# ============================================================================
# Run Preprocessing
# ============================================================================

cd "$ORIG_DIR"

python src/preprocess_videos_simple.py \
    --project-root "$ORIG_DIR" \
    --num-augmentations "$NUM_AUGMENTATIONS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="

