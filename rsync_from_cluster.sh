#!/bin/bash
# Rsync script to fetch data, logs, and mlruns from Great Lakes cluster
# This script downloads results and data from the cluster to local machine
# Uses relative paths and preserves directory structure
# Excludes data/augmented_videos and data/scaled_videos to save space

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Cluster connection details
SOURCE_HOST="${SOURCE_HOST:-santoshd@greatlakes.arc-ts.umich.edu}"
SOURCE_PATH="${SOURCE_PATH:-/scratch/si670f25_class_root/si670f25_class/santoshd/fvc/}"
DEST_DIR="$PROJECT_ROOT"

# Create temporary directory for SSH control socket
SSH_CONTROL_DIR="$HOME/.ssh/controlmasters"
mkdir -p "$SSH_CONTROL_DIR"

# Clean up any stale sockets from previous runs (older than 5 minutes)
find "$SSH_CONTROL_DIR" -name "greatlakes_control_*" -type s -mmin +5 -delete 2>/dev/null || true

# Use a unique socket name with PID and timestamp to avoid conflicts
SSH_CONTROL_SOCKET="$SSH_CONTROL_DIR/greatlakes_control_$$_$(date +%s)"

# Cleanup function to close SSH connection
cleanup() {
    if [ -S "$SSH_CONTROL_SOCKET" ] || [ -e "$SSH_CONTROL_SOCKET" ]; then
        ssh -S "$SSH_CONTROL_SOCKET" -O exit "$SOURCE_HOST" 2>/dev/null || true
        rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Ensure socket doesn't exist before we start
rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true

echo "Fetching data, logs, and mlruns from Great Lakes cluster..."
echo "Source: $SOURCE_HOST:$SOURCE_PATH"
echo "Destination: $DEST_DIR"
echo ""
echo "This will fetch:"
echo "  ✓ data/ (excluding augmented_videos and scaled_videos)"
echo "  ✓ logs/"
echo "  ✓ mlruns/"
echo ""
echo "This will NOT fetch:"
echo "  ✗ data/augmented_videos/ (excluded to save space)"
echo "  ✗ data/scaled_videos/ (excluded to save space)"
echo "  ✗ Hidden files/directories (.*) (excluded)"
echo ""

# Use SSH connection sharing for all operations
SSH_OPTS=(-o ControlMaster=yes -o ControlPath="$SSH_CONTROL_SOCKET" -o ControlPersist=60)

# Fetch data/ directory (excluding augmented_videos and scaled_videos)
echo "Fetching data/ directory (excluding augmented_videos and scaled_videos)..."
rsync -avh --progress \
  --relative \
  --exclude='augmented_videos' \
  --exclude='scaled_videos' \
  --exclude='.*' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_HOST:$SOURCE_PATH./data/" \
  "$DEST_DIR/"

# Fetch logs/ directory
echo ""
echo "Fetching logs/ directory..."
rsync -avh --progress \
  --relative \
  --exclude='.*' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_HOST:$SOURCE_PATH./logs/" \
  "$DEST_DIR/"

# Fetch mlruns/ directory
echo ""
echo "Fetching mlruns/ directory..."
rsync -avh --progress \
  --relative \
  --exclude='.*' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_HOST:$SOURCE_PATH./mlruns/" \
  "$DEST_DIR/"

echo ""
echo "✓ Fetch complete!"
echo ""
echo "Fetched directories:"
echo "  - data/ (excluding augmented_videos and scaled_videos)"
echo "  - logs/"
echo "  - mlruns/"

