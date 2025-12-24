#!/bin/bash
# Rsync script to fetch stage5 data, logs, and mlruns from Great Lakes cluster
# This script downloads stage5 training results, logs, and MLflow tracking from the cluster
# Uses relative paths and preserves directory structure
# Only fetches stage5-related content to save space and time

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

echo "Fetching stage5 data, logs, and mlruns from Great Lakes cluster..."
echo "Source: $SOURCE_HOST:$SOURCE_PATH"
echo "Destination: $DEST_DIR"
echo ""
echo "This will fetch:"
echo "  ✓ data/stage5/ (stage5 training results only)"
echo "  ✓ logs/stage5/ (stage5 training logs only)"
echo "  ✓ mlruns/ (MLflow experiment tracking)"
echo ""
echo "This will NOT fetch:"
echo "  ✗ data/stage1-4/ (excluded)"
echo "  ✗ logs/stage1-4/ (excluded)"
echo "  ✗ logs/validation/ (excluded)"
echo "  ✗ logs/repairs/ (excluded)"
echo "  ✗ logs/sanity_checks/ (excluded)"
echo "  ✗ Hidden files/directories (.*) (excluded)"
echo ""

# Use SSH connection sharing for all operations
SSH_OPTS=(-o ControlMaster=yes -o ControlPath="$SSH_CONTROL_SOCKET" -o ControlPersist=60)

# Fetch data/stage5/ directory only
echo "Fetching data/stage5/ directory..."
rsync -avh --progress \
  --relative \
  --exclude='.*' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_HOST:$SOURCE_PATH./data/stage5/" \
  "$DEST_DIR/"

# Fetch logs/stage5/ directory only
echo ""
echo "Fetching logs/stage5/ directory..."
rsync -avh --progress \
  --relative \
  --exclude='.*' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_HOST:$SOURCE_PATH./logs/stage5/" \
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
echo "  - data/stage5/ (stage5 training results)"
echo "  - logs/stage5/ (stage5 training logs)"
echo "  - mlruns/ (MLflow experiment tracking)"

