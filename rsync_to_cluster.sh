#!/bin/bash
# Safe rsync script to sync AURA project code to Great Lakes cluster
# SAFETY: Only syncs lib/ and src/ directories, never deletes files
# This script only adds/updates files, never deletes anything
# Uses SSH connection sharing to avoid multiple password prompts

SOURCE_DIR="/Users/santoshdesai/Downloads/fvc/"
DEST_HOST="santoshd@greatlakes.arc-ts.umich.edu"
DEST_PATH="/scratch/si670f25_class_root/si670f25_class/santoshd/fvc/"

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
        ssh -S "$SSH_CONTROL_SOCKET" -O exit "$DEST_HOST" 2>/dev/null || true
        rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Ensure socket doesn't exist before we start (shouldn't with unique name, but be safe)
rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true

echo "Syncing AURA project code to Great Lakes cluster..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_HOST:$DEST_PATH"
echo ""
echo "✓ SAFETY: This script:"
echo "   - Only syncs lib/ and src/ directories"
echo "   - Does NOT use --delete flag (won't remove any files)"
echo "   - Will NOT touch: data/, archive/, models/, logs/, venv/, etc."
echo "   - Only adds/updates files, never deletes"
echo "   - Uses SSH connection sharing (single password prompt)"
echo ""

# Use SSH connection sharing for all operations
SSH_OPTS=(-o ControlMaster=yes -o ControlPath="$SSH_CONTROL_SOCKET" -o ControlPersist=60)

# Sync lib/ directory WITHOUT --delete (safest option)
echo "Syncing lib/ directory (updates only, no deletion)..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_DIR/lib/" \
  "$DEST_HOST:$DEST_PATH/lib/"

# Sync src/ directory WITHOUT --delete (reuses SSH connection)
echo ""
echo "Syncing src/ directory (updates only, no deletion)..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_DIR/src/" \
  "$DEST_HOST:$DEST_PATH/src/"

# Clear Python cache (reuses SSH connection)
echo ""
echo "Clearing Python cache..."
ssh -T "${SSH_OPTS[@]}" "$DEST_HOST" << 'ENDSSH'
    DEST_PATH="/scratch/si670f25_class_root/si670f25_class/santoshd/fvc"
    
    # Clear Python cache
    find "$DEST_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$DEST_PATH" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "$DEST_PATH" -type f -name "*.pyo" -delete 2>/dev/null || true
    find "$DEST_PATH" -type f -name "*.pyd" -delete 2>/dev/null || true
    echo "✓ Python cache cleared"
ENDSSH

echo ""
echo "Sync complete!"
