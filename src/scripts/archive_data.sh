#!/bin/bash
#
# Archive data folder to archive/data.tar.gz
# Excludes data/scaled_videos folder to save space
#
# Usage:
#   bash src/scripts/archive_data.sh
#   OR
#   bash src/scripts/archive_data.sh update  # Update existing archive

set -euo pipefail

ORIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ARCHIVE_DIR="$ORIG_DIR/archive"
ARCHIVE_FILE="$ARCHIVE_DIR/data.tar.gz"
DATA_DIR="$ORIG_DIR/data"

# Create archive directory if it doesn't exist
mkdir -p "$ARCHIVE_DIR"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "✗ ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Check if update mode
if [ "${1:-}" = "update" ] && [ -f "$ARCHIVE_FILE" ]; then
    echo "Updating existing archive: $ARCHIVE_FILE"
    echo "Excluding: data/scaled_videos"
    
    # For update, we need to recreate the archive (tar doesn't support selective updates well)
    # So we'll extract, merge, and recreate
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    # Extract existing archive
    echo "Extracting existing archive..."
    tar xzf "$ARCHIVE_FILE" -C "$TEMP_DIR" 2>/dev/null || true
    
    # Copy current data (excluding scaled_videos) to temp
    echo "Copying current data (excluding scaled_videos)..."
    rsync -av --exclude='scaled_videos' "$DATA_DIR/" "$TEMP_DIR/data/" || \
        cp -r "$DATA_DIR"/* "$TEMP_DIR/data/" 2>/dev/null || true
    
    # Remove scaled_videos if it exists in temp
    rm -rf "$TEMP_DIR/data/scaled_videos" 2>/dev/null || true
    
    # Create new archive
    echo "Creating updated archive..."
    cd "$TEMP_DIR"
    tar czf "$ARCHIVE_FILE" data/
    cd "$ORIG_DIR"
    
    echo "✓ Archive updated: $ARCHIVE_FILE"
else
    echo "Creating new archive: $ARCHIVE_FILE"
    echo "Excluding: data/scaled_videos"
    
    # Create archive excluding scaled_videos
    cd "$ORIG_DIR"
    tar czf "$ARCHIVE_FILE" \
        --exclude='data/scaled_videos' \
        --exclude='data/scaled_videos/*' \
        ./data/
    
    echo "✓ Archive created: $ARCHIVE_FILE"
    
    # Show archive size
    if [ -f "$ARCHIVE_FILE" ]; then
        SIZE=$(du -h "$ARCHIVE_FILE" | cut -f1)
        echo "  Archive size: $SIZE"
    fi
fi

echo ""
echo "Archive contents (excluding scaled_videos):"
tar tzf "$ARCHIVE_FILE" | head -20
echo "..."
echo ""
echo "Total files in archive: $(tar tzf "$ARCHIVE_FILE" | wc -l)"
