#!/bin/bash

# Usage: ./retain_latest_triplets.sh [relative_path]
# Default: ./logs/stage5

# Set path
TARGET_DIR="${1:-./logs/stage5}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: Directory '$TARGET_DIR' does not exist."
  exit 1
fi

echo "Working in: $TARGET_DIR"
cd "$TARGET_DIR"

shopt -s nullglob

declare -A kind2maxjob

# Step 1: Find latest jobid for each kind
for file in *.{log,out,err}; do
  # Handle both - and _ as separators before jobid
  if [[ "$file" =~ ^(.*?)[-_]([0-9]+)\.(log|out|err)$ ]]; then
    kind="${BASH_REMATCH[1]}"
    jobid="${BASH_REMATCH[2]}"
    if [[ -z "${kind2maxjob[$kind]}" || "$jobid" -gt "${kind2maxjob[$kind]}" ]]; then
      kind2maxjob[$kind]="$jobid"
    fi
  fi
done

# Step 2: Delete all except latest triplet for each kind
for file in *.{log,out,err}; do
  if [[ "$file" =~ ^(.*?)[-_]([0-9]+)\.(log|out|err)$ ]]; then
    kind="${BASH_REMATCH[1]}"
    jobid="${BASH_REMATCH[2]}"
    maxjob="${kind2maxjob[$kind]}"
    if [[ "$jobid" != "$maxjob" ]]; then
      echo "Deleting $file"
      rm "$file"
    fi
  fi
done

shopt -u nullglob

echo "Retention complete in $TARGET_DIR."