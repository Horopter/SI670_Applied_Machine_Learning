# Stage 4 Re-run Steps

## Issue Fixed
Stage 4 was only extracting 6 scaled-specific features instead of 23 total features (15 base + 6 scaled + 2 indicators).

## Solution Applied
Modified `extract_scaled_features()` to extract BOTH base handcrafted features AND scaled-specific features.

## Next Steps

### Step 1: Test on Small Subset (10 videos)

```bash
python src/scripts/run_stage4_scaled_features.py \
    --project-root . \
    --scaled-metadata data/scaled_videos/scaled_metadata.parquet \
    --start-idx 0 \
    --end-idx 10 \
    --delete-existing
```

**Note**: The script will automatically find the metadata file if it exists in `.arrow` or `.csv` format.

**What to expect**:
- Verbose logging showing each video being processed
- Feature extraction details (15 base + 6 scaled = 21 features)
- Scaling indicators added (is_upscaled, is_downscaled)
- Total: 23 features per video
- Progress updates every 10 videos

### Step 2: Verify Features

```bash
python src/scripts/sanity_check_features.py
```

**Expected output**:
- ✅ Stage 2: 23 features (or 15 if codec cues unavailable)
- ✅ Stage 4: **23 features** (15 base + 6 scaled + 2 indicators)
- ✅ Feature files have correct shapes
- ✅ No corrupted files

**If Stage 4 shows 6 features instead of 23**:
- The old feature files are still there
- Re-run Step 1 with `--delete-existing` to regenerate

### Step 3: Re-run for All Videos (if Step 2 passes)

```bash
python src/scripts/run_stage4_scaled_features.py \
    --project-root . \
    --scaled-metadata data/scaled_videos/scaled_metadata.parquet \
    --delete-existing
```

**This will**:
- Process all videos in the metadata
- Delete old 6-feature files
- Generate new 23-feature files
- Take time depending on dataset size

### Step 4: Final Verification

```bash
python src/scripts/sanity_check_features.py
```

**Should show**:
- ✅ Stage 4: 23 features for all videos
- ✅ All feature files have correct aggregated shapes
- ✅ Metadata file is valid

## Troubleshooting

### If metadata file not found:
The script will automatically:
1. Check for `.arrow` and `.csv` versions
2. List available files in the directory
3. Suggest which file to use

### If you see "6 features" in sanity check:
- Old feature files weren't deleted
- Re-run with `--delete-existing` flag

### If feature extraction fails:
- Check the log file: `logs/stage4_scaled_features_<timestamp>.log`
- Look for specific error messages
- Verify video files exist and are valid

## Feature Count Breakdown

### Stage 2 (15 features)
- Noise residual: 3
- DCT statistics: 5
- Blur/sharpness: 3
- Boundary inconsistency: 1
- Codec cues: 3

### Stage 4 (23 features)
- **Base features (15)**: Same as Stage 2
- **Scaled-specific (6)**: edge_density, texture_uniformity, color_consistency_r/g/b, compression_artifacts
- **Scaling indicators (2)**: is_upscaled, is_downscaled

**Total: 15 + 6 + 2 = 23 features** ✅

