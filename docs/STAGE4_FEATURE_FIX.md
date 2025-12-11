# Stage 4 Feature Fix - Action Plan

## Issues Identified

### 1. Stage 2 Arrow File Corrupted
**Problem**: `features_metadata.arrow` exists but cannot be read (os error 22)
**Root Cause**: Arrow file is corrupted or in invalid format
**Solution**: Automatic reconstruction from parquet files (already implemented)

### 2. Stage 4 Missing Base Features
**Problem**: Stage 4 only has 6 features instead of 23
- Current: 6 scaled-specific features only
- Expected: 15 base + 6 scaled-specific + 2 indicators = 23 features

**Root Cause**: `extract_scaled_features()` was only extracting scaled-specific features, not the base handcrafted features

**Fix Applied**: Modified `extract_scaled_features()` to:
- Extract 15 base handcrafted features using `extract_all_features()`
- Extract 6 scaled-specific features
- Combine both (total 21 features)
- Add `is_upscaled` and `is_downscaled` separately (total 23)

### 3. Feature File Shape Mismatch
**Problem**: Feature files have shape (1, 8) instead of aggregated features
**Root Cause**: Files are storing frame-level features (8 frames) instead of aggregated
**Fix Applied**: Updated reconstruction script to aggregate frame-level features by mean

## What Was Fixed

### 1. `lib/features/scaled.py`
- ✅ Modified `extract_scaled_features()` to extract BOTH base and scaled-specific features
- ✅ Now extracts 15 base features + 6 scaled-specific = 21 features
- ✅ Combined with `is_upscaled` and `is_downscaled` = 23 total features

### 2. `src/scripts/reconstruct_metadata.py`
- ✅ Improved handling of frame-level features (shape 1, 8)
- ✅ Aggregates nested arrays by mean
- ✅ Better video_id extraction from filenames

### 3. `src/scripts/sanity_check_features.py`
- ✅ Better error handling for corrupted Arrow files
- ✅ Validates file has data before considering it valid

## What Needs to Be Done Next

### Option 1: Re-run Stage 4 (Recommended)
**Action**: Re-run Stage 4 to regenerate features with the fix
```bash
# This will regenerate all Stage 4 features with 23 features each
python src/scripts/run_stage4_scaled_features.py \
    --project-root /path/to/project \
    --scaled-metadata data/scaled_videos/scaled_metadata.parquet \
    --output-dir data/features_stage4 \
    --delete-existing  # Delete old 6-feature files
```

**Pros**:
- Clean regeneration with correct 23 features
- All videos will have consistent feature counts
- No need to reconstruct metadata

**Cons**:
- Takes time to re-process all videos
- Requires scaled videos from Stage 3

### Option 2: Reconstruct Stage 4 Metadata from Existing Files
**Action**: Use reconstruction script to combine existing parquet files
```bash
python src/scripts/reconstruct_metadata.py \
    --features-dir data/features_stage4 \
    --output-filename features_scaled_metadata.parquet \
    --pattern "*_scaled_features.parquet"
```

**Note**: This will work, but the existing parquet files only have 6 features, so the reconstructed metadata will also only have 6 features. You'll still need to re-run Stage 4 to get the full 23 features.

### Option 3: Hybrid Approach (Best for Production)
1. **Reconstruct Stage 2 metadata** from parquet files (fixes corrupted Arrow file)
2. **Re-run Stage 4** for a subset to verify the fix works
3. **If verified, re-run Stage 4** for all videos

## Recommended Next Steps

1. **Immediate**: Reconstruct Stage 2 metadata
   ```bash
   python src/scripts/reconstruct_metadata.py \
       --features-dir data/features_stage2 \
       --output-filename features_metadata.parquet \
       --pattern "*_features.parquet"
   ```

2. **Verify Stage 4 fix**: Re-run Stage 4 on a small subset
   ```bash
   python src/scripts/run_stage4_scaled_features.py \
       --project-root /path/to/project \
       --scaled-metadata data/scaled_videos/scaled_metadata.parquet \
       --start-idx 0 \
       --end-idx 10 \
       --delete-existing
   ```
   
   Then check:
   ```bash
   python src/scripts/sanity_check_features.py
   ```
   
   Should show 23 features for Stage 4.

3. **If verified**: Re-run Stage 4 for all videos
   ```bash
   python src/scripts/run_stage4_scaled_features.py \
       --project-root /path/to/project \
       --scaled-metadata data/scaled_videos/scaled_metadata.parquet \
       --delete-existing
   ```

## Verification

After fixes, run sanity check:
```bash
python src/scripts/sanity_check_features.py
```

Expected output:
- ✅ Stage 2: 23 features (or 15 if codec cues unavailable)
- ✅ Stage 4: 23 features (15 base + 6 scaled + 2 indicators)
- ✅ Feature files have correct shapes
- ✅ No corrupted files

## Feature Count Breakdown

### Stage 2 (15 features)
- Noise residual: 3 (noise_energy, noise_mean, noise_std)
- DCT statistics: 5 (dct_dc_mean, dct_dc_std, dct_ac_mean, dct_ac_std, dct_ac_energy)
- Blur/sharpness: 3 (laplacian_var, gradient_mean, gradient_std)
- Boundary inconsistency: 1 (boundary_inconsistency)
- Codec cues: 3 (codec_bitrate, codec_fps, codec_resolution)

### Stage 4 (23 features)
- **Base features (15)**: Same as Stage 2
- **Scaled-specific (6)**: edge_density, texture_uniformity, color_consistency_r/g/b, compression_artifacts
- **Scaling indicators (2)**: is_upscaled, is_downscaled

**Total: 15 + 6 + 2 = 23 features**

