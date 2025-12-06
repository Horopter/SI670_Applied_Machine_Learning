# FVC Project Architecture Diagrams

This document provides detailed ASCII/text diagrams of the 5-stage pipeline architecture, data flow, and system components.

## 5-Stage Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FVC 5-STAGE PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

N original videos (298)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: VIDEO AUGMENTATION                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│  Input:  N original videos (298)                                        │
│  Output: 11N videos (N original + 10N augmented = 3,244 actual)        │
│                                                                          │
│  Process:                                                                │
│    For each video:                                                       │
│      1. Load video metadata (frame count, fps, dimensions)              │
│      2. Sample frames uniformly (6-8 frames)                            │
│      3. Apply 10 augmentations:                                         │
│         - Rotation, Flip, Brightness, Contrast, Saturation              │
│         - Gaussian Noise, Gaussian Blur, Affine, Elastic                │
│      4. Save augmented videos as MP4                                    │
│      5. Write metadata row to CSV/Arrow                                  │
│                                                                          │
│  Key Features:                                                           │
│    • Frame-by-frame decoding (50x memory reduction)                     │
│    • Deterministic seeding (reproducible)                               │
│    • Chunked processing for long videos                                  │
│    • Resume capability                                                   │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
11N videos (3,244)
    │
    ├──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    ▼                                                                      ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│  STAGE 2: HANDCRAFTED FEATURES       │    │  STAGE 3: VIDEO SCALING               │
│  ──────────────────────────────────  │    │  ──────────────────────────────────  │
│  Input:  11N videos (3,244)         │    │  Input:  11N videos (3,244)          │
│  Output: M features per video (~15)  │    │  Output: 11N scaled videos (256px)   │
│                                      │    │                                      │
│  Features Extracted:                 │    │  Scaling Methods:                    │
│    • Noise residual (3 features)     │    │    • Autoencoder (default)           │
│    • DCT statistics (5 features)    │    │    • Letterbox resize (fallback)    │
│    • Blur/sharpness (3 features)    │    │                                      │
│    • Boundary inconsistency (1)     │    │  Process:                             │
│    • Codec cues (3 features)        │    │    For each video:                     │
│                                      │    │      1. Load video                    │
│  Process:                             │    │      2. Process in chunks (100 frames)│
│    For each video:                    │    │      3. Scale frames (autoencoder or │
│      1. Sample frames (10%, min=5,    │    │         letterbox)                    │
│         max=50)                        │    │      4. Concatenate scaled chunks    │
│      2. Extract features per frame    │    │      5. Save scaled video             │
│      3. Aggregate (mean) across frames │    │      6. Write metadata                │
│      4. Save as .npy or .parquet      │    │                                      │
│      5. Write metadata row            │    │  Key Features:                        │
│                                      │    │    • Chunked processing (100 frames)   │
│  Key Features:                       │    │    • OOM detection & fallback         │
│    • Adaptive frame sampling          │    │    • Aspect ratio preservation        │
│    • Frame aggregation                │    │    • Resume capability                │
│    • Resume capability                │    │    • Scaling direction tracking       │
└──────────────────────────────────────┘    └──────────────────────────────────────┘
    │                                                                      │
    │                                                                      ▼
    │                                                            11N scaled videos
    │                                                                      │
    │                                                                      ▼
    │                                                          ┌──────────────────────────────────────┐
    │                                                          │  STAGE 4: SCALED VIDEO FEATURES     │
    │                                                          │  ──────────────────────────────────  │
    │                                                          │  Input:  11N scaled videos          │
    │                                                          │  Output: P features per video (~8)  │
    │                                                          │                                      │
    │                                                          │  Features Extracted:                │
    │                                                          │    • Edge preservation (1)          │
    │                                                          │    • Texture uniformity (1)         │
    │                                                          │    • Color consistency (3)          │
    │                                                          │    • Compression artifacts (1)      │
    │                                                          │    • Scaling direction (2)          │
    │                                                          │                                      │
    │                                                          │  Process:                             │
    │                                                          │    Similar to Stage 2                │
    │                                                          │                                      │
    │                                                          │  Key Features:                        │
    │                                                          │    • Adaptive frame sampling          │
    │                                                          │    • Scaling direction detection     │
    │                                                          │    • Resume capability                │
    │                                                          └──────────────────────────────────────┘
    │                                                                      │
    │                                                                      ▼
    │                                                              P features (~8)
    │                                                                      │
    ▼                                                                      │
M features (~15)                                                          │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │  STAGE 5: MODEL TRAINING             │
                    │  ──────────────────────────────────  │
                    │  Input:                              │
                    │    • 11N scaled videos (256×256)     │
                    │    • M features from Stage 2 (~15)   │
                    │    • P features from Stage 4 (~8)     │
                    │                                      │
                    │  Output: Trained models + metrics    │
                    │                                      │
                    │  Process:                             │
                    │    1. Load all inputs                │
                    │    2. Create 5-fold stratified splits │
                    │    3. For each model (7 models):      │
                    │       For each fold (5 folds):        │
                    │         a. Combine features           │
                    │         b. Remove collinear features  │
                    │         c. Train model                │
                    │         d. Evaluate on validation     │
                    │         e. Save checkpoint            │
                    │       f. Aggregate fold results       │
                    │    4. Compare all models              │
                    │                                      │
                    │  Models:                              │
                    │    • Logistic Regression              │
                    │    • Linear SVM                       │
                    │    • Naive CNN                        │
                    │    • ViT-B/16 + GRU                   │
                    │    • ViT-B/16 + Transformer           │
                    │    • SlowFast                         │
                    │    • X3D                              │
                    │                                      │
                    │  Key Features:                        │
                    │    • K-fold cross-validation          │
                    │    • Multi-model training             │
                    │    • Resume capability                │
                    │    • Experiment tracking              │
                    └──────────────────────────────────────┘
                                    │
                                    ▼
                            Trained Models
                            + Evaluation Metrics
```

## Data Flow Diagram

```
┌──────────────┐
│ Original     │
│ Videos (298) │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: AUGMENTATION                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Video 1  │→ │ Video 1 │  │ Video 1 │  │ Video 1 │ ... │
│  │ Original │  │ Aug 0   │  │ Aug 1   │  │ Aug 9   │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│                                                              │
│  Output: augmented_metadata.arrow (3,244 rows)               │
└─────────────────────────────────────────────────────────────┘
       │                              │
       │                              │
       ▼                              ▼
┌──────────────────────┐    ┌──────────────────────┐
│  STAGE 2: FEATURES   │    │  STAGE 3: SCALING    │
│  ─────────────────── │    │  ─────────────────── │
│  Input: 3,244 videos │    │  Input: 3,244 videos │
│                      │    │                      │
│  For each video:     │    │  For each video:     │
│    • Sample frames   │    │    • Load video      │
│    • Extract:        │    │    • Scale to 256px  │
│      - Noise (3)     │    │    • Save scaled    │
│      - DCT (5)       │    │                      │
│      - Blur (3)      │    │  Output:             │
│      - Boundary (1)  │    │  scaled_metadata.arrow│
│      - Codec (3)     │    │  (3,244 rows)        │
│    • Aggregate       │    │                      │
│    • Save .npy       │    │                      │
│                      │    │                      │
│  Output:            │    │                      │
│  features_metadata. │    │                      │
│  arrow (3,244 rows) │    │                      │
└──────────────────────┘    └──────────────────────┘
       │                              │
       │                              ▼
       │                      ┌──────────────────────┐
       │                      │  STAGE 4: SCALED     │
       │                      │  FEATURES            │
       │                      │  ─────────────────── │
       │                      │  Input: 3,244 scaled│
       │                      │                      │
       │                      │  For each video:     │
       │                      │    • Sample frames   │
       │                      │    • Extract:       │
       │                      │      - Edge (1)     │
       │                      │      - Texture (1)  │
       │                      │      - Color (3)    │
       │                      │      - Compression(1)│
       │                      │      - Scaling (2)   │
       │                      │    • Aggregate      │
       │                      │    • Save .parquet  │
       │                      │                      │
       │                      │  Output:            │
       │                      │  features_scaled_   │
       │                      │  metadata.arrow     │
       │                      │  (3,244 rows)        │
       │                      └──────────────────────┘
       │                              │
       └──────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────────┐
          │  STAGE 5: TRAINING        │
          │  ───────────────────────  │
          │  Input:                   │
          │    • 3,244 scaled videos  │
          │    • M features (~15)     │
          │    • P features (~8)     │
          │                           │
          │  Process:                  │
          │    1. Combine features     │
          │    2. 5-fold CV splits     │
          │    3. Train 7 models:     │
          │       - Logistic Reg.     │
          │       - SVM                │
          │       - Naive CNN          │
          │       - ViT+GRU            │
          │       - ViT+Transformer    │
          │       - SlowFast           │
          │       - X3D                │
          │    4. Evaluate & compare   │
          │                           │
          │  Output:                  │
          │    • Trained models        │
          │    • Evaluation metrics    │
          │    • Comparison results    │
          └───────────────────────────┘
```

## Multi-Node Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLURM CLUSTER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Node 0     │  │   Node 1     │  │   Node 2     │  ...     │
│  │  (PROCID=0)  │  │  (PROCID=1)  │  │  (PROCID=2)  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         │                 │                 │                   │
│  ┌──────▼─────────────────▼─────────────────▼──────┐          │
│  │         Dynamic Load Distribution                │          │
│  │  ─────────────────────────────────────────────  │          │
│  │  Total videos: 3,244                             │          │
│  │  Number of nodes: 3                              │          │          │
│  │                                                   │          │
│  │  Node 0: videos [0, 1081)      (1,081 videos)    │          │
│  │  Node 1: videos [1081, 2162)   (1,081 videos)     │          │
│  │  Node 2: videos [2162, 3244)   (1,082 videos)    │          │
│  │                                                   │          │
│  │  Formula:                                         │          │
│  │    start = (PROCID * total) // NTASKS            │          │
│  │    end = ((PROCID + 1) * total) // NTASKS       │          │
│  └───────────────────────────────────────────────────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐              │
│  │ Process  │      │ Process  │      │ Process  │              │
│  │ Range    │      │ Range    │      │ Range    │              │
│  │ [0,1081) │      │ [1081,   │      │ [2162,   │              │
│  │          │      │  2162)   │      │  3244)   │              │
│  └──────────┘      └──────────┘      └──────────┘              │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐              │
│  │ Log File │      │ Log File │      │ Log File │              │
│  │ node0.log│      │ node1.log│      │ node2.log│              │
│  └──────────┘      └──────────┘      └──────────┘              │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│              ┌─────────────────────────┐                        │
│              │ Combined Log File       │                        │
│              │ combined_${JOB_ID}.log  │                        │
│              └─────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## Model Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURES                          │
└─────────────────────────────────────────────────────────────────┘

BASELINE MODELS (Features Only)
┌─────────────────────┐    ┌─────────────────────┐
│ Logistic Regression │    │ Linear SVM          │
│ ─────────────────── │    │ ─────────────────── │
│ Input: Features     │    │ Input: Features     │
│ (M + P = ~23)       │    │ (M + P = ~23)       │
│                     │    │                     │
│ sklearn             │    │ sklearn             │
│ LogisticRegression  │    │ LinearSVC           │
│                     │    │                     │
│ Output: Probabilities│    │ Output: Probabilities│
└─────────────────────┘    └─────────────────────┘

┌─────────────────────┐
│ Naive CNN           │
│ ─────────────────── │
│ Input: Frames       │
│ (N, 3, 8, 256, 256)│
│                     │
│ Conv2D blocks       │
│ → GlobalAvgPool     │
│ → FC layers         │
│                     │
│ Output: Logits (2)  │
└─────────────────────┘

FRAME→TEMPORAL MODELS
┌─────────────────────┐    ┌─────────────────────┐
│ ViT-B/16 + GRU      │    │ ViT-B/16 +          │
│ ─────────────────── │    │ Transformer        │
│                     │    │ ─────────────────── │
│ Frames → ViT        │    │ Frames → ViT        │
│ → [CLS] tokens      │    │ → [CLS] tokens      │
│ (768-dim each)      │    │ (768-dim each)      │
│                     │    │                     │
│ → GRU (256 hidden)  │    │ → Transformer       │
│ (2 layers)          │    │ Encoder (2 layers)  │
│                     │    │                     │
│ → Linear(256→1)     │    │ → Mean Pool         │
│                     │    │ → Linear(768→1)     │
│ Output: Logits (1)  │    │                     │
│                     │    │ Output: Logits (1)  │
└─────────────────────┘    └─────────────────────┘

SPATIOTEMPORAL MODELS
┌─────────────────────┐    ┌─────────────────────┐
│ SlowFast            │    │ X3D                 │
│ ─────────────────── │    │ ─────────────────── │
│                     │    │                     │
│ Slow Pathway:       │    │ Efficient 3D CNN    │
│ 16 frames @ 2fps    │    │ (X3D-M variant)     │
│                     │    │                     │
│ Fast Pathway:       │    │ Pretrained on       │
│ 64 frames @ 8fps    │    │ Kinetics-400        │
│                     │    │                     │
│ → Fusion            │    │ → Binary Head       │
│ → Binary Head       │    │                     │
│                     │    │ Output: Logits (1)  │
│ Output: Logits (1)  │    │                     │
└─────────────────────┘    └─────────────────────┘
```

## Memory Optimization Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│              MEMORY OPTIMIZATION HIERARCHY                      │
└─────────────────────────────────────────────────────────────────┘

LEVEL 1: Video Loading (50x reduction)
┌─────────────────────────────────────────────────────────────┐
│ BEFORE: Load entire video                                    │
│  1920×1080, 30fps, 10s = 300 frames                         │
│  Memory: ~1.87 GB per video                                 │
└─────────────────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ AFTER: Frame-by-frame decoding                               │
│  Seek to specific frames → decode only 6 frames             │
│  Memory: ~37 MB per video (50x reduction)                   │
└─────────────────────────────────────────────────────────────┘

LEVEL 2: Resolution Reduction (4x reduction)
┌─────────────────────────────────────────────────────────────┐
│ BEFORE: 224×224 pixels                                      │
│  Memory per frame: 224² × 3 bytes = ~150 KB                 │
└─────────────────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ AFTER: 112×112 pixels                                        │
│  Memory per frame: 112² × 3 bytes = ~38 KB (4x reduction)  │
└─────────────────────────────────────────────────────────────┘

LEVEL 3: Batch Size Reduction (4-32x reduction)
┌─────────────────────────────────────────────────────────────┐
│ BEFORE: batch_size=32                                        │
│  Memory per batch: ~2-3 GB                                   │
└─────────────────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ AFTER: batch_size=1-8 (model-dependent)                      │
│  Memory per batch: ~0.1-0.75 GB (4-32x reduction)           │
│  Compensation: Gradient accumulation (8-16 steps)            │
└─────────────────────────────────────────────────────────────┘

LEVEL 4: Frame Count Reduction (2.7x reduction)
┌─────────────────────────────────────────────────────────────┐
│ BEFORE: 16 frames per video                                  │
│  Memory per sample: ~2.4 MB                                 │
└─────────────────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ AFTER: 6 frames per video                                    │
│  Memory per sample: ~0.9 MB (2.7x reduction)                │
└─────────────────────────────────────────────────────────────┘

LEVEL 5: Incremental Writing (constant memory)
┌─────────────────────────────────────────────────────────────┐
│ BEFORE: Accumulate metadata in list                          │
│  Memory grows unbounded with dataset size                    │
└─────────────────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ AFTER: Write each row immediately to CSV/Arrow              │
│  Memory: Constant regardless of dataset size                │
└─────────────────────────────────────────────────────────────┘

TOTAL MEMORY REDUCTION: ~8-16x overall, 50x for video loading
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-06

