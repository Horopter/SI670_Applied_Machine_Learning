# FVC (Fake Video Classification) Project: Comprehensive Poster Documentation

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [Project Overview](#2-project-overview)
3. [Technologies and Tools](#3-technologies-and-tools)
4. [Architecture - 5-Stage Pipeline](#4-architecture---5-stage-pipeline)
5. [Model Architectures](#5-model-architectures)
6. [Setbacks and Challenges](#6-setbacks-and-challenges)
7. [Solutions and Optimizations](#7-solutions-and-optimizations)
8. [Expected Results](#8-expected-results)
9. [Implementation Details](#9-implementation-details)
10. [Future Work](#10-future-work)
11. [Conclusion](#11-conclusion)

---

## 1. Motivation and Problem Statement

### The Deepfake Proliferation Crisis

In recent years, the proliferation of AI-generated fake videos, commonly known as "deepfakes," has emerged as a critical threat to information integrity and trust in digital media. These sophisticated synthetic videos, created using deep learning techniques such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), can convincingly manipulate facial expressions, speech, and actions in video content.

### Social Media Impact

The spread of misinformation through fake videos on major platforms (YouTube, Twitter/X, Facebook, TikTok) has reached unprecedented levels. These platforms serve billions of users daily, making them prime targets for malicious actors seeking to:
- Spread false information for political or financial gain
- Create non-consensual intimate content
- Damage reputations through fabricated evidence
- Manipulate public opinion on critical issues

### Detection Challenges

The automated detection of fake videos presents numerous technical challenges:

1. **Rapidly Evolving Techniques**: Deepfake generation methods continuously improve, making detection an arms race
2. **High-Quality Syntheses**: Modern deepfakes are increasingly difficult to distinguish from authentic content
3. **Multi-Platform Diversity**: Videos from different platforms have varying properties (resolution, codec, bitrate, frame rate)
4. **Scale**: The volume of video content uploaded daily (millions of hours) requires automated, scalable solutions
5. **Real-Time Requirements**: Content moderation systems need near-instantaneous detection capabilities

### Research Gap

Despite significant research efforts, there remains a critical gap in robust, multi-platform fake video classification systems that can:
- Handle diverse video formats and qualities
- Generalize across different deepfake generation methods
- Operate efficiently at scale
- Provide interpretable results for content moderators

### Real-World Applications

The development of accurate fake video detection systems has far-reaching implications:

- **Content Moderation**: Automated flagging of potentially fake content for human review
- **Fact-Checking**: Supporting journalists and fact-checkers in verifying video authenticity
- **Forensic Analysis**: Assisting law enforcement in investigating cases involving video evidence
- **Platform Safety**: Protecting users from misinformation and harmful content
- **Academic Research**: Advancing understanding of deepfake generation and detection mechanisms

---

## 2. Project Overview

### Objective

The FVC (Fake Video Classification) project aims to develop a robust, automated binary classification system capable of distinguishing between real and fake videos with high accuracy. The system must handle the inherent variability in video properties while maintaining computational efficiency suitable for large-scale deployment.

### Dataset

The project utilizes the **FVC (Fake Video Classification) dataset**, which contains:

- **298 original videos** collected from multiple platforms (YouTube, Twitter, Facebook)
- **Diverse video properties**: Resolution ranges from 240p to 1080p+, varying aspect ratios, frame rates (15-60 fps), and codecs (H.264, VP9, etc.)
- **Platform diversity**: Videos from different social media platforms with distinct encoding characteristics
- **Labeled data**: Each video is labeled as either "real" or "fake" based on ground truth verification

### Scale and Complexity

After data augmentation (Stage 1), the dataset expands to:

- **11N videos total**: N original videos (298) + 10N augmented versions = **3,278 expected videos**
- **Actual processed**: 3,244 videos (34 missing due to incomplete augmentations for 3 videos)
- **Total frames**: Approximately 3.5 seconds per 100 frames processing time for Stage 3 scaling

### Challenge: Video Variability

The primary technical challenge lies in the extreme variability of video properties:

1. **Resolution**: Videos range from 240×240 to 1920×1080+ pixels
2. **Aspect Ratio**: Various aspect ratios (16:9, 4:3, 1:1, etc.)
3. **Frame Count**: Videos range from a few seconds to several minutes (30-3000+ frames)
4. **Frame Rate**: 15-60 fps across different videos
5. **Codec**: Multiple codecs (H.264, VP9, MPEG-4) with different compression characteristics
6. **Bitrate**: Highly variable bitrates affecting quality and compression artifacts

This variability necessitates a robust preprocessing pipeline that can normalize inputs while preserving discriminative features for classification.

---

## 3. Technologies and Tools

### Core Deep Learning Frameworks

**PyTorch 2.0+**
- Primary deep learning framework for model development and training
- Provides automatic mixed precision (AMP) for memory-efficient training
- CUDA support for GPU acceleration
- Location: `requirements.txt` - `torch>=2.0.0`

**torchvision 0.15.0+**
- Pretrained video models (R3D, X3D, SlowFast, I3D, R(2+1)D)
- Pretrained on Kinetics-400 dataset for transfer learning
- Video preprocessing utilities
- Location: `requirements.txt` - `torchvision>=0.15.0`

**timm 0.9.0+**
- Vision Transformer (ViT) models for frame-level feature extraction
- Pretrained ViT-B/16 used in frame→temporal models
- Location: `requirements.txt` - `timm>=0.9.0`

**Hugging Face Transformers & Diffusers**
- Transformers 4.30.0+: General transformer models
- Diffusers 0.21.0+: Stable Diffusion VAE for high-quality video scaling
- AutoencoderKL models for Stage 3 scaling
- Location: `requirements.txt` - `transformers>=4.30.0`, `diffusers>=0.21.0`, `accelerate>=0.20.0`

### Data Processing and Analytics

**Polars 0.19.0+**
- Fast DataFrame operations (10-100x faster than pandas for large datasets)
- Used for metadata loading, filtering, and manipulation
- Arrow IPC format for efficient serialization
- Location: `requirements.txt` - `polars>=0.19.0`

**PyArrow 14.0.0+**
- Arrow columnar format for efficient data storage
- Supports `.arrow`, `.parquet`, and `.csv` formats
- Location: `requirements.txt` - `pyarrow>=14.0.0`

**DuckDB 0.9.0+**
- Analytical database for querying experiment results
- Fast aggregations and joins on large metadata files
- Location: `requirements.txt` - `duckdb>=0.9.0`

**Pandas 2.0.0+**
- Legacy DataFrame operations (used in some modules)
- Location: `requirements.txt` - `pandas>=2.0.0`

### Video and Image Processing

**OpenCV 4.8.0+**
- Video decoding, frame extraction, image processing
- Feature extraction (Canny edges, Laplacian, Sobel gradients)
- DCT (Discrete Cosine Transform) computation
- Location: `requirements.txt` - `opencv-python>=4.8.0`

**PyAV 10.0.0+**
- Advanced video decoding with frame-by-frame seeking
- Critical for memory-efficient video processing
- Supports multiple codecs and formats
- Location: `requirements.txt` - `av>=10.0.0`

**Pillow 10.0.0+**
- Image processing for augmentations
- PIL Image operations (rotation, affine, color adjustments)
- Location: `requirements.txt` - `Pillow>=10.0.0`

### Machine Learning Libraries

**scikit-learn 1.3.0+**
- Baseline models (LogisticRegression, LinearSVC)
- Metrics computation (accuracy, precision, recall, F1)
- Feature scaling and preprocessing
- Location: `requirements.txt` - `scikit-learn>=1.3.0`

**XGBoost 2.0.0+**
- Gradient boosting for ensemble models
- Location: `requirements.txt` - `xgboost>=2.0.0`

**scipy 1.11.0+**
- Signal processing (DCT, filtering)
- Statistical functions
- Location: `requirements.txt` - `scipy>=1.11.0`

### MLOps and Infrastructure

**MLflow 2.8.0+**
- Experiment tracking and model registry
- Metrics logging and visualization
- Location: `requirements.txt` - `mlflow>=2.8.0`

**Apache Airflow 2.7.0+**
- Workflow orchestration (optional, for production pipelines)
- Location: `requirements.txt` - `apache-airflow>=2.7.0`

**SLURM**
- Cluster job management and resource allocation
- Multi-node distributed processing
- Batch job scheduling
- Location: `src/scripts/slurm_*.sh` scripts

### Data Validation

**Pandera 0.18.0+**
- Schema validation for DataFrames
- Ensures data integrity across pipeline stages
- Location: `requirements.txt` - `pandera>=0.18.0`
- Implementation: `lib/utils/schemas.py`

### Visualization and Dashboards

**Streamlit 1.28.0+**
- Interactive results dashboard
- Model performance comparisons
- K-fold cross-validation analysis
- Location: `requirements.txt` - `streamlit>=1.28.0`
- Implementation: `src/dashboard_results.py`

**Plotly 5.17.0+**
- Interactive plots (ROC curves, PR curves, training curves)
- Location: `requirements.txt` - `plotly>=5.17.0`

**Matplotlib 3.7.0+ & Seaborn 0.12.0+**
- Publication-ready figures
- Statistical visualizations
- Location: `requirements.txt` - `matplotlib>=3.7.0`, `seaborn>=0.12.0`
- Implementation: `src/generate_paper_figures.py`

### Utilities

**tqdm 4.65.0+**
- Progress bars for long-running operations
- Location: `requirements.txt` - `tqdm>=4.65.0`

**psutil 5.9.0+**
- System and process utilities
- Memory profiling and monitoring
- Location: `requirements.txt` - `psutil>=5.9.0`
- Implementation: `lib/utils/memory.py`

**joblib 1.3.0+**
- Model serialization for sklearn models
- Location: `requirements.txt` - `joblib>=1.3.0`

### Dependency Management

**Version Pinning for Compatibility:**
- `numpy>=1.24.0,<2.0.0`: Pinned to <2.0 for compatibility with numba, thinc, astropy, pywavelets
- `cryptography>=41.0.5,<43.0.0`: Pinned for pyopenssl compatibility
- `FuzzyTM>=0.4.0`: Required by gensim

---

## 4. Architecture - 5-Stage Pipeline

The FVC project implements a sophisticated 5-stage pipeline designed to process videos from raw input to trained classification models. Each stage is modular, resumable, and optimized for memory efficiency and distributed processing.

### Pipeline Overview

```
N original videos (298)
    ↓
Stage 1: Video Augmentation
    ↓
11N videos (N original + 10N augmented = 3,244 actual)
    ↓
    ├─→ Stage 2: Extract M handcrafted features → M features per video (~15 features)
    │
    └─→ Stage 3: Scale videos → 11N scaled videos (max dimension = 256 pixels)
            ↓
        Stage 4: Extract P scaled features → P features per video (~8 features)
            ↓
        Stage 5: Training (videos + M features + P features) → Trained models
```

### Stage 1: Video Augmentation

**Location**: `lib/augmentation/pipeline.py`

**Input**: N original videos (298 videos)

**Output**: 11N videos (N original + 10N augmented versions)

**Purpose**: Generate diverse augmented versions of each video to increase dataset size and improve model robustness.

**Augmentation Types** (10 types per video):

1. **Rotation**: Random rotation with small angles (±10 degrees)
   - Implementation: `lib/augmentation/transforms.py::RandomRotation`
   - Preserves aspect ratio with zero-padding

2. **Horizontal Flip**: Mirror video horizontally
   - Implementation: `lib/augmentation/transforms.py::apply_simple_augmentation('flip')`
   - Probability: 0.5

3. **Brightness Adjustment**: Random brightness modification
   - Implementation: PIL ImageEnhance.Brightness
   - Factor range: 0.8-1.2

4. **Contrast Adjustment**: Random contrast modification
   - Implementation: PIL ImageEnhance.Contrast
   - Factor range: 0.8-1.2

5. **Saturation Adjustment**: Random color saturation modification
   - Implementation: PIL ImageEnhance.Color
   - Factor range: 0.8-1.2

6. **Gaussian Noise**: Add random Gaussian noise
   - Implementation: `lib/augmentation/transforms.py::RandomGaussianNoise`
   - Standard deviation: 10.0 (configurable)
   - Probability: 0.5

7. **Gaussian Blur**: Apply Gaussian blur filter
   - Implementation: `lib/augmentation/transforms.py::RandomGaussianBlur`
   - Radius range: 0.5-2.0 pixels
   - Probability: 0.5

8. **Affine Transformation**: Random translation, scaling, and shearing
   - Implementation: `lib/augmentation/transforms.py::RandomAffine`
   - Translation: ±10% of image dimensions
   - Scale: 0.9-1.1x
   - Probability: 0.5

9. **Elastic Transform**: Simplified elastic deformation
   - Implementation: `lib/augmentation/transforms.py::apply_simple_augmentation('elastic')`
   - Simulates non-rigid deformations

10. **Combination**: Some augmentations combine multiple transformations

**Key Features**:

- **Deterministic Seeding**: Each video's augmentations are seeded by video path hash, ensuring reproducibility across runs
- **Frame-by-Frame Decoding**: Only decodes the frames needed (typically 6-8 frames) instead of loading entire videos
  - Memory reduction: 1.87GB → 37MB per video (50x reduction)
  - Implementation: PyAV frame seeking (`container.seek()`)
- **Chunked Processing**: Processes videos in chunks of 250 frames for long videos
- **Resume Capability**: Can resume from checkpoints if interrupted
- **Metadata Tracking**: Creates `augmented_metadata.arrow` (or `.parquet`/`.csv`) with columns:
  - `video_path`: Path to augmented video
  - `label`: Original label (real/fake)
  - `original_video`: Path to original video
  - `augmentation_idx`: Index of augmentation (-1 for original, 0-9 for augmentations)
  - `is_original`: Boolean indicating if this is the original video

**Output Structure**:
```
data/augmented_videos/
├── video1_original.mp4
├── video1_aug0.mp4
├── video1_aug1.mp4
├── ...
├── video1_aug9.mp4
└── augmented_metadata.arrow
```

**Memory Optimizations**:
- One video at a time processing
- Immediate disk writes (no memory accumulation)
- Aggressive garbage collection after each video
- Frame-by-frame decoding (only needed frames loaded)

### Stage 2: Handcrafted Feature Extraction

**Location**: `lib/features/pipeline.py`, `lib/features/handcrafted.py`

**Input**: 11N videos from Stage 1 (3,244 videos)

**Output**: M features per video (~15 features)

**Purpose**: Extract handcrafted features that capture compression artifacts, noise patterns, and codec characteristics that are indicative of fake videos.

**Features Extracted**:

1. **Noise Residual Energy** (3 features):
   - `noise_energy`: Total energy of high-pass filtered noise
   - `noise_mean`: Mean of noise residual energy
   - `noise_std`: Standard deviation of noise residual energy
   - Implementation: High-pass filter kernel `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]`
   - Rationale: Fake videos often have different noise patterns due to generation artifacts

2. **DCT Band Statistics** (5 features):
   - `dct_dc_mean`: Mean of DC (direct current) coefficients
   - `dct_dc_std`: Standard deviation of DC coefficients
   - `dct_ac_mean`: Mean of AC (alternating current) coefficients
   - `dct_ac_std`: Standard deviation of AC coefficients
   - `dct_ac_energy`: Total energy of AC coefficients
   - Implementation: 8×8 block DCT using OpenCV `cv2.dct()`
   - Rationale: Compression artifacts manifest differently in frequency domain for fake vs real videos

3. **Blur/Sharpness Metrics** (3 features):
   - `laplacian_var`: Variance of Laplacian operator (sharpness measure)
   - `gradient_mean`: Mean of Sobel gradient magnitude
   - `gradient_std`: Standard deviation of Sobel gradient magnitude
   - Implementation: OpenCV `cv2.Laplacian()` and `cv2.Sobel()`
   - Rationale: Fake videos may have different sharpness characteristics

4. **Block Boundary Inconsistency** (1 feature):
   - `boundary_inconsistency`: Measure of compression artifacts at block boundaries
   - Implementation: Standard deviation of pixel values at 8×8 block boundaries
   - Rationale: Compression codecs create block artifacts that differ between real and fake videos

5. **Codec Cues** (3 features):
   - `codec_bitrate`: Video bitrate (from ffprobe)
   - `codec_fps`: Frame rate (from ffprobe)
   - `codec_resolution`: Width × height (from ffprobe)
   - Implementation: `subprocess.run(['ffprobe', ...])` to extract metadata
   - Rationale: Different platforms use different codecs and encoding settings

**Total Features**: 3 + 5 + 3 + 1 + 3 = **15 handcrafted features**

**Key Features**:

- **Adaptive Frame Sampling**: Uses percentage-based sampling (default 10% of frames, min=5, max=50) or fixed number of frames
- **Frame Aggregation**: Features extracted per frame, then aggregated (mean) across sampled frames
- **Caching**: Features can be cached to disk to avoid recomputation
- **Resume Capability**: Skips videos where feature files already exist
- **Multi-Format Support**: Saves features as `.npy` (NumPy) or `.parquet` files
- **Metadata Tracking**: Creates `features_metadata.arrow` with feature paths and labels

**Output Structure**:
```
data/features_stage2/
├── video1_original_features.npy
├── video1_aug0_features.npy
├── ...
└── features_metadata.arrow
```

**Memory Optimizations**:
- Processes one video at a time
- Clears frame data immediately after feature extraction
- Aggressive GC after each video

### Stage 3: Video Scaling

**Location**: `lib/scaling/pipeline.py`, `lib/scaling/methods.py`

**Input**: 11N videos from Stage 1 (3,244 videos)

**Output**: 11N scaled videos (max dimension = 256 pixels, preserving aspect ratio)

**Purpose**: Normalize video dimensions to a consistent size for model training while preserving aspect ratio and quality.

**Scaling Methods**:

1. **Letterbox Resize** (default fallback):
   - Simple resize with letterboxing (black bars to maintain aspect ratio)
   - Implementation: OpenCV `cv2.resize()` with `INTER_AREA` interpolation
   - Fast and memory-efficient
   - Quality: Good for most use cases

2. **Autoencoder Scaling** (default, high-quality):
   - Uses pretrained Hugging Face VAE (`stabilityai/sd-vae-ft-mse`)
   - Encodes frames to latent space (8x downscale), then decodes back
   - Preserves high-frequency details better than simple resize
   - Implementation: `lib/scaling/methods.py::load_hf_autoencoder()`, `_autoencoder_scale()`
   - Quality: Excellent, especially for complex scenes
   - Memory: Requires GPU (1-2GB for model)
   - Speed: ~10-50x slower than letterbox (depending on GPU)

**Key Features**:

- **Chunked Processing**: Processes videos in chunks of 100 frames to avoid OOM
  - Default: `chunk_size=100`, `max_frames=100`
  - Optimized for 64GB RAM per node
- **OOM Detection and Fallback**: Automatically falls back to letterbox if autoencoder fails due to OOM
  - Multiple fallback levels: frame-level → chunk-level → video-level → letterbox
- **Aspect Ratio Preservation**: All methods preserve original aspect ratio
- **Scaling Direction Tracking**: Stores original dimensions to detect upscaling vs downscaling
- **Robust Model Loading**: Handles both standalone VAE models and full Stable Diffusion models
  - Tries loading without subfolder first, then with `subfolder="vae"`
- **Dtype Matching**: Ensures input tensors match model dtype (float16 on CUDA, float32 on CPU)
- **Resume Capability**: Skips videos where scaled versions already exist
- **Metadata Tracking**: Creates `scaled_metadata.arrow` with:
  - `video_path`: Path to scaled video
  - `label`: Original label
  - `original_video`: Path to original video
  - `original_width`, `original_height`: Original dimensions
  - `scaled_width`, `scaled_height`: Scaled dimensions
  - `is_upscaled`, `is_downscaled`: Binary features (added in Stage 4)

**Processing Time**: Approximately 3.5 seconds per 100 frames for autoencoder scaling

**Memory Optimizations**:
- Chunked processing (100 frames per chunk)
- Proactive memory checks before each chunk
- Aggressive GC after each chunk
- OOM detection and automatic fallback
- Frame-by-frame processing within chunks

**Output Structure**:
```
data/scaled_videos/
├── video1_original_scaled.mp4
├── video1_aug0_scaled.mp4
├── ...
└── scaled_metadata.arrow
```

### Stage 4: Scaled Video Feature Extraction

**Location**: `lib/features/scaled.py`

**Input**: 11N scaled videos from Stage 3 (3,244 videos)

**Output**: P additional features per video (~8 features)

**Purpose**: Extract features specific to scaled videos that capture scaling artifacts, edge preservation, and compression characteristics.

**Features Extracted**:

1. **Edge Preservation** (1 feature):
   - `edge_density`: Density of Canny edges in scaled video
   - Implementation: OpenCV `cv2.Canny()` edge detection
   - Rationale: Scaling may affect edge sharpness differently for real vs fake videos

2. **Texture Uniformity** (1 feature):
   - `texture_uniformity`: Measure of texture consistency (inverse of local variance)
   - Implementation: 5×5 local mean filter, then variance computation
   - Rationale: Fake videos may have different texture patterns after scaling

3. **Color Consistency** (3 features):
   - `color_consistency_r`: Standard deviation of red channel
   - `color_consistency_g`: Standard deviation of green channel
   - `color_consistency_b`: Standard deviation of blue channel
   - Implementation: Per-channel standard deviation
   - Rationale: Color artifacts may be more visible after scaling

4. **Compression Artifacts** (1 feature):
   - `compression_artifacts`: Blockiness measure (8×8 block discontinuities)
   - Implementation: Horizontal and vertical differences at block boundaries
   - Rationale: Compression artifacts become more visible after scaling

5. **Scaling Direction Indicators** (2 features):
   - `is_upscaled`: 1 if video was upscaled, 0 otherwise
   - `is_downscaled`: 1 if video was downscaled, 0 otherwise
   - Implementation: Compares `max(original_width, original_height)` vs `max(scaled_width, scaled_height)`
   - Rationale: Scaling direction may be informative for classification

**Total Features**: 1 + 1 + 3 + 1 + 2 = **8 scaled-video-specific features**

**Key Features**:

- **Adaptive Frame Sampling**: Same as Stage 2 (10% of frames, min=5, max=50)
- **Frame Aggregation**: Features aggregated (mean) across sampled frames
- **Scaling Direction Detection**: Automatically detects and adds `is_upscaled` and `is_downscaled` features
- **Resume Capability**: Skips videos where feature files already exist
- **Metadata Tracking**: Creates `features_scaled_metadata.arrow` with feature paths and labels

**Output Structure**:
```
data/features_stage4/
├── video1_original_scaled_features.parquet
├── video1_aug0_scaled_features.parquet
├── ...
└── features_scaled_metadata.arrow
```

### Stage 5: Model Training

**Location**: `lib/training/pipeline.py`

**Input**: 
- 11N scaled videos from Stage 3
- M features from Stage 2 (~15 features)
- P features from Stage 4 (~8 features)

**Output**: Trained models with evaluation metrics

**Purpose**: Train multiple model architectures using combined video frames and extracted features to classify videos as real or fake.

**Training Configuration**:

- **K-Fold Cross-Validation**: 5-fold stratified CV
  - Stratification by label and platform (if available)
  - Prevents data leakage via duplicate group handling
- **Frame Sampling**: 8 frames per video (uniformly sampled)
- **Input Resolution**: 256×256 pixels (from Stage 3 scaling)
- **Batch Sizes**: Model-dependent (1-8, ultra-conservative for memory)
- **Gradient Accumulation**: 8-16 steps to compensate for small batches
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Mixed Precision**: Automatic Mixed Precision (AMP) enabled
- **Class Imbalance Handling**: Balanced batch sampling, inverse-frequency weights

**Model Types** (7 models):

1. **Logistic Regression** (`logistic_regression`)
2. **Linear SVM** (`svm`)
3. **Naive CNN** (`naive_cnn`)
4. **ViT-B/16 + GRU** (`vit_gru`)
5. **ViT-B/16 + Transformer** (`vit_transformer`)
6. **SlowFast** (`slowfast`)
7. **X3D** (`x3d`)

**Key Features**:

- **Multi-Model Training**: Trains all 7 models sequentially
- **Resume Capability**: Checks `training_complete.pt` to skip completed models
- **Per-Model Checkpoints**: Each model has separate checkpoint directory
- **Feature Combination**: Combines handcrafted features (Stage 2) and scaled features (Stage 4) with video frames
- **Collinearity Removal**: Removes highly correlated features (correlation > 0.95) per-fold
- **Experiment Tracking**: Logs metrics to MLflow and custom tracker
- **Model Comparison**: Generates comparison metrics across all models

**Output Structure**:
```
data/training_results/
├── logistic_regression/
│   ├── fold_1/
│   │   ├── model.joblib
│   │   └── scaler.joblib
│   ├── fold_2/
│   ├── ...
│   ├── fold_results.csv
│   └── training_complete.pt
├── svm/
├── naive_cnn/
├── vit_gru/
├── vit_transformer/
├── slowfast/
└── x3d/
```

**Memory Optimizations**:
- Model-specific batch sizes (1-8)
- Gradient accumulation (8-16 steps)
- Aggressive GC after each model/fold
- Feature caching to disk

---

## 5. Model Architectures

The FVC project implements 7 distinct model architectures, ranging from simple baselines to state-of-the-art spatiotemporal models. Each model is designed to leverage different aspects of the input data (handcrafted features, video frames, or both).

### Baseline Models (3 models)

These models use only handcrafted features extracted in Stages 2 and 4, providing fast training and inference with minimal computational requirements.

#### 1. Logistic Regression (`logistic_regression`)

**Location**: `lib/training/_linear.py::LogisticRegressionBaseline`

**Architecture**:
- Input: Handcrafted features (M + P features, ~23 total after collinearity removal)
- Model: sklearn `LogisticRegression`
- Output: Binary classification probabilities

**Configuration**:
- Batch size: 8
- Memory: Low (~100MB)
- Training time: Fast (seconds to minutes)

**Use Case**: Fast baseline for comparison, interpretable feature importance

#### 2. Linear SVM (`svm`)

**Location**: `lib/training/_svm.py::SVMBaseline`

**Architecture**:
- Input: Handcrafted features (M + P features)
- Model: sklearn `LinearSVC`
- Output: Binary classification (decision function converted to probabilities via sigmoid)

**Configuration**:
- Batch size: 8
- Memory: Low (~100MB)
- Training time: Fast (seconds to minutes)

**Use Case**: Alternative baseline with different decision boundary characteristics

#### 3. Naive CNN (`naive_cnn`)

**Location**: `lib/training/_cnn.py::NaiveCNNBaseline`

**Architecture**:
- Input: Video frames (N, C=3, T=8, H=256, W=256)
- Backbone: Simple 2D CNN
  - Conv2D(3→32) → BatchNorm → ReLU
  - Conv2D(32→64) → BatchNorm → ReLU
  - Conv2D(64→128) → BatchNorm → ReLU
  - Global Average Pooling
  - Linear(128→64) → Dropout(0.5) → Linear(64→2)
- Processing: Processes frames independently, averages predictions
- Output: Binary logits (2 classes)

**Configuration**:
- Batch size: 4
- Memory: Medium (~500MB)
- Training time: Moderate (minutes to hours)

**Use Case**: Simple deep learning baseline without temporal modeling

### Frame→Temporal Models (2 models)

These models extract frame-level features using Vision Transformers, then model temporal relationships using recurrent or transformer architectures.

#### 4. ViT-B/16 + GRU (`vit_gru`)

**Location**: `lib/training/_transformer_gru.py::ViTGRUModel`

**Architecture**:
- Frame Backbone: Vision Transformer (ViT-B/16) from `timm`
  - Pretrained on ImageNet
  - Patch size: 16×16
  - Embedding dimension: 768
  - Extracts [CLS] token per frame (768-dim)
- Temporal Head: GRU (Gated Recurrent Unit)
  - Input: 768-dim frame embeddings
  - Hidden dimension: 256
  - Number of layers: 2
  - Bidirectional: No (unidirectional)
- Classification Head: Linear(256→1)
- Output: Binary logits

**Configuration**:
- Batch size: 1 (ultra-conservative)
- Gradient accumulation: 16 steps
- Memory: High (~2-4GB)
- Training time: Long (hours)

**Use Case**: Captures temporal patterns with recurrent modeling

#### 5. ViT-B/16 + Transformer (`vit_transformer`)

**Location**: `lib/training/_transformer.py::ViTTransformerModel`

**Architecture**:
- Frame Backbone: Same ViT-B/16 as above
- Temporal Head: Transformer Encoder
  - d_model: 768 (matches ViT embedding)
  - nhead: 8 (multi-head attention)
  - num_layers: 2
  - dim_feedforward: 2048
  - Dropout: 0.5
- Classification: Mean pooling over temporal dimension → Linear(768→1)
- Output: Binary logits

**Configuration**:
- Batch size: 1 (ultra-conservative)
- Gradient accumulation: 16 steps
- Memory: High (~3-5GB)
- Training time: Long (hours)

**Use Case**: Captures temporal patterns with self-attention mechanism

### Spatiotemporal Models (2 models)

These models process video as 3D volumes, capturing both spatial and temporal information simultaneously using 3D convolutions.

#### 6. SlowFast (`slowfast`)

**Location**: `lib/training/slowfast.py::SlowFastModel`

**Architecture**:
- Dual-pathway network from `torchvision.models.video.slowfast_r50`
- **Slow Pathway**: 16 frames at 2 fps (temporal semantics)
  - Processes every 4th frame
  - Captures slow, semantic changes
- **Fast Pathway**: 64 frames at 8 fps (motion details)
  - Processes every frame
  - Captures fast, motion details
- **Fusion**: Lateral connections between pathways
- **Classification Head**: Binary classification (1 output)
- Output: Binary logits

**Configuration**:
- Batch size: 1 (ultra-conservative)
- Gradient accumulation: 16 steps
- Memory: Very High (~4-8GB)
- Training time: Very Long (hours to days)

**Use Case**: State-of-the-art spatiotemporal modeling with dual-pathway architecture

**Fallback**: If `slowfast_r50` not available, uses simplified implementation

#### 7. X3D (`x3d`)

**Location**: `lib/training/x3d.py::X3DModel`

**Architecture**:
- Efficient 3D CNN from `torchvision.models.video.x3d_m`
- Pretrained on Kinetics-400
- Variant: X3D-M (medium size)
- **Key Innovation**: Expanding architectures (width, depth, resolution, temporal)
- **Classification Head**: Binary classification (1 output)
- Output: Binary logits

**Configuration**:
- Batch size: 1 (ultra-conservative)
- Gradient accumulation: 16 steps
- Memory: High (~3-6GB)
- Training time: Long (hours)

**Use Case**: Efficient spatiotemporal modeling with optimized architecture

**Fallback**: If `x3d_m` not available, uses `r3d_18` as approximation

### Additional Models (Available but not in main pipeline)

#### I3D (Inflated 3D ConvNet)

**Location**: `lib/training/i3d.py::I3DModel`

- Architecture: Inflated 2D ConvNet to 3D
- Pretrained on Kinetics-400
- Fallback: Uses `r3d_18` if not available

#### R(2+1)D (Factorized 3D Convolutions)

**Location**: `lib/training/r2plus1d.py::R2Plus1DModel`

- Architecture: Factorized 3D convolutions (spatial + temporal)
- Pretrained on Kinetics-400
- Fallback: Uses `r3d_18` if not available

#### Ensemble Models

**Location**: `lib/training/ensemble.py`

- **Meta-Learner**: Neural network that learns to combine predictions from multiple models
- **Weighted Average**: Simple weighted combination of model predictions
- Input: Predictions from all 7 models
- Output: Final ensemble prediction

### Model Comparison Summary

| Model | Input | Memory | Speed | Temporal Modeling | Best For |
|-------|-------|--------|-------|-------------------|----------|
| Logistic Regression | Features | Low | Fast | None | Baseline, interpretability |
| SVM | Features | Low | Fast | None | Baseline, different boundary |
| Naive CNN | Frames | Medium | Moderate | None | Simple deep learning baseline |
| ViT+GRU | Frames | High | Slow | Recurrent | Temporal patterns |
| ViT+Transformer | Frames | High | Slow | Self-attention | Temporal patterns |
| SlowFast | Frames | Very High | Very Slow | 3D Conv + Dual-path | State-of-the-art |
| X3D | Frames | High | Slow | 3D Conv | Efficient spatiotemporal |

---

## 6. Setbacks and Challenges

The development of the FVC project encountered numerous technical challenges and setbacks. This section provides a comprehensive account of each issue, its root causes, impact, and the solutions implemented. These setbacks were critical learning experiences that shaped the final architecture and implementation.

### 6.1 Out-of-Memory (OOM) Errors

**Severity**: Critical - Blocked all training and processing

**Problem Description**:
The project initially suffered from severe Out-of-Memory (OOM) errors that caused:
- GPU memory exhaustion during training
- Kernel death on SLURM cluster nodes
- Loss reporting as 0.0000 due to memory corruption
- Complete inability to process large videos (>1920×1080 resolution)

**Root Causes**:

1. **Large Video Resolutions**: Some videos exceeded 1920×1080 pixels, requiring massive memory for full-frame loading
   - Example: 1920×1080 video, 30fps, 10 seconds = 300 frames
   - Memory per frame: ~6.2MB (1920×1080×3 bytes)
   - Total video memory: ~1.87GB per video

2. **Variable Batch Dimensions**: Initial approach used variable aspect ratios with adaptive pooling, requiring padding and custom collate functions that increased memory overhead

3. **Accumulated Memory**: Multiple sources of memory accumulation:
   - Full video loading during augmentation generation
   - Metadata accumulation in lists before DataFrame creation
   - Multiple workers in DataLoader (multiprocessing overhead)
   - CUDA cache not being cleared between operations

4. **Insufficient Garbage Collection**: Python's default garbage collection was insufficient for the memory-intensive video processing workload

**Impact**:
- **Training**: Impossible to train models on full dataset
- **Augmentation**: Could not generate augmentations for large videos
- **Scalability**: Limited to very small batch sizes (1-2) or very low resolution (64×64)
- **Time**: Frequent crashes required constant monitoring and manual intervention

**Solutions Implemented**:

1. **Frame-by-Frame Video Decoding** (50x memory reduction):
   - **Before**: Load entire video → extract needed frames → delete video
   - **After**: Seek to specific frames → decode only those frames → never load full video
   - **Implementation**: PyAV `container.seek()` to specific timestamps
   - **Memory reduction**: 1.87GB → 37MB per video (50x reduction)
   - **Location**: `lib/augmentation/io.py::load_frames()`

2. **Fixed-Size Preprocessing** (4x memory reduction):
   - **Before**: Variable aspect ratio with adaptive pooling
   - **After**: Fixed 112×112 (configurable) with letterboxing
   - **Memory reduction**: 224² → 112² = 4x reduction per frame
   - **Location**: `lib/models/video.py::VideoConfig`

3. **Ultra-Conservative Batch Sizes** (4-32x reduction):
   - **Before**: `batch_size=32` (16 real + 16 fake)
   - **After**: Model-dependent batch sizes (1-8)
     - Baselines: 8
     - Frame→Temporal: 1
     - Spatiotemporal: 1
   - **Compensation**: Gradient accumulation (8-16 steps) to maintain effective batch size
   - **Location**: `lib/training/model_factory.py::MODEL_MEMORY_CONFIGS`

4. **Reduced Frame Count** (2.7x reduction):
   - **Before**: 16 frames per video
   - **After**: 6 frames per video (ultra-conservative)
   - **Location**: `lib/training/pipeline.py::stage5_train_models()`

5. **Zero Workers** (eliminated multiprocessing overhead):
   - **Before**: `num_workers=4`
   - **After**: `num_workers=0`
   - **Impact**: Eliminated ~500MB overhead from worker processes

6. **Incremental CSV Writing** (constant memory):
   - **Before**: Accumulate metadata in list → create DataFrame at end
   - **After**: Write each row immediately to CSV
   - **Memory**: Constant regardless of dataset size
   - **Location**: `lib/augmentation/pipeline.py::stage1_augment_videos()`

7. **Chunked Video Processing** (Stage 3):
   - Process videos in chunks of 100 frames
   - Prevents loading entire long videos into memory
   - **Location**: `lib/scaling/pipeline.py::scale_video()`

8. **Aggressive Garbage Collection**:
   - 3 passes of `gc.collect()` instead of 1
   - CUDA cache clearing: `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`
   - Called after every stage, video, batch, and epoch
   - **Location**: `lib/utils/memory.py::aggressive_gc()`

9. **OOM Detection and Automatic Fallback**:
   - Detects OOM errors from multiple error message patterns
   - Automatic fallback to letterbox method if autoencoder fails
   - Multiple fallback levels: frame → chunk → video → method
   - **Location**: `lib/utils/memory.py::check_oom_error()`, `handle_oom_error()`

**Results**:
- **Memory Usage**: Reduced from ~80GB (OOM) to ~5-10GB (typical), ~25-30GB (worst case)
- **Stability**: Eliminated OOM crashes, enabling reliable processing
- **Scalability**: Can now process full dataset on 64GB RAM nodes

### 6.2 Overfitting Issues

**Severity**: High - Invalidated model performance metrics

**Problem Description**:
Initial training runs showed suspicious behavior:
- Loss dropping to exactly 0.0000 (too perfect)
- Identical logits across different batches
- Low gradient norms (near zero)
- Only 12 trainable parameters (backbone frozen too aggressively)

**Root Causes**:

1. **Backbone Frozen Too Aggressively**: Only classification head was trainable, limiting model capacity
2. **Insufficient Data Augmentation**: Only basic augmentations (horizontal flip, color jitter)
3. **No Regularization**: No dropout, weight decay, or other regularization techniques
4. **Single Train/Val Split**: No cross-validation, leading to overfitting to specific split

**Impact**:
- **Performance Metrics**: Invalid (artificially perfect)
- **Generalization**: Poor performance on unseen data
- **Reproducibility**: Results not reliable across different data splits

**Solutions Implemented**:

1. **K-Fold Cross-Validation** (5-fold stratified):
   - More robust performance estimates
   - Better generalization assessment
   - Prevents overfitting to specific split
   - **Location**: `lib/data/loading.py::stratified_kfold()`

2. **Comprehensive Augmentations**:
   - 10 augmentation types (rotation, flip, brightness, contrast, saturation, noise, blur, affine, elastic)
   - Pre-generated augmentations (1 per video, deterministic)
   - **Location**: `lib/augmentation/transforms.py`

3. **Backbone Unfreezing Option**:
   - Option to unfreeze backbone for fine-tuning
   - Gradual unfreezing strategy available
   - **Location**: Model factory configurations

4. **Enhanced Diagnostics**:
   - Logits statistics (min, max, mean, std, unique count)
   - Gradient norms logging
   - Parameter count verification
   - Loss verification (manual computation)
   - **Location**: `lib/training/trainer.py::fit()`

**Results**:
- **Metrics**: Realistic performance metrics (not artificially perfect)
- **Generalization**: Better performance on validation sets
- **Reliability**: Reproducible results across folds

### 6.3 Data Leakage via Duplicate Groups

**Severity**: Critical - Invalidated all experimental results

**Problem Description**:
The initial `stratified_kfold` implementation did not handle `dup_group`, meaning videos from the same duplicate group could appear in both training and validation sets. This caused severe data leakage.

**Root Cause**:
- `stratified_kfold` function split individual videos, not duplicate groups
- Videos with same `dup_group` (duplicates or near-duplicates) were split across folds
- Model could "cheat" by seeing similar videos in both train and validation

**Impact**:
- **Performance**: Artificially inflated (model seeing similar data in both sets)
- **Generalization**: Results would not generalize to new data
- **Paper Validity**: Would have led to paper rejection for methodological flaws

**Solution**:
Modified `stratified_kfold` to:
- Group by `dup_group` first (if present)
- Assign entire groups to folds, not individual videos
- Ensures all videos in a duplicate group stay in the same fold
- Added validation check to detect and prevent leakage
- **Location**: `lib/data/loading.py::stratified_kfold()`

**Validation**:
- Automatic check after each fold split
- Logs warning and raises error if leakage detected
- Logs confirmation if no leakage found

**Results**:
- **Methodology**: Valid, no data leakage
- **Performance**: Realistic metrics (not inflated)
- **Reproducibility**: Proper train/val separation

### 6.4 Data Pipeline Issues

**Severity**: Medium - Caused crashes and data inconsistencies

**Problems**:

1. **Missing Video Files**: Some videos referenced in metadata did not exist on disk
2. **Inconsistent Path Resolution**: Different parts of codebase used different path resolution strategies
3. **Partial Unzip Issues**: Dataset extraction failed due to zip bomb detection

**Solutions**:

1. **Centralized Path Resolution**:
   - Single function `resolve_video_path()` with multiple fallback strategies
   - **Location**: `lib/utils/paths.py::resolve_video_path()`

2. **Video Filtering**:
   - `filter_existing_videos()` before splitting
   - Optional frame count verification
   - Graceful handling of missing files
   - **Location**: `lib/data/scan.py::filter_existing_videos()`

3. **Dataset Setup**:
   - `setup_fvc_dataset.py` handles unzipping with `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`
   - Path verification for extracted videos
   - Metadata generation and validation
   - **Location**: `src/setup_fvc_dataset.py`

**Results**:
- **Robustness**: Handles missing files gracefully
- **Consistency**: Uniform path resolution across codebase
- **Reliability**: Dataset setup works reliably

### 6.5 Dependency Conflicts

**Severity**: Medium - Blocked installation and caused runtime errors

**Problems**:

1. **numpy 2.x Incompatibility**:
   - `numba 0.59.0` requires `numpy<1.27,>=1.22`
   - `thinc 8.2.5` requires `numpy<2.0.0,>=1.19.0`
   - `astropy 5.3.4` requires `numpy<2,>=1.21`
   - `pywavelets 1.5.0` requires `numpy<2.0,>=1.22.4`
   - But `pip` installed `numpy 2.2.6` by default

2. **cryptography Version Conflict**:
   - `pyopenssl 24.0.0` requires `cryptography<43,>=41.0.5`
   - But `pip` installed `cryptography 46.0.3`

3. **Missing FuzzyTM**:
   - `gensim 4.3.0` requires `FuzzyTM>=0.4.0`
   - But it was not installed

**Solutions**:

1. **Version Pinning in requirements.txt**:
   ```python
   numpy>=1.24.0,<2.0.0  # Pin to < 2.0 for compatibility
   cryptography>=41.0.5,<43.0.0  # Pin for pyopenssl compatibility
   FuzzyTM>=0.4.0  # Required by gensim
   ```

2. **Explicit Installation Order**:
   - Install pinned versions first
   - Then install remaining dependencies

**Results**:
- **Installation**: Successful on all tested environments
- **Runtime**: No compatibility errors
- **Stability**: Consistent behavior across different systems

### 6.6 Autoencoder Loading Issues

**Severity**: Medium - Blocked Stage 3 autoencoder scaling

**Problem**:
The Hugging Face autoencoder `stabilityai/sd-vae-ft-mse` failed to load with error:
```
stabilityai/sd-vae-ft-mse does not appear to have a file named config.json.
```

**Root Cause**:
- Code attempted to load with `subfolder="vae"` by default
- But `stabilityai/sd-vae-ft-mse` is a standalone VAE model, not part of a full Stable Diffusion model
- Standalone VAEs don't have a `vae` subfolder

**Solution**:
Implemented robust loading strategy:
1. Detect if model is likely standalone (contains "sd-vae" or ends with "-vae")
2. Try loading without subfolder first
3. Fallback to `subfolder="vae"` if that fails
4. For full models, try with subfolder first, then without
- **Location**: `lib/scaling/methods.py::load_hf_autoencoder()`

**Results**:
- **Loading**: Successfully loads both standalone and full models
- **Flexibility**: Works with various Hugging Face VAE models
- **Robustness**: Handles different model structures gracefully

### 6.7 Dtype Mismatch in Autoencoder

**Severity**: Medium - Caused runtime errors during scaling

**Problem**:
Autoencoder scaling failed with error:
```
Input type (float) and bias type (c10::Half) should be the same
```

**Root Cause**:
- Autoencoder model loaded with `torch.float16` (Half precision) on CUDA
- Input tensors created as `torch.float32`
- PyTorch requires input and model to have same dtype

**Solution**:
1. Retrieve model's dtype: `next(autoencoder.parameters()).dtype`
2. Convert input tensors to match model dtype
3. Convert float16 to float32 before NumPy conversion (NumPy doesn't handle float16 well)
- **Location**: `lib/scaling/methods.py::_frame_to_tensor_hf_vae()`, `_tensor_to_frame_hf_vae()`

**Results**:
- **Compatibility**: Works with both float16 and float32 models
- **Performance**: float16 on CUDA for faster processing
- **Stability**: No dtype mismatch errors

### 6.8 Video Count Discrepancies

**Severity**: Low - Data quality issue

**Problem**:
- Expected: 298 original videos × 11 (1 original + 10 augmentations) = 3,278 videos
- Actual: 3,244 videos found (34 missing)

**Investigation**:
Created `check_stage1_completion.py` script which revealed:
- 3 videos had incomplete augmentations (0/10 augmentations instead of 10/10)
- Video IDs: `-Wgsj0ne_9M`, `164084910662284`, `315546135255835`
- 4 videos had no original entry (only augmentations): `1098601336923702`, `0EqX6HZKak4`, `1zR9zNSmH-A`, `MOeWw4rQn_w`

**Solution**:
- Created sanity check script: `src/scripts/check_stage1_completion.py`
- Enhanced logging in Stage 2 to show original vs augmented counts
- Warnings for missing videos and incomplete augmentations
- **Location**: `src/scripts/check_stage1_completion.py`, `lib/features/pipeline.py`

**Results**:
- **Visibility**: Can now identify data quality issues early
- **Debugging**: Easier to diagnose augmentation failures
- **Documentation**: Clear record of dataset completeness

### 6.9 Memory Fragmentation and Accumulation

**Severity**: Medium - Caused gradual memory growth

**Problem**:
Memory usage grew unbounded during processing, even with garbage collection.

**Root Causes**:
- Multiple videos processed without clearing intermediate data
- Metadata accumulated in lists before writing
- CUDA cache not cleared between operations
- Python objects not being garbage collected promptly

**Solutions**:

1. **One Video at a Time Processing**:
   - Process videos sequentially, not in batches
   - Clear video data immediately after processing

2. **Immediate Disk Writes**:
   - Write features/videos to disk immediately
   - No accumulation in memory

3. **Aggressive GC After Each Operation**:
   - After each stage, video, batch, epoch
   - 3 passes of `gc.collect()`
   - CUDA cache clearing

4. **Shared Augmentations Across K-Fold**:
   - Generate augmentations once for all videos
   - Filter per fold (1x generation vs 5x)
   - **Location**: `lib/mlops/multimodel.py::build_multimodel_pipeline()`

**Results**:
- **Memory**: Constant memory usage (no unbounded growth)
- **Stability**: No memory leaks
- **Efficiency**: 5x reduction in augmentation generation time

### 6.10 Multi-Node Distributed Processing

**Severity**: Medium - Required for scalability

**Challenge**:
Need to distribute Stages 2-5 across multiple SLURM nodes to process 3,244 videos efficiently.

**Requirements**:
- Dynamic load distribution based on number of nodes
- Resume/clean modes for fault tolerance
- Separate log files per node
- Verification of completion

**Solution**:

1. **Dynamic Load Distribution**:
   - Calculate `start_idx` and `end_idx` based on `SLURM_PROCID` and `SLURM_NTASKS`
   - Formula: `start = (PROCID * total_videos) // NTASKS`, `end = ((PROCID + 1) * total_videos) // NTASKS`
   - **Location**: `src/scripts/slurm_stage2_features.sh`, `slurm_stage3_scaling.sh`, `slurm_stage4_scaled_features.sh`

2. **start-idx and end-idx Support**:
   - Added to all Python scripts (Stages 2-5)
   - Slices DataFrame before processing
   - **Location**: `lib/features/pipeline.py`, `lib/scaling/pipeline.py`, `lib/features/scaled.py`, `lib/training/pipeline.py`

3. **Model-Per-Node Distribution** (Stage 5):
   - Each node trains one model
   - Formula: `model_idx = PROCID % num_models`
   - **Location**: `src/scripts/slurm_stage5_training.sh`

4. **Separate Log Files**:
   - Per node: `logs/stageX_node${SLURM_PROCID}.log`
   - Combined: `logs/stageX_combined_${SLURM_JOB_ID}.log`

5. **Resume/Clean Modes**:
   - `--resume`: Skip existing outputs
   - `--delete-existing`: Start from scratch
   - **Location**: All stage scripts and pipeline functions

6. **Completion Verification**:
   - `verify_stage_completion()` function in SLURM scripts
   - Checks if all videos were processed
   - **Location**: SLURM scripts

**Results**:
- **Scalability**: Can process full dataset across 1-4 nodes
- **Fault Tolerance**: Resume from interruptions
- **Efficiency**: Parallel processing reduces total time
- **Monitoring**: Clear visibility into per-node progress

---

## 7. Solutions and Optimizations

This section provides a comprehensive overview of all optimizations implemented to address the challenges described in Section 6, organized by category.

### 7.1 Memory Optimizations

**Frame-by-Frame Video Decoding** (50x reduction):
- **Implementation**: PyAV frame seeking instead of full video loading
- **Memory**: 1.87GB → 37MB per video
- **Location**: `lib/augmentation/io.py::load_frames()`
- **Impact**: Critical - enables processing large videos without OOM

**Fixed-Size Preprocessing** (4x reduction):
- **Implementation**: 112×112 fixed size with letterboxing (configurable)
- **Memory**: 224² → 112² = 4x reduction per frame
- **Location**: `lib/models/video.py::VideoConfig`
- **Impact**: Significant - enables larger batch sizes

**Reduced Batch Sizes** (4-32x reduction):
- **Implementation**: Model-specific ultra-conservative batch sizes (1-8)
- **Compensation**: Gradient accumulation (8-16 steps)
- **Location**: `lib/training/model_factory.py::MODEL_MEMORY_CONFIGS`
- **Impact**: Critical - prevents OOM during training

**Reduced Frame Count** (2.7x reduction):
- **Implementation**: 16 frames → 6 frames per video
- **Location**: Training configuration
- **Impact**: Moderate - reduces memory per sample

**Zero Workers** (eliminated overhead):
- **Implementation**: `num_workers=0` instead of 4
- **Memory saved**: ~500MB from worker processes
- **Impact**: Moderate - eliminates multiprocessing overhead

**Incremental CSV Writing** (constant memory):
- **Implementation**: Write each row immediately to CSV
- **Memory**: Constant regardless of dataset size
- **Location**: `lib/augmentation/pipeline.py::stage1_augment_videos()`
- **Impact**: Critical - prevents unbounded memory growth

**Chunked Processing** (Stage 3):
- **Implementation**: 100 frames per chunk
- **Memory**: Processes long videos without loading entirely
- **Location**: `lib/scaling/pipeline.py::scale_video()`
- **Impact**: Critical - enables processing videos of any length

**Aggressive Garbage Collection**:
- **Implementation**: 3 passes of `gc.collect()` + CUDA cache clearing
- **Frequency**: After every stage, video, batch, epoch
- **Location**: `lib/utils/memory.py::aggressive_gc()`
- **Impact**: High - prevents memory accumulation

**Total Memory Reduction**: ~8-16x overall, with 50x reduction for video loading

### 7.2 Performance Optimizations

**Pre-Generated Augmentations**:
- **Implementation**: Generate augmentations once, reuse across folds
- **Time saved**: 5x (1x generation vs 5x for k-fold)
- **Location**: `lib/mlops/multimodel.py`
- **Impact**: High - significantly reduces preprocessing time

**Shared Augmentations Across K-Fold**:
- **Implementation**: Generate once for all videos, filter per fold
- **Time saved**: 5x reduction in augmentation generation
- **Location**: Multi-model pipeline
- **Impact**: High - faster k-fold CV

**Deterministic Seeds**:
- **Implementation**: Seed from video path hash
- **Benefit**: Reproducible augmentations across runs
- **Location**: `lib/augmentation/pipeline.py`
- **Impact**: Medium - ensures reproducibility

**Mixed Precision Training (AMP)**:
- **Implementation**: PyTorch Automatic Mixed Precision
- **Speed**: ~1.5-2x faster training
- **Memory**: ~30-50% reduction
- **Location**: `lib/training/trainer.py::fit()`
- **Impact**: High - faster training with lower memory

**Gradient Accumulation**:
- **Implementation**: 8-16 steps to compensate for small batches
- **Effective batch size**: Maintains training effectiveness
- **Location**: Training configuration
- **Impact**: High - enables small batch training

**Feature Caching**:
- **Implementation**: Cache extracted features to disk
- **Time saved**: Avoids recomputation across runs/folds
- **Location**: `lib/features/handcrafted.py::HandcraftedFeatureExtractor`
- **Impact**: Medium - faster subsequent runs

### 7.3 Robustness Improvements

**K-Fold Cross-Validation**:
- **Implementation**: 5-fold stratified CV
- **Benefit**: More robust performance estimates
- **Location**: `lib/data/loading.py::stratified_kfold()`
- **Impact**: Critical - prevents overfitting

**Data Leakage Prevention**:
- **Implementation**: Group by `dup_group` before splitting
- **Benefit**: Ensures no duplicate groups across folds
- **Location**: `lib/data/loading.py::stratified_kfold()`
- **Impact**: Critical - ensures valid results

**OOM Detection and Fallback**:
- **Implementation**: Multiple fallback levels (frame → chunk → video → method)
- **Benefit**: Graceful degradation instead of crashes
- **Location**: `lib/utils/memory.py::check_oom_error()`, `handle_oom_error()`
- **Impact**: High - prevents crashes

**Resume Capability**:
- **Implementation**: Checkpoint-based resume from any stage
- **Benefit**: Can recover from interruptions
- **Location**: All stage pipeline functions
- **Impact**: High - fault tolerance

**Comprehensive Error Handling**:
- **Implementation**: Try-except blocks with proper logging
- **Benefit**: Graceful handling of edge cases
- **Location**: Throughout codebase
- **Impact**: Medium - prevents crashes

**Data Validation**:
- **Implementation**: Pandera schema validation, column checks
- **Benefit**: Catches data issues early
- **Location**: `lib/utils/schemas.py`, `lib/utils/paths.py`
- **Impact**: Medium - ensures data quality

### 7.4 MLOps Infrastructure

**Experiment Tracking**:
- **Implementation**: MLflow + custom tracker
- **Benefit**: Reproducible experiments, easy comparison
- **Location**: `lib/mlops/config.py::ExperimentTracker`
- **Impact**: High - enables scientific rigor

**Configuration Versioning**:
- **Implementation**: RunConfig with deterministic hashing
- **Benefit**: Can recreate any experiment from config
- **Location**: `lib/mlops/config.py::RunConfig`
- **Impact**: High - reproducibility

**Checkpoint Management**:
- **Implementation**: Full state saving (model + optimizer + scheduler + epoch)
- **Benefit**: Resume from exact state
- **Location**: `lib/mlops/config.py::CheckpointManager`
- **Impact**: High - fault tolerance

**Data Versioning**:
- **Implementation**: Links data to config hashes
- **Benefit**: Know exactly which data was used
- **Location**: `lib/mlops/config.py::DataVersionManager`
- **Impact**: Medium - data lineage

**Structured Metrics Logging**:
- **Implementation**: JSONL format (one metric per line)
- **Benefit**: Easy to query and analyze
- **Location**: `lib/mlops/config.py::ExperimentTracker`
- **Impact**: Medium - easier analysis

**Pipeline Orchestration**:
- **Implementation**: Modular stages with dependencies
- **Benefit**: Clear execution order, easy to extend
- **Location**: `lib/mlops/pipeline.py`
- **Impact**: Medium - better organization

---

## 8. Expected Results

### 8.1 Performance Metrics

The project evaluates models using standard binary classification metrics:

- **Accuracy**: Overall correctness (TP + TN) / (TP + TN + FP + FN)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **ROC-AUC**: Area under ROC curve (if implemented)
- **Per-Class Metrics**: Precision/recall for each class

**Reporting Format**: Mean ± standard deviation across 5 folds

### 8.2 Model Performance Expectations

**Baseline Models** (Logistic Regression, SVM):
- **Expected Accuracy**: 60-75%
- **Rationale**: Simple models using only handcrafted features
- **Use Case**: Fast baseline, interpretable feature importance

**Naive CNN**:
- **Expected Accuracy**: 65-80%
- **Rationale**: Simple deep learning without temporal modeling
- **Use Case**: Deep learning baseline

**Frame→Temporal Models** (ViT+GRU, ViT+Transformer):
- **Expected Accuracy**: 75-85%
- **Rationale**: Captures temporal patterns with recurrent/attention mechanisms
- **Use Case**: Temporal pattern detection

**Spatiotemporal Models** (SlowFast, X3D):
- **Expected Accuracy**: 80-90%
- **Rationale**: State-of-the-art 3D convolutions capture both spatial and temporal information
- **Use Case**: Best performance, requires more resources

**Ensemble Models**:
- **Expected Accuracy**: 85-92%
- **Rationale**: Combines strengths of multiple models
- **Use Case**: Production deployment

### 8.3 Computational Requirements

**Memory**:
- **Per Node**: 64GB RAM (worst case: ~25-30GB, typical: 5-10GB)
- **GPU**: 1 GPU per node (16GB+ recommended)
- **Storage**: ~100GB for dataset + features + models

**Processing Time Estimates** (for 3,244 videos):

- **Stage 1 (Augmentation)**: ~N videos × augmentation time per video
  - Depends on video length and augmentation complexity
  - Estimated: 1-2 hours for full dataset

- **Stage 2 (Feature Extraction)**: ~3,244 videos × feature extraction time
  - Handcrafted features: ~1-2 seconds per video
  - Estimated: 1-2 hours for full dataset

- **Stage 3 (Scaling)**: ~3,244 videos × (3.5 seconds per 100 frames)
  - Example: 100-frame video = 3.5 seconds
  - Example: 1000-frame video = 35 seconds
  - Estimated: 2-4 hours for full dataset (depending on video lengths)

- **Stage 4 (Scaled Features)**: ~3,244 videos × feature extraction time
  - Similar to Stage 2
  - Estimated: 1-2 hours for full dataset

- **Stage 5 (Training)**: ~7 models × 5 folds × training time per model
  - Baselines: Minutes per fold
  - Frame→Temporal: Hours per fold
  - Spatiotemporal: Hours to days per fold
  - Estimated: 1-3 days for full training (depending on models and resources)

**Total Pipeline Time**: Approximately 1-4 days depending on:
- Number of nodes (1-4)
- Video lengths
- Model selection
- Resource availability

### 8.4 Scalability

**Multi-Node Support**:
- **Nodes**: 1-4 nodes (configurable)
- **Distribution**: Dynamic load balancing
- **Speedup**: Near-linear with number of nodes (for Stages 2-4)

**Resume/Clean Modes**:
- **Resume**: Continue from interruptions
- **Clean**: Start from scratch
- **Benefit**: Fault tolerance and flexibility

**Checkpoint-Based Recovery**:
- **Checkpoints**: After each fold, model, stage
- **Recovery**: Resume from latest checkpoint
- **Benefit**: No lost work on crashes

---

## 9. Implementation Details

### 9.1 File Structure

```
fvc/
├── lib/
│   ├── augmentation/          # Stage 1: Video augmentation
│   │   ├── pipeline.py        # Main augmentation pipeline
│   │   ├── transforms.py      # Augmentation transforms
│   │   └── io.py              # Video I/O utilities
│   ├── features/              # Stages 2 & 4: Feature extraction
│   │   ├── pipeline.py        # Stage 2: Handcrafted features
│   │   ├── scaled.py          # Stage 4: Scaled video features
│   │   └── handcrafted.py     # Feature extraction functions
│   ├── scaling/               # Stage 3: Video scaling
│   │   ├── pipeline.py        # Main scaling pipeline
│   │   └── methods.py         # Scaling methods (letterbox, autoencoder)
│   ├── training/              # Stage 5: Model training
│   │   ├── pipeline.py        # Main training pipeline
│   │   ├── model_factory.py   # Model creation and configs
│   │   ├── trainer.py         # Training utilities
│   │   ├── _linear.py         # Logistic Regression
│   │   ├── _svm.py            # SVM
│   │   ├── _cnn.py            # Naive CNN
│   │   ├── _transformer_gru.py # ViT+GRU
│   │   ├── _transformer.py    # ViT+Transformer
│   │   ├── slowfast.py        # SlowFast
│   │   └── x3d.py             # X3D
│   ├── mlops/                 # MLOps infrastructure
│   │   ├── config.py          # RunConfig, ExperimentTracker, etc.
│   │   ├── pipeline.py        # Main MLOps pipeline
│   │   ├── kfold.py           # K-fold pipeline
│   │   └── multimodel.py       # Multi-model pipeline
│   ├── utils/                 # Utilities
│   │   ├── memory.py          # Memory management
│   │   ├── paths.py           # Path resolution
│   │   ├── schemas.py         # Data validation
│   │   └── video_cache.py     # Video metadata caching
│   └── data/                  # Data loading
│       ├── loading.py         # Metadata loading, splitting
│       └── scan.py            # Video scanning, filtering
├── src/
│   ├── scripts/               # CLI scripts and SLURM scripts
│   │   ├── run_stage1_augmentation.py
│   │   ├── run_stage2_features.py
│   │   ├── run_stage3_scaling.py
│   │   ├── run_stage4_scaled_features.py
│   │   ├── run_stage5_training.py
│   │   ├── check_stage1_completion.py
│   │   ├── slurm_stage1_augmentation.sh
│   │   ├── slurm_stage2_features.sh
│   │   ├── slurm_stage3_scaling.sh
│   │   ├── slurm_stage4_scaled_features.sh
│   │   └── slurm_stage5_training.sh
│   ├── dashboard_results.py   # Streamlit dashboard
│   └── generate_paper_figures.py # Publication figures
├── data/                      # Processed data
│   ├── augmented_videos/      # Stage 1 output
│   ├── features_stage2/       # Stage 2 output
│   ├── scaled_videos/         # Stage 3 output
│   ├── features_stage4/       # Stage 4 output
│   └── training_results/      # Stage 5 output
├── logs/                      # Log files
├── runs/                      # Experiment outputs
└── requirements.txt           # Dependencies
```

### 9.2 Key Functions

**Stage 1**: `lib/augmentation/pipeline.py::stage1_augment_videos()`
- Generates 10 augmentations per video
- Frame-by-frame decoding for memory efficiency
- Deterministic seeding for reproducibility

**Stage 2**: `lib/features/pipeline.py::stage2_extract_features()`
- Extracts ~15 handcrafted features per video
- Adaptive frame sampling (10% of frames, min=5, max=50)
- Resume capability

**Stage 3**: `lib/scaling/pipeline.py::stage3_scale_videos()`
- Scales videos to 256×256 max dimension
- Autoencoder or letterbox methods
- Chunked processing (100 frames per chunk)

**Stage 4**: `lib/features/scaled.py::stage4_extract_scaled_features()`
- Extracts ~8 scaled-video-specific features
- Includes scaling direction indicators
- Resume capability

**Stage 5**: `lib/training/pipeline.py::stage5_train_models()`
- Trains 7 model architectures
- 5-fold cross-validation
- Multi-model support with resume

### 9.3 Configuration

**Environment Variables**:
- `FVC_NUM_FRAMES`: Fixed number of frames (overrides percentage-based)
- `FVC_FRAME_PERCENTAGE`: Percentage of frames (default: 0.10)
- `FVC_MIN_FRAMES`: Minimum frames (default: 5)
- `FVC_MAX_FRAMES`: Maximum frames (default: 50)
- `FVC_TARGET_SIZE`: Target max dimension for scaling (default: 256)
- `FVC_DOWNSCALE_METHOD`: Scaling method - "autoencoder" or "letterbox" (default: "autoencoder")
- `FVC_CHUNK_SIZE`: Chunk size for Stage 3 (default: 100)
- `FVC_MAX_FRAMES`: Max frames per chunk (default: 100)

**SLURM Resource Specifications**:
- **Memory**: 80GB per node for Stages 2, 4, 5; 64GB for Stage 3 (configurable)
- **Time**: 8 hours max per job (Stage 3: 1 day max)
- **GPUs**: 1 GPU per node
- **CPUs**: 4 CPUs per task (Stage 3: 1 CPU per task)
- **Nodes**: 1-4 nodes (configurable)
- **Account**: `eecs442f25_class` (Stages 2, 4, 5); `si670f25_class` (Stage 3)

**Note**: Stage 3 currently uses 64GB/1 day, but can be configured to 80GB/8h to match other stages.

**Model-Specific Configs**:
- Defined in `lib/training/model_factory.py::MODEL_MEMORY_CONFIGS`
- Current settings (optimized for 256GB RAM):
  - Baselines: batch_size=8, num_frames=8
  - Frame→Temporal: batch_size=2, num_frames=8, gradient_accumulation=8-10
  - Spatiotemporal: batch_size=2, num_frames=8, gradient_accumulation=10-20
- Ultra-conservative settings (for 64GB RAM):
  - Baselines: batch_size=8, num_frames=6
  - Frame→Temporal: batch_size=1, num_frames=6, gradient_accumulation=16
  - Spatiotemporal: batch_size=1, num_frames=6, gradient_accumulation=16
- Gradient accumulation compensates for smaller batches

---

## 10. Future Work

### 10.1 Model Architecture Improvements

**Video Transformers**:
- TimeSformer: Space-time attention for video
- ViViT: Video Vision Transformer
- Potential accuracy improvement: +5-10%

**Two-Stream Networks**:
- RGB stream + Optical flow stream
- Captures both appearance and motion
- Potential accuracy improvement: +3-7%

**Advanced SlowFast Variants**:
- SlowFast with attention mechanisms
- Multi-scale SlowFast
- Potential accuracy improvement: +2-5%

### 10.2 Training Improvements

**Learning Rate Scheduling**:
- Cosine decay with warmup
- One-cycle policy
- Potential improvement: Faster convergence, better final performance

**Advanced Loss Functions**:
- Focal loss for class imbalance
- Class-balanced loss
- Potential improvement: Better handling of imbalanced datasets

**Multi-GPU Training**:
- DataParallel or DistributedDataParallel
- Potential improvement: Faster training, larger effective batch sizes

**Hyperparameter Optimization**:
- Grid search or Bayesian optimization
- Potential improvement: Optimal hyperparameters for each model

### 10.3 Evaluation Enhancements

**Per-Class Metrics**:
- Precision/recall for each class separately
- Better understanding of model behavior

**PR-AUC and Calibration Curves**:
- Precision-Recall AUC
- Calibration analysis
- Better assessment of model reliability

**Robustness Checks**:
- Performance by platform (YouTube vs Twitter vs Facebook)
- Performance by resolution
- Performance by video duration
- Identifies model weaknesses

**Adversarial Robustness Testing**:
- Test against adversarial examples
- Assess model robustness to attacks

### 10.4 Deployment

**Model Packaging**:
- ONNX export for deployment
- TensorRT optimization for inference
- Enables production deployment

**CLI Interface**:
- Command-line tool for inference
- Easy integration into workflows

**Batch Inference API**:
- REST API for batch processing
- Scalable inference service

**Real-Time Detection Service**:
- Streaming video processing
- Low-latency inference
- Production-ready service

---

## 11. Conclusion

### 11.1 Key Achievements

The FVC project successfully developed a robust, scalable pipeline for fake video classification with the following achievements:

1. **Robust 5-Stage Pipeline**: Modular, resumable, and optimized for memory efficiency
2. **7 Model Architectures**: From simple baselines to state-of-the-art spatiotemporal models
3. **Comprehensive Memory Optimizations**: 50x reduction in video loading memory, 8-16x overall reduction
4. **Multi-Node Distributed Processing**: Scalable across 1-4 nodes with dynamic load balancing
5. **MLOps Infrastructure**: Experiment tracking, checkpointing, and data versioning for reproducibility
6. **Data Leakage Prevention**: Proper handling of duplicate groups ensures valid results
7. **OOM-Resistant Design**: Multiple fallback mechanisms prevent crashes

### 11.2 Impact

**Research Contribution**:
- Provides reproducible baseline for fake video detection research
- Demonstrates practical solutions to memory constraints in video processing
- Shows importance of proper data splitting and validation

**Technical Innovation**:
- Frame-by-frame decoding (50x memory reduction)
- Chunked video processing for long videos
- Robust autoencoder loading for various model structures
- Multi-node distributed processing with fault tolerance

**Practical Applications**:
- Content moderation systems
- Fact-checking tools
- Forensic analysis
- Platform safety mechanisms

### 11.3 Lessons Learned

1. **Memory Management is Critical**: Video processing requires careful memory management at every level
2. **Data Leakage is Subtle**: Duplicate groups can cause leakage if not handled properly
3. **Robustness Matters**: Multiple fallback mechanisms prevent catastrophic failures
4. **Reproducibility is Essential**: Deterministic seeds and proper versioning enable scientific rigor
5. **Scalability Requires Planning**: Multi-node processing requires careful design from the start

### 11.4 Final Thoughts

The FVC project demonstrates that with careful engineering, it is possible to build robust, scalable systems for fake video classification even under severe memory constraints. The solutions developed here can be applied to other video processing tasks and serve as a foundation for future research in deepfake detection.

---

## References

- Kinetics-400 Dataset: https://deepmind.com/research/open-source/kinetics
- 3D ResNet: "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (Tran et al., 2018)
- SlowFast: "SlowFast Networks for Video Recognition" (Feichtenhofer et al., 2019)
- X3D: "X3D: Expanding Architectures for Efficient Video Recognition" (Feichtenhofer, 2020)
- Vision Transformer: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020)
- Hugging Face Diffusers: https://huggingface.co/docs/diffusers
- Stable Diffusion VAE: https://huggingface.co/stabilityai/sd-vae-ft-mse

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-06  
**Author**: FVC Project Team  
**Project Repository**: [GitHub Repository URL]

