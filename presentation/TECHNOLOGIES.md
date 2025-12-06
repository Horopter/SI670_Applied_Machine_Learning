# FVC Project: Detailed Technology Stack

This document provides an exhaustive list of all technologies, libraries, and tools used in the FVC project, along with their versions, use cases, and implementation locations.

## Core Deep Learning Frameworks

### PyTorch 2.0+
- **Version**: `torch>=2.0.0`
- **Purpose**: Primary deep learning framework
- **Use Cases**:
  - Model definition and training
  - Automatic Mixed Precision (AMP) for memory efficiency
  - CUDA support for GPU acceleration
  - Tensor operations and neural network layers
- **Key Features Used**:
  - `torch.nn.Module` for model definitions
  - `torch.optim.Adam` for optimization
  - `torch.cuda.amp.autocast()` for mixed precision
  - `torch.utils.data.DataLoader` for data loading
- **Location**: Used throughout `lib/training/`, `lib/models/`, `lib/scaling/`

### torchvision 0.15.0+
- **Version**: `torchvision>=0.15.0`
- **Purpose**: Pretrained video models and utilities
- **Use Cases**:
  - Pretrained 3D CNNs (R3D, X3D, SlowFast, I3D, R(2+1)D)
  - Video preprocessing utilities
  - Transfer learning from Kinetics-400
- **Models Used**:
  - `torchvision.models.video.r3d_18` (3D ResNet-18)
  - `torchvision.models.video.x3d_m` (X3D Medium)
  - `torchvision.models.video.slowfast_r50` (SlowFast ResNet-50)
  - `torchvision.models.video.i3d_r50` (I3D ResNet-50)
  - `torchvision.models.video.r2plus1d_18` (R(2+1)D-18)
- **Location**: `lib/training/x3d.py`, `lib/training/slowfast.py`, `lib/training/i3d.py`, `lib/training/r2plus1d.py`

### timm 0.9.0+
- **Version**: `timm>=0.9.0`
- **Purpose**: Vision Transformer models
- **Use Cases**:
  - Pretrained ViT-B/16 for frame-level feature extraction
  - ImageNet pretrained weights
- **Models Used**:
  - `timm.create_model('vit_base_patch16_224', pretrained=True)`
- **Location**: `lib/training/_transformer_gru.py`, `lib/training/_transformer.py`

### Hugging Face Libraries

#### transformers 4.30.0+
- **Version**: `transformers>=4.30.0`
- **Purpose**: General transformer models and utilities
- **Use Cases**:
  - Model loading utilities
  - Tokenizers (if needed for future text features)
- **Location**: Used indirectly through diffusers

#### diffusers 0.21.0+
- **Version**: `diffusers>=0.21.0`
- **Purpose**: Stable Diffusion and VAE models for video scaling
- **Use Cases**:
  - `AutoencoderKL` for high-quality video scaling
  - Pretrained VAE models from Hugging Face
- **Models Used**:
  - `stabilityai/sd-vae-ft-mse` (default)
  - `stabilityai/sd-vae-ft-ema` (alternative)
- **Location**: `lib/scaling/methods.py::load_hf_autoencoder()`

#### accelerate 0.20.0+
- **Version**: `accelerate>=0.20.0`
- **Purpose**: Accelerated model loading and inference
- **Use Cases**:
  - Optimized model loading for Hugging Face models
  - Multi-GPU support (future)
- **Location**: Used by diffusers and transformers

## Data Processing and Analytics

### Polars 0.19.0+
- **Version**: `polars>=0.19.0`
- **Purpose**: Fast DataFrame operations
- **Use Cases**:
  - Metadata loading and manipulation
  - Filtering and slicing video datasets
  - Aggregations and joins
  - 10-100x faster than pandas for large datasets
- **Key Operations**:
  - `pl.read_csv()`, `pl.read_parquet()`, `pl.read_ipc()`
  - `pl.DataFrame.filter()`, `pl.DataFrame.slice()`
  - `pl.DataFrame.group_by()`, `pl.DataFrame.agg()`
- **Location**: Used throughout `lib/data/`, `lib/features/`, `lib/scaling/`, `lib/training/`

### PyArrow 14.0.0+
- **Version**: `pyarrow>=14.0.0`
- **Purpose**: Arrow columnar format for efficient data storage
- **Use Cases**:
  - Serialization of DataFrames (`.arrow` format)
  - Interoperability with Polars
  - Efficient binary format for metadata
- **Formats Supported**: `.arrow` (IPC), `.parquet`, `.csv`
- **Location**: Used by Polars for I/O operations

### DuckDB 0.9.0+
- **Version**: `duckdb>=0.9.0`
- **Purpose**: Analytical database for querying experiment results
- **Use Cases**:
  - Fast aggregations on large metadata files
  - Complex queries on experiment results
  - Analytics and reporting
- **Location**: `lib/utils/duckdb_analytics.py`

### Pandas 2.0.0+
- **Version**: `pandas>=2.0.0`
- **Purpose**: Legacy DataFrame operations (used in some modules)
- **Use Cases**:
  - Compatibility with older code
  - Some sklearn integrations
- **Note**: Most code migrated to Polars for performance
- **Location**: Used in some legacy modules

## Video and Image Processing

### OpenCV 4.8.0+
- **Version**: `opencv-python>=4.8.0`
- **Purpose**: Video decoding, image processing, feature extraction
- **Use Cases**:
  - Video frame extraction: `cv2.VideoCapture()`
  - Image resizing: `cv2.resize()` with letterboxing
  - Feature extraction:
    - Canny edge detection: `cv2.Canny()`
    - Laplacian operator: `cv2.Laplacian()`
    - Sobel gradients: `cv2.Sobel()`
    - DCT computation: `cv2.dct()`
    - Filtering: `cv2.filter2D()`
  - Color space conversion: `cv2.cvtColor()`
- **Location**: `lib/features/handcrafted.py`, `lib/features/scaled.py`, `lib/scaling/methods.py`

### PyAV 10.0.0+
- **Version**: `av>=10.0.0`
- **Purpose**: Advanced video decoding with frame-by-frame seeking
- **Use Cases**:
  - Frame-by-frame decoding (critical for memory efficiency)
  - Seeking to specific timestamps: `container.seek()`
  - Video metadata extraction: `stream.frames`, `stream.width`, `stream.height`
  - Multiple codec support (H.264, VP9, MPEG-4)
- **Key Features**:
  - `av.open()` for opening video files
  - `container.seek()` for seeking to specific frames
  - `packet.decode()` for decoding frames
  - `frame.to_ndarray()` for converting to NumPy arrays
- **Location**: `lib/augmentation/io.py`, `lib/features/pipeline.py`, `lib/scaling/pipeline.py`

### Pillow 10.0.0+
- **Version**: `Pillow>=10.0.0`
- **Purpose**: Image processing for augmentations
- **Use Cases**:
  - Image loading and saving: `Image.open()`, `Image.save()`
  - Augmentation transforms:
    - Rotation: `Image.rotate()`
    - Flipping: `Image.transpose()`
    - Color adjustments: `ImageEnhance.Brightness/Contrast/Color`
    - Blur: `ImageFilter.GaussianBlur()`
  - Format conversion: RGB, grayscale
- **Location**: `lib/augmentation/transforms.py`

### torchcodec 0.1.0+
- **Version**: `torchcodec>=0.1.0`
- **Purpose**: PyTorch-native video codec (optional, for future use)
- **Use Cases**:
  - Potential future optimization for video decoding
  - GPU-accelerated video processing
- **Location**: Listed in requirements, not actively used yet

## Machine Learning Libraries

### scikit-learn 1.3.0+
- **Version**: `scikit-learn>=1.3.0`
- **Purpose**: Baseline models and metrics
- **Use Cases**:
  - Baseline models:
    - `LogisticRegression` for logistic regression baseline
    - `LinearSVC` for SVM baseline
  - Metrics:
    - `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`
  - Feature scaling:
    - `StandardScaler` for feature normalization
- **Location**: `lib/training/_linear.py`, `lib/training/_svm.py`, `lib/training/pipeline.py`

### XGBoost 2.0.0+
- **Version**: `xgboost>=2.0.0`
- **Purpose**: Gradient boosting for ensemble models
- **Use Cases**:
  - Ensemble model training (future)
  - Feature importance analysis
- **Location**: `lib/training/_xgboost_pretrained.py` (if implemented)

### scipy 1.11.0+
- **Version**: `scipy>=1.11.0`
- **Purpose**: Scientific computing and signal processing
- **Use Cases**:
  - Signal processing functions
  - Statistical functions
  - DCT and FFT operations (if needed)
- **Location**: Used in feature extraction modules

### statsmodels 0.14.0+
- **Version**: `statsmodels>=0.14.0`
- **Purpose**: Statistical modeling (optional, for future analysis)
- **Use Cases**:
  - Statistical significance testing
  - Confidence intervals
- **Location**: Listed in requirements, not actively used yet

## MLOps and Infrastructure

### MLflow 2.8.0+
- **Version**: `mlflow>=2.8.0`
- **Purpose**: Experiment tracking and model registry
- **Use Cases**:
  - Logging metrics, parameters, and artifacts
  - Model versioning and registry
  - Experiment comparison
  - UI for browsing experiments
- **Location**: `lib/mlops/mlflow_tracker.py`

### Apache Airflow 2.7.0+
- **Version**: `apache-airflow>=2.7.0`
- **Purpose**: Workflow orchestration (optional, for production)
- **Use Cases**:
  - Pipeline scheduling
  - Dependency management
  - Production workflow automation
- **Location**: `airflow/dags/fvc_pipeline_dag.py` (if implemented)

### SLURM
- **Version**: System-dependent (cluster-specific)
- **Purpose**: Cluster job management and resource allocation
- **Use Cases**:
  - Multi-node job submission
  - Resource allocation (CPU, GPU, memory, time)
  - Job scheduling and queuing
  - Log file management
- **Key Features**:
  - `sbatch` for job submission
  - `srun` for multi-node execution
  - Environment variables: `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_JOB_ID`
- **Location**: `src/scripts/slurm_*.sh` scripts

## Data Validation

### Pandera 0.18.0+
- **Version**: `pandera>=0.18.0`
- **Purpose**: Schema validation for DataFrames
- **Use Cases**:
  - Validating metadata schemas at each stage
  - Ensuring required columns are present
  - Type checking and constraints
  - Data quality assurance
- **Schemas Defined**:
  - `Stage1AugmentedMetadataSchema`
  - `Stage2FeaturesMetadataSchema`
  - `Stage3ScaledMetadataSchema`
  - `Stage4ScaledFeaturesMetadataSchema`
- **Location**: `lib/utils/schemas.py`

## Visualization and Dashboards

### Streamlit 1.28.0+
- **Version**: `streamlit>=1.28.0`
- **Purpose**: Interactive results dashboard
- **Use Cases**:
  - Model performance comparisons
  - K-fold cross-validation analysis
  - Interactive plots and tables
  - Results exploration
- **Location**: `src/dashboard_results.py`

### Plotly 5.17.0+
- **Version**: `plotly>=5.17.0`
- **Purpose**: Interactive plots
- **Use Cases**:
  - ROC curves
  - Precision-Recall curves
  - Training curves (loss/accuracy over epochs)
  - Interactive exploration
- **Location**: `src/dashboard_results.py`, `src/generate_paper_figures.py`

### Matplotlib 3.7.0+
- **Version**: `matplotlib>=3.7.0`
- **Purpose**: Publication-ready static plots
- **Use Cases**:
  - High-quality figures for papers
  - Statistical visualizations
  - Custom styling and formatting
- **Location**: `src/generate_paper_figures.py`

### Seaborn 0.12.0+
- **Version**: `seaborn>=0.12.0`
- **Purpose**: Statistical visualizations
- **Use Cases**:
  - Violin plots for k-fold distributions
  - Statistical significance visualizations
  - Enhanced matplotlib plots
- **Location**: `src/generate_paper_figures.py`

## Utilities

### tqdm 4.65.0+
- **Version**: `tqdm>=4.65.0`
- **Purpose**: Progress bars for long-running operations
- **Use Cases**:
  - Video processing progress
  - Training progress
  - Feature extraction progress
- **Location**: Used throughout pipeline scripts

### psutil 5.9.0+
- **Version**: `psutil>=5.9.0`
- **Purpose**: System and process utilities
- **Use Cases**:
  - Memory profiling: `psutil.Process().memory_info()`
  - CPU usage monitoring
  - System resource tracking
- **Location**: `lib/utils/memory.py::get_memory_stats()`

### joblib 1.3.0+
- **Version**: `joblib>=1.3.0`
- **Purpose**: Model serialization for sklearn models
- **Use Cases**:
  - Saving/loading sklearn models (LogisticRegression, SVM)
  - Parallel processing (if needed)
- **Location**: `lib/training/_linear.py`, `lib/training/_svm.py`

### numpy 1.24.0+ (<2.0.0)
- **Version**: `numpy>=1.24.0,<2.0.0` (pinned for compatibility)
- **Purpose**: Numerical computing
- **Use Cases**:
  - Array operations
  - Mathematical computations
  - Image/video data representation
- **Compatibility Note**: Pinned to <2.0.0 for compatibility with:
  - `numba 0.59.0` (requires `numpy<1.27,>=1.22`)
  - `thinc 8.2.5` (requires `numpy<2.0.0,>=1.19.0`)
  - `astropy 5.3.4` (requires `numpy<2,>=1.21`)
  - `pywavelets 1.5.0` (requires `numpy<2.0,>=1.22.4`)

### cryptography 41.0.5+ (<43.0.0)
- **Version**: `cryptography>=41.0.5,<43.0.0` (pinned for compatibility)
- **Purpose**: Cryptographic functions (used by other libraries)
- **Compatibility Note**: Pinned for `pyopenssl 24.0.0` compatibility (requires `cryptography<43,>=41.0.5`)

### FuzzyTM 0.4.0+
- **Version**: `FuzzyTM>=0.4.0`
- **Purpose**: Required dependency for gensim
- **Compatibility Note**: `gensim 4.3.0` requires `FuzzyTM>=0.4.0`

## System Tools

### ffprobe (FFmpeg)
- **Version**: System-dependent (via `module load ffmpeg`)
- **Purpose**: Video metadata extraction
- **Use Cases**:
  - Extracting codec information (bitrate, fps, resolution)
  - Video file analysis
  - Codec cue extraction
- **Location**: `lib/features/handcrafted.py::extract_codec_cues()`

### Python 3.11+
- **Version**: Python 3.11.7 (via Anaconda)
- **Purpose**: Primary programming language
- **Use Cases**: All project code
- **Location**: All Python files

## Development and Testing

### pytest
- **Purpose**: Unit testing framework
- **Use Cases**: Testing individual functions and modules
- **Location**: `test/` directory

### ipykernel
- **Purpose**: Jupyter notebook kernel (if notebooks are used)
- **Use Cases**: Interactive development and exploration

### papermill
- **Purpose**: Parameterized notebook execution (if notebooks are used)
- **Use Cases**: Automated notebook runs with different parameters

## Technology Stack Summary

### By Category

**Deep Learning**: PyTorch, torchvision, timm, Hugging Face (transformers, diffusers, accelerate)

**Data Processing**: Polars, PyArrow, DuckDB, Pandas, NumPy

**Video/Image Processing**: OpenCV, PyAV, Pillow, torchcodec

**Machine Learning**: scikit-learn, XGBoost, scipy, statsmodels

**MLOps**: MLflow, Apache Airflow, SLURM

**Validation**: Pandera

**Visualization**: Streamlit, Plotly, Matplotlib, Seaborn

**Utilities**: tqdm, psutil, joblib

**System Tools**: ffprobe (FFmpeg), Python 3.11+

### By Memory Impact

**High Memory**: PyTorch models, torchvision models, Hugging Face models, video data

**Medium Memory**: Polars DataFrames, NumPy arrays, OpenCV images

**Low Memory**: scikit-learn models, metadata, utilities

### By Processing Speed

**Fast**: Polars operations, NumPy operations, scikit-learn inference

**Medium**: OpenCV operations, PyAV decoding, feature extraction

**Slow**: Deep learning training, autoencoder scaling, video processing

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-06

