#!/usr/bin/env python3
"""
Generate comprehensive presentation-quality notebooks for all models 5c-5u.

Each notebook includes:
- Model architecture deep-dive
- Training methodology (5-fold CV, hyperparameters, regularization)
- MLOps integration (MLflow, DuckDB, Airflow)
- Example usage with saved models
- Video demonstrations
- Commented training code
"""

import json
from pathlib import Path

# Model configurations
MODEL_CONFIGS = {
    "5c": {
        "name": "Naive CNN",
        "model_type": "naive_cnn",
        "description": "Simple 3D CNN baseline that processes frames independently",
        "architecture": "2D CNN per frame → temporal averaging",
        "key_features": ["Chunked processing (10 frames)", "Memory efficient", "Baseline comparison"],
        "input_shape": "(N, C, T, H, W) where T=1000, H=W=256",
        "pretrained": False,
        "category": "CNN Baseline"
    },
    "5d": {
        "name": "Pretrained Inception",
        "model_type": "pretrained_inception",
        "description": "R3D-18 pretrained backbone + Inception3D head",
        "architecture": "torchvision R3D-18 (Kinetics-400) → Inception3DBlock → Classification",
        "key_features": ["Pretrained on Kinetics-400", "Freezable backbone", "Variable aspect ratio support"],
        "input_shape": "(N, C, T, H, W) where T=1000, H=W=256",
        "pretrained": True,
        "category": "Pretrained CNN"
    },
    "5e": {
        "name": "Variable AR CNN",
        "model_type": "variable_ar_cnn",
        "description": "Inception-like 3D CNN supporting variable aspect ratios",
        "architecture": "Inception3D blocks → AdaptiveAvgPool3d → Classification",
        "key_features": ["Variable aspect ratio support", "Global pooling", "Efficient memory usage"],
        "input_shape": "(N, C, T, H, W) with arbitrary H, W",
        "pretrained": False,
        "category": "CNN"
    },
    "5f": {
        "name": "XGBoost + Pretrained Inception",
        "model_type": "xgboost_pretrained_inception",
        "description": "XGBoost classifier on features extracted from Pretrained Inception",
        "architecture": "Pretrained Inception → Feature extraction → XGBoost",
        "key_features": ["Pretrained features", "Gradient boosting", "Fast inference"],
        "input_shape": "Features from Pretrained Inception model",
        "pretrained": True,
        "category": "XGBoost + Pretrained"
    },
    "5g": {
        "name": "XGBoost + I3D",
        "model_type": "xgboost_i3d",
        "description": "XGBoost classifier on features extracted from I3D",
        "architecture": "I3D (PyTorchVideo) → Feature extraction → XGBoost",
        "key_features": ["I3D features", "Gradient boosting", "Spatiotemporal patterns"],
        "input_shape": "Features from I3D model",
        "pretrained": True,
        "category": "XGBoost + Pretrained"
    },
    "5h": {
        "name": "XGBoost + R(2+1)D",
        "model_type": "xgboost_r2plus1d",
        "description": "XGBoost classifier on features extracted from R(2+1)D",
        "architecture": "R(2+1)D (torchvision) → Feature extraction → XGBoost",
        "key_features": ["Factorized 3D convolutions", "Gradient boosting", "Efficient 3D patterns"],
        "input_shape": "Features from R(2+1)D model",
        "pretrained": True,
        "category": "XGBoost + Pretrained"
    },
    "5i": {
        "name": "XGBoost + ViT-GRU",
        "model_type": "xgboost_vit_gru",
        "description": "XGBoost classifier on features extracted from ViT-GRU",
        "architecture": "ViT (timm) → GRU temporal → Feature extraction → XGBoost",
        "key_features": ["Vision Transformer features", "Temporal GRU", "Gradient boosting"],
        "input_shape": "Features from ViT-GRU model (256x256 input required)",
        "pretrained": True,
        "category": "XGBoost + Pretrained"
    },
    "5j": {
        "name": "XGBoost + ViT-Transformer",
        "model_type": "xgboost_vit_transformer",
        "description": "XGBoost classifier on features extracted from ViT-Transformer",
        "architecture": "ViT (timm) → Transformer temporal → Feature extraction → XGBoost",
        "key_features": ["Vision Transformer features", "Temporal Transformer", "Gradient boosting"],
        "input_shape": "Features from ViT-Transformer model (256x256 input required)",
        "pretrained": True,
        "category": "XGBoost + Pretrained"
    },
    "5k": {
        "name": "ViT-GRU",
        "model_type": "vit_gru",
        "description": "Vision Transformer per frame + GRU temporal modeling",
        "architecture": "ViT-B/16 (timm) → Frame features → GRU → Classification",
        "key_features": ["Pretrained ViT", "GRU temporal", "256x256 input"],
        "input_shape": "(N, T, C, H, W) where H=W=256",
        "pretrained": True,
        "category": "Vision Transformer"
    },
    "5l": {
        "name": "ViT-Transformer",
        "model_type": "vit_transformer",
        "description": "Vision Transformer per frame + Transformer encoder temporal",
        "architecture": "ViT-B/16 (timm) → Frame features → Transformer Encoder → Classification",
        "key_features": ["Pretrained ViT", "Transformer temporal", "256x256 input"],
        "input_shape": "(N, T, C, H, W) where H=W=256",
        "pretrained": True,
        "category": "Vision Transformer"
    },
    "5m": {
        "name": "TimeSformer",
        "model_type": "timesformer",
        "description": "Divided space-time attention for video understanding",
        "architecture": "ViT patch embedding → Divided attention (spatial then temporal) → Classification",
        "key_features": ["Space-time divided attention", "Official implementation", "256x256 input"],
        "input_shape": "(N, T, C, H, W) where H=W=256",
        "pretrained": True,
        "category": "Video Transformer"
    },
    "5n": {
        "name": "ViViT",
        "model_type": "vivit",
        "description": "Video Vision Transformer with tubelet embedding",
        "architecture": "3D tubelet embedding → Transformer encoder → Classification",
        "key_features": ["Tubelet embedding (3D patches)", "ViT blocks", "256x256 input"],
        "input_shape": "(N, C, T, H, W) where H=W=256",
        "pretrained": True,
        "category": "Video Transformer"
    },
    "5o": {
        "name": "I3D",
        "model_type": "i3d",
        "description": "Inflated 3D ConvNet for spatiotemporal video understanding",
        "architecture": "I3D R50 (PyTorchVideo) → Classification head",
        "key_features": ["PyTorchVideo I3D", "Kinetics-400 pretrained", "Spatiotemporal 3D convolutions"],
        "input_shape": "(N, C, T, H, W) where H=W=256",
        "pretrained": True,
        "category": "3D CNN"
    },
    "5p": {
        "name": "R(2+1)D",
        "model_type": "r2plus1d",
        "description": "Factorized 3D convolutions (spatial + temporal separately)",
        "architecture": "R(2+1)D-18 (torchvision) → Classification head",
        "key_features": ["Factorized convolutions", "Kinetics-400 pretrained", "Efficient 3D processing"],
        "input_shape": "(N, C, T, H, W) where H=W=256",
        "pretrained": True,
        "category": "3D CNN"
    },
    "5q": {
        "name": "X3D",
        "model_type": "x3d",
        "description": "Efficient video models with optimized architecture",
        "architecture": "X3D-M (PyTorchVideo/torchvision) → Classification head",
        "key_features": ["Efficient video model", "Kinetics-400 pretrained", "Memory optimized"],
        "input_shape": "(N, C, T, H, W) where H=W=256",
        "pretrained": True,
        "category": "3D CNN"
    },
    "5r": {
        "name": "SlowFast",
        "model_type": "slowfast",
        "description": "Dual pathway architecture: slow (temporal) + fast (spatial)",
        "architecture": "SlowFast R50 (PyTorchVideo/torchvision) → Pathway fusion → Classification",
        "key_features": ["Dual pathway", "Kinetics-400 pretrained", "Temporal + spatial modeling"],
        "input_shape": "(N, C, T, H, W) where H=W=256",
        "pretrained": True,
        "category": "SlowFast"
    },
    "5s": {
        "name": "SlowFast with Attention",
        "model_type": "slowfast_attention",
        "description": "SlowFast with cross-attention between pathways",
        "architecture": "SlowFast base → Attention mechanisms → Classification",
        "key_features": ["Cross-pathway attention", "Self-attention", "Enhanced fusion"],
        "input_shape": "(N, C, T, H, W) where H=W=256",
        "pretrained": True,
        "category": "SlowFast Advanced"
    },
    "5t": {
        "name": "Multi-Scale SlowFast",
        "model_type": "slowfast_multiscale",
        "description": "Multiple temporal sampling rates for multi-scale analysis",
        "architecture": "Multiple SlowFast pathways (different scales) → Fusion → Classification",
        "key_features": ["Multi-scale temporal", "Multiple pathways", "Scale fusion"],
        "input_shape": "(N, C, T, H, W) where H=W=256",
        "pretrained": True,
        "category": "SlowFast Advanced"
    },
    "5u": {
        "name": "Two-Stream",
        "model_type": "two_stream",
        "description": "RGB stream + Optical flow stream with fusion",
        "architecture": "RGB backbone (ResNet/ViT) + Flow backbone → Fusion → Classification",
        "key_features": ["Dual streams", "Optical flow", "Multiple fusion methods"],
        "input_shape": "(N, C, T, H, W) where H=W=256",
        "pretrained": True,
        "category": "Two-Stream"
    }
}

def generate_notebook(model_id: str, config: dict) -> dict:
    """Generate a comprehensive notebook for a model."""
    
    model_type = config["model_type"]
    name = config["name"]
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Model {model_id}: {name}\n",
            "\n",
            f"**{config['description']}**\n",
            "\n",
            "This notebook demonstrates the complete pipeline for training and evaluating this model, including:\n",
            "- Architecture deep-dive with mathematical foundations\n",
            "- Training methodology (5-fold CV, hyperparameter optimization)\n",
            "- Regularization and optimization strategies\n",
            "- MLOps integration (MLflow, DuckDB, Airflow)\n",
            "- Model evaluation with saved checkpoints\n",
            "- Video demonstrations and examples\n",
            "\n",
            "**Category**: {category}  \n".format(category=config['category']),
            "**Pretrained**: {pretrained}  \n".format(pretrained="Yes" if config['pretrained'] else "No"),
            "**Input Shape**: {input_shape}".format(input_shape=config['input_shape'])
        ]
    })
    
    # Setup cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "import json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import polars as pl\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from IPython.display import display, HTML, Video, Image\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Set style\n",
            "plt.style.use('seaborn-v0_8-darkgrid')\n",
            "sns.set_palette('husl')\n",
            "\n",
            "# Add project root\n",
            "project_root = Path().absolute().parent.parent\n",
            "sys.path.insert(0, str(project_root))\n",
            "\n",
            "print(f'📁 Project root: {project_root}')\n",
            "print(f'✅ Imports successful')"
        ]
    })
    
    # Architecture section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Architecture Deep-Dive\n",
            "\n",
            f"### {name} Architecture\n",
            "\n",
            f"**{config['architecture']}**\n",
            "\n",
            "### Key Features\n",
            "\n"
        ] + [f"- **{feature}**\n" for feature in config['key_features']] + [
            "\n",
            "### Implementation Location\n",
            "\n",
            f"- **Model Class**: `lib/training/{model_type.replace('xgboost_', '_xgboost_pretrained' if 'xgboost' in model_type else '')}.py`\n",
            f"- **Factory**: `lib/training/model_factory.py` (create_model function)\n",
            f"- **Training**: `lib/training/pipeline.py` (stage5_train_models function)\n"
        ]
    })
    
    # Check for saved model
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Model Checkpoint Verification\n",
            "\n",
            "Check if trained model exists before demonstrating usage."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# Check for saved model\n",
            "model_dir = project_root / \"data\" / \"stage5\" / \"{model_type}\"\n".format(model_type=model_type),
            "\n",
            "if model_dir.exists():\n",
            "    # Find model checkpoints\n",
            "    checkpoint_files = list(model_dir.glob(\"**/*.pt\")) + list(model_dir.glob(\"**/*.joblib\"))\n",
            "    metrics_files = list(model_dir.glob(\"**/metrics.json\"))\n",
            "    \n",
            "    print(f\"✅ Model directory found: {model_dir}\")\n",
            "    print(f\"   Checkpoints: {len(checkpoint_files)}\")\n",
            "    print(f\"   Metrics files: {len(metrics_files)}\")\n",
            "    \n",
            "    if metrics_files:\n",
            "        with open(metrics_files[0], 'r') as f:\n",
            "            metrics = json.load(f)\n",
            "        \n",
            "        print(f\"\\n📊 Model Performance:\")\n",
            "        print(f\"   Mean F1: {metrics.get('mean_test_f1', 'N/A'):.4f}\" if isinstance(metrics.get('mean_test_f1'), (int, float)) else f\"   Mean F1: {metrics.get('mean_test_f1', 'N/A')}\")\n",
            "        print(f\"   Mean Accuracy: {metrics.get('mean_test_acc', 'N/A'):.4f}\" if isinstance(metrics.get('mean_test_acc'), (int, float)) else f\"   Mean Accuracy: {metrics.get('mean_test_acc', 'N/A')}\")\n",
            "        \n",
            "        model_available = True\n",
            "    else:\n",
            "        print(\"⚠️ No metrics file found\")\n",
            "        model_available = False\n",
            "else:\n",
            "    print(f\"⚠️ Model not trained yet: {model_dir}\")\n",
            "    model_available = False"
        ]
    })
    
    # Training code (commented)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Training Code (Commented)\n",
            "\n",
            "**Note**: This section shows how to train the model. The code is commented out to prevent accidental training.\n",
            "\n",
            "### SLURM Script\n",
            "\n",
            "```bash\n",
            f"# sbatch scripts/slurm_jobs/slurm_stage{model_id}.sh\n",
            "```\n",
            "\n",
            "### Python API\n",
            "\n",
            "```python\n",
            "# from lib.training.pipeline import stage5_train_models\n",
            "# \n",
            "# results = stage5_train_models(\n",
            "#     project_root='.',\n",
            "#     scaled_metadata_path='data/scaled_videos/scaled_metadata.parquet',\n",
            "#     features_stage2_path='data/features_stage2/features_metadata.parquet',\n",
            "#     features_stage4_path='data/features_stage4/features_metadata.parquet',\n",
            "#     model_types=['{model_type}'],\n".format(model_type=model_type),
            "#     n_splits=5,\n",
            "#     num_frames=1000,\n",
            "#     output_dir='data/stage5',\n",
            "#     use_tracking=True,\n",
            "#     use_mlflow=True\n",
            "# )\n",
            "```"
        ]
    })
    
    # Hyperparameters section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Hyperparameter Configuration\n",
            "\n",
            "**Single Hyperparameter Combination** (optimized for efficiency):\n",
            "\n",
            "See `lib/training/grid_search.py` for full configuration.\n",
            "\n",
            "**Rationale for Single Combination**:\n",
            "- Reduced from 5+ combinations to 1 for training efficiency\n",
            "- Hyperparameters selected based on model architecture best practices\n",
            "- Grid search performed on sample, best params applied to full dataset"
        ]
    })
    
    # MLOps section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## MLOps Integration\n",
            "\n",
            "### Experiment Tracking with MLflow\n",
            "\n",
            "**Location**: `lib/mlops/mlflow_tracker.py`\n",
            "\n",
            "**What's Tracked**:\n",
            "- Hyperparameters (learning_rate, batch_size, weight_decay, etc.)\n",
            "- Metrics (train_loss, val_acc, test_f1, precision, recall, AUC-ROC)\n",
            "- Model artifacts (checkpoints, configs, plots)\n",
            "- Run metadata (tags, timestamps, fold numbers, model_type)\n",
            "\n",
            "**Access MLflow UI**:\n",
            "```bash\n",
            "mlflow ui --port 5000\n",
            "# Open http://localhost:5000\n",
            "```\n",
            "\n",
            "### Analytics with DuckDB\n",
            "\n",
            "**Location**: `lib/utils/duckdb_analytics.py`\n",
            "\n",
            "**Fast SQL Queries on Training Results**:\n",
            "```python\n",
            "from lib.utils.duckdb_analytics import DuckDBAnalytics\n",
            "\n",
            "analytics = DuckDBAnalytics()\n",
            "analytics.register_parquet('results', 'data/stage5/{model_type}/metrics.json')\n".format(model_type=model_type),
            "result = analytics.query(\"\"\"\n",
            "    SELECT \n",
            "        fold,\n",
            "        AVG(test_f1) as avg_f1,\n",
            "        STDDEV(test_f1) as std_f1\n",
            "    FROM results\n",
            "    GROUP BY fold\n",
            "\"\"\")\n",
            "```\n",
            "\n",
            "### Airflow Orchestration\n",
            "\n",
            "**Location**: `airflow/dags/fvc_pipeline_dag.py`\n",
            "\n",
            "**Pipeline Stages**:\n",
            "1. Stage 1: Video Augmentation\n",
            "2. Stage 2: Feature Extraction\n",
            "3. Stage 3: Video Scaling\n",
            "4. Stage 4: Scaled Feature Extraction\n",
            "5. Stage 5: Model Training (this model)\n",
            "\n",
            "**Benefits**:\n",
            "- Dependency management (automatic task ordering)\n",
            "- Retry logic (automatic retries on failure)\n",
            "- Monitoring (web UI for pipeline status)\n",
            "- Scheduling (cron-based scheduling support)"
        ]
    })
    
    # Training methodology
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Training Methodology\n",
            "\n",
            "### 5-Fold Stratified Cross-Validation\n",
            "\n",
            "**Why 5-Fold CV?**\n",
            "- **Robust Estimates**: More reliable than single train/test split\n",
            "- **Stratification**: Ensures class balance in each fold\n",
            "- **Group-Aware**: Prevents data leakage (same video ID not in train/val)\n",
            "- **Reproducibility**: Fixed random seed (42)\n",
            "\n",
            "**Evaluation**: Metrics averaged across 5 folds with standard deviation\n",
            "\n",
            "### Regularization Strategy\n",
            "\n",
            "**L2 Regularization (Weight Decay)**:\n",
            "- **Value**: 1e-4 (standard) to 1e-3 (stronger)\n",
            "- **Rationale**: Prevents overfitting, improves generalization\n",
            "- **Implementation**: AdamW optimizer with weight_decay parameter\n",
            "\n",
            "**Dropout**:\n",
            "- **Value**: 0.3-0.5 in classification heads\n",
            "- **Rationale**: Prevents co-adaptation of neurons\n",
            "- **Location**: Fully connected layers before final classification\n",
            "\n",
            "**Batch Normalization**:\n",
            "- **Rationale**: Stabilizes training, enables higher learning rates\n",
            "- **Location**: After convolutional layers\n",
            "\n",
            "**Gradient Clipping**:\n",
            "- **Value**: max_norm=1.0\n",
            "- **Rationale**: Prevents exploding gradients in deep networks\n",
            "\n",
            "**Early Stopping**:\n",
            "- **Patience**: 5 epochs\n",
            "- **Metric**: Validation F1 score\n",
            "- **Rationale**: Prevents overfitting, saves training time\n",
            "\n",
            "### Optimization Strategy\n",
            "\n",
            "**Optimizer**: AdamW\n",
            "- **Learning Rate**: 1e-4 to 5e-4 (model-dependent)\n",
            "- **Betas**: (0.9, 0.999)\n",
            "- **Weight Decay**: 1e-4\n",
            "- **Rationale**: AdamW decouples weight decay from gradient updates\n",
            "\n",
            "**Learning Rate Schedule**:\n",
            "- **Type**: Cosine annealing with warmup\n",
            "- **Warmup Epochs**: 2\n",
            "- **Warmup Factor**: 0.1 (starts at 10% of LR)\n",
            "- **Rationale**: Smooth learning rate decay improves convergence\n",
            "\n",
            "**Differential Learning Rates** (for pretrained models):\n",
            "- **Backbone LR**: 5e-6 (frozen or fine-tuned slowly)\n",
            "- **Head LR**: 5e-4 (new layers trained faster)\n",
            "- **Rationale**: Preserves pretrained features while adapting to new task\n",
            "\n",
            "**Mixed Precision Training (AMP)**:\n",
            "- **Enabled**: Yes (default)\n",
            "- **Benefits**: 2x speedup, 50% memory reduction\n",
            "- **Rationale**: FP16 operations faster on modern GPUs\n",
            "\n",
            "**Gradient Accumulation**:\n",
            "- **Dynamic**: Based on batch size and memory constraints\n",
            "- **Effective Batch Size**: batch_size × gradient_accumulation_steps\n",
            "- **Rationale**: Maintains large effective batch size despite memory constraints\n",
            "\n",
            "### Activation Functions\n",
            "\n",
            "**ReLU**:\n",
            "- **Location**: Convolutional layers\n",
            "- **Rationale**: Standard for CNNs, prevents vanishing gradients\n",
            "\n",
            "**GELU**:\n",
            "- **Location**: Transformer layers\n",
            "- **Rationale**: Smoother gradients than ReLU, better for Transformers\n",
            "\n",
            "**Sigmoid**:\n",
            "- **Location**: Final output (binary classification)\n",
            "- **Rationale**: Maps logits to [0, 1] probability\n",
            "\n",
            "### Data Pipeline\n",
            "\n",
            "**Video Loading**:\n",
            "- **Method**: Frame-by-frame decoding (50x memory reduction)\n",
            "- **Chunked Loading**: Process videos in chunks to avoid OOM\n",
            "- **Caching**: Frame cache for faster subsequent loads\n",
            "\n",
            "**Augmentation**:\n",
            "- **Method**: Pre-generated augmentations (reproducible, fast)\n",
            "- **Types**: Spatial (rotation, flip, color jitter, noise, blur) + Temporal (frame drop, duplicate, reverse)\n",
            "- **Rationale**: Increases dataset diversity, prevents overfitting\n",
            "\n",
            "**Scaling**:\n",
            "- **Target**: 256x256 max dimension (letterboxing preserves aspect ratio)\n",
            "- **Method**: Bilinear interpolation (default) or autoencoder upscaling (optional)\n",
            "- **Rationale**: Consistent input size, memory efficiency\n",
            "\n",
            "**Normalization**:\n",
            "- **Method**: ImageNet statistics or [0, 1] normalization\n",
            "- **Rationale**: Consistent input distribution improves training stability\n",
            "\n",
            "**Frame Sampling**:\n",
            "- **Method**: Uniform sampling across video duration\n",
            "- **Frames**: 1000 frames per video (configurable)\n",
            "- **Rationale**: Captures temporal patterns across entire video"
        ]
    })
    
    # Feature engineering (for XGBoost models)
    if "xgboost" in model_type:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Feature Extraction Strategy\n",
                "\n",
                f"**Base Model**: {config['name'].replace('XGBoost + ', '')}\n",
                "\n",
                "**Feature Extraction Process**:\n",
                "1. Load pretrained base model\n",
                "2. Remove classification head\n",
                "3. Extract features from intermediate layers\n",
                "4. Aggregate features (mean pooling over temporal dimension)\n",
                "5. Cache features for XGBoost training\n",
                "\n",
                "**Location**: `lib/training/_xgboost_pretrained.py`\n",
                "\n",
                "**Benefits**:\n",
                "- Leverages pretrained representations\n",
                "- Fast XGBoost training on extracted features\n",
                "- Interpretable feature importance\n",
                "- Efficient inference"
            ]
        })
    
    # Video demonstration
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Video Demonstration\n",
            "\n",
            "Load and display sample videos for model evaluation."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load sample videos for demonstration\n",
            "scaled_metadata = project_root / \"data\" / \"scaled_videos\" / \"scaled_metadata.parquet\"\n",
            "\n",
            "if scaled_metadata.exists():\n",
            "    from lib.utils.paths import load_metadata_flexible\n",
            "    \n",
            "    df = load_metadata_flexible(str(scaled_metadata))\n",
            "    \n",
            "    if df is not None and df.height > 0:\n",
            "        # Sample real and fake videos\n",
            "        real_videos = df.filter(pl.col('label') == 'real').head(2)\n",
            "        fake_videos = df.filter(pl.col('label') == 'fake').head(2)\n",
            "        \n",
            "        print(f\"📹 Sample Videos:\")\n",
            "        print(f\"   Real videos: {real_videos.height}\")\n",
            "        print(f\"   Fake videos: {fake_videos.height}\")\n",
            "        \n",
            "        # Display video paths (actual video display requires video files)\n",
            "        if real_videos.height > 0:\n",
            "            real_path = real_videos['video_path'][0]\n",
            "            print(f\"\\n✅ Real video: {Path(real_path).name}\")\n",
            "            # Video(real_path, width=400)  # Uncomment if video file exists\n",
            "        \n",
            "        if fake_videos.height > 0:\n",
            "            fake_path = fake_videos['video_path'][0]\n",
            "            print(f\"\\n❌ Fake video: {Path(fake_path).name}\")\n",
            "            # Video(fake_path, width=400)  # Uncomment if video file exists\n",
            "    else:\n",
            "        print(\"⚠️ No videos found in metadata\")\n",
            "else:\n",
            "    print(\"⚠️ Scaled videos metadata not found\")"
        ]
    })
    
    # Model inference example
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Model Inference Example\n",
            "\n",
            "Load saved model and perform inference on sample videos."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load model and perform inference\n",
            "# Check if model checkpoint exists\n",
            "checkpoint_files = list(model_dir.glob(\"**/*.pt\")) + list(model_dir.glob(\"**/*.joblib\")) if model_dir.exists() else []\n",
            "\n",
            "if len(checkpoint_files) > 0:\n",
                "    import torch\n",
                "    from lib.training.model_factory import create_model\n",
                "    from lib.mlops.config import RunConfig\n",
                "    \n",
                "    # Create model\n",
                "    config = RunConfig(\n",
                "        run_id='demo',\n",
                "        experiment_name='demo',\n",
                "        model_type='{model_type}',\n".format(model_type=model_type),
                "        num_frames=1000\n",
                "    )\n",
                "    \n",
                "    model = create_model('{model_type}', config)\n".format(model_type=model_type),
                "    \n",
                "    # Load checkpoint\n",
                "    checkpoint_path = checkpoint_files[0]\n",
                "    \n",
                "    if checkpoint_path.suffix == '.pt':\n",
                "        checkpoint = torch.load(checkpoint_path, map_location='cpu')\n",
                "        if 'model_state_dict' in checkpoint:\n",
                "            model.load_state_dict(checkpoint['model_state_dict'])\n",
                "        else:\n",
                "            model.load_state_dict(checkpoint)\n",
                "        print(f\"✅ Loaded PyTorch checkpoint: {checkpoint_path.name}\")\n",
                "    else:\n",
                "        import joblib\n",
                "        model = joblib.load(checkpoint_path)\n",
                "        print(f\"✅ Loaded sklearn/XGBoost model: {checkpoint_path.name}\")\n",
                "    \n",
            "        model.eval()\n",
            "        print(f\"\\n📊 Model loaded and ready for inference\")\n",
            "        print(f\"   Model type: {type(model).__name__}\")\n",
            "    except Exception as e:\n",
            "        print(f\"⚠️ Error loading model: {e}\")\n",
            "else:\n",
            "    print(\"⚠️ Model checkpoint not available for inference\")"
            ]
        })
    
    # Results visualization
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Results Visualization\n",
            "\n",
            "Visualize training results and metrics."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load and visualize metrics\n",
            "if model_available and metrics_files:\n",
            "    with open(metrics_files[0], 'r') as f:\n",
            "        metrics = json.load(f)\n",
            "    \n",
            "    # Create visualization\n",
            "    fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "    \n",
            "    # F1 Score across folds\n",
            "    if 'fold_results' in metrics:\n",
            "        fold_f1s = [fold.get('test_f1', 0) for fold in metrics['fold_results']]\n",
            "        axes[0, 0].bar(range(1, len(fold_f1s)+1), fold_f1s, color='#4CAF50')\n",
            "        axes[0, 0].axhline(metrics.get('mean_test_f1', 0), color='red', linestyle='--', label='Mean')\n",
            "        axes[0, 0].set_title('F1 Score by Fold', fontsize=12, fontweight='bold')\n",
            "        axes[0, 0].set_xlabel('Fold')\n",
            "        axes[0, 0].set_ylabel('F1 Score')\n",
            "        axes[0, 0].legend()\n",
            "        axes[0, 0].grid(axis='y', alpha=0.3)\n",
            "    \n",
            "    # Accuracy across folds\n",
            "    if 'fold_results' in metrics:\n",
            "        fold_accs = [fold.get('test_acc', 0) for fold in metrics['fold_results']]\n",
            "        axes[0, 1].bar(range(1, len(fold_accs)+1), fold_accs, color='#2196F3')\n",
            "        axes[0, 1].axhline(metrics.get('mean_test_acc', 0), color='red', linestyle='--', label='Mean')\n",
            "        axes[0, 1].set_title('Accuracy by Fold', fontsize=12, fontweight='bold')\n",
            "        axes[0, 1].set_xlabel('Fold')\n",
            "        axes[0, 1].set_ylabel('Accuracy')\n",
            "        axes[0, 1].legend()\n",
            "        axes[0, 1].grid(axis='y', alpha=0.3)\n",
            "    \n",
            "    # Metrics summary\n",
            "    metrics_summary = {\n",
            "        'F1 Score': metrics.get('mean_test_f1', 0),\n",
            "        'Accuracy': metrics.get('mean_test_acc', 0),\n",
            "        'Precision': metrics.get('mean_test_precision', 0),\n",
            "        'Recall': metrics.get('mean_test_recall', 0)\n",
            "    }\n",
            "    \n",
            "    axes[1, 0].bar(metrics_summary.keys(), metrics_summary.values(), color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])\n",
            "    axes[1, 0].set_title('Average Metrics', fontsize=12, fontweight='bold')\n",
            "    axes[1, 0].set_ylabel('Score')\n",
            "    axes[1, 0].grid(axis='y', alpha=0.3)\n",
            "    axes[1, 0].set_ylim([0, 1])\n",
            "    \n",
            "    # Confusion matrix (if available)\n",
            "    if 'confusion_matrix' in metrics:\n",
            "        cm = np.array(metrics['confusion_matrix'])\n",
            "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])\n",
            "        axes[1, 1].set_title('Confusion Matrix', fontsize=12, fontweight='bold')\n",
            "        axes[1, 1].set_xlabel('Predicted')\n",
            "        axes[1, 1].set_ylabel('Actual')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    # Print summary\n",
            "    print(\"\\n📊 Performance Summary:\")\n",
            "    for key, value in metrics_summary.items():\n",
            "        if isinstance(value, (int, float)):\n",
            "            print(f\"   {key}: {value:.4f}\")\n",
            "else:\n",
            "    print(\"⚠️ Metrics not available for visualization\")"
        ]
    })
    
    # Conclusion
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusion\n",
            "\n",
            f"This notebook demonstrated the {name} model for deepfake video detection, including:\n",
            "\n",
            "- ✅ **Architecture**: {architecture}\n".format(architecture=config['architecture']),
            "- ✅ **Training Methodology**: 5-fold CV, hyperparameter optimization, regularization\n",
            "- ✅ **MLOps Integration**: MLflow tracking, DuckDB analytics, Airflow orchestration\n",
            "- ✅ **Evaluation**: Model checkpoint verification and inference examples\n",
            "\n",
            "**Next Steps**:\n",
            "1. Compare with other models (see other notebooks 5a-5u)\n",
            "2. Explore MLflow UI for detailed experiment tracking\n",
            "3. Use DuckDB for custom analytics queries\n",
            "4. Deploy best model to production"
        ]
    })
    
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def main():
    """Generate all model notebooks."""
    notebooks_dir = Path(__file__).parent
    
    for model_id, config in MODEL_CONFIGS.items():
        notebook = generate_notebook(model_id, config)
        notebook_path = notebooks_dir / f"{model_id}_{config['model_type']}.ipynb"
        
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"✅ Generated: {notebook_path.name}")
    
    print(f"\n✅ Generated {len(MODEL_CONFIGS)} model notebooks")

if __name__ == "__main__":
    main()
