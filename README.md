# AURA - Authenticity Understanding of Real vs Artificial Shortform Videos

A deep learning project for binary classification of videos (real vs. fake) using 3D Convolutional Neural Networks with transfer learning.

## Overview

AURA (Authenticity Understanding of Real vs Artificial) is a comprehensive system for detecting fake videos in shortform content. The project implements multiple model architectures for binary video classification using the Fake Video Corpus dataset. The system uses pretrained 3D ResNet backbones (Kinetics-400 weights) with custom classification heads, along with handcrafted feature extraction and ensemble methods.

## Key Features

- **Multiple Model Architectures**: 14 models evaluated including baseline (Logistic Regression, SVM), gradient boosting (XGBoost), and ensemble models (XGBoost + pretrained features)
- **Pretrained Feature Extraction**: Uses Inception-v3, I3D, R(2+1)D, and ViT backbones for transfer learning
- **Handcrafted Features**: 26 engineered features capturing compression artifacts, temporal consistency, and codec parameters
- **Comprehensive Augmentations**: 11× augmentation pipeline with compression, temporal, spatial, and noise perturbations
- **K-Fold Cross-Validation**: Stratified 5-fold CV for robust evaluation with detailed fold-wise analysis
- **MLOps Infrastructure**: MLflow experiment tracking, checkpointing, and data versioning
- **Memory Optimized**: Aggressive GC, OOM handling, progressive batch size fallback
- **Class Imbalance Handling**: Balanced batch sampling and inverse-frequency weights
- **Enhanced Reporting**: Detailed LaTeX reports with comprehensive visualizations and model-specific conclusions

## Project Structure

```
aura/
├── archive/                    # Original dataset archives
├── data/                       # Processed metadata
├── docs/                       # Documentation
├── lib/                        # Core library modules
├── src/                        # Scripts and notebooks
├── videos/                     # Extracted video files
├── runs/                       # Experiment outputs
├── models/                     # Saved models
└── logs/                       # Log files
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Dataset

```bash
python3 src/setup_fvc_dataset.py
```

### 3. Run Training

**Option A: Using 5-Stage Pipeline (Recommended)**
```bash
python3 src/run_new_pipeline.py
```

**Option B: Using Individual Stage Scripts**
```bash
# Run stages individually
python3 src/scripts/run_stage1_augmentation.py
python3 src/scripts/run_stage2_features.py
python3 src/scripts/run_stage3_scaling.py
python3 src/scripts/run_stage4_scaled_features.py
python3 src/scripts/run_stage5_training.py
```

See [src/scripts/README.md](src/scripts/README.md) for detailed usage.

**Option D: Using SLURM (Cluster)**
```bash
sbatch src/scripts/run_fvc_training.sh
```

### 4. View Results Dashboard

After training completes, launch the interactive results dashboard:

```bash
streamlit run src/dashboard_results.py
```

The enhanced dashboard provides:
- **Model performance comparisons** with confidence intervals
- **K-fold cross-validation analysis** with violin plots
- **ROC curves and Precision-Recall curves**
- **Training curves** (loss/accuracy over epochs)
- **Statistical significance testing** (pairwise t-tests)
- **Comprehensive dataset statistics**
- **Publication-ready visualizations**

See [Dashboard Usage Guide](docs/DASHBOARD_USAGE.md) for details.

### 5. Generate Paper Figures

Generate publication-ready figures for IEEE paper submission:

```bash
python src/generate_paper_figures.py
```

This creates high-quality figures in PNG, PDF, and SVG formats with:
- 300 DPI resolution
- Academic font styling (Times New Roman)
- Proper labels and legends
- Statistical significance analysis
- LaTeX-ready tables

See [Paper Figure Generation Guide](docs/PAPER_FIGURE_GENERATION.md) for details.

### 6. View Analysis Reports

Comprehensive LaTeX reports are available in the `report/` directory:

- **[Final Report](report/final_report.tex)** - Complete analysis with detailed visualizations including:
  - ROC and Precision-Recall curves with comprehensive descriptions
  - Cross-validation stability analysis across all model families
  - Confusion matrices and calibration curves
  - Per-class metrics and error analysis
  - Hyperparameter sensitivity analysis
  - Statistical significance testing
  - Ablation studies and overfitting analysis

- **[Submission Report](report/submission_report.tex)** - Concise version for paper submission with:
  - Key findings and performance metrics
  - Essential visualizations (ROC/PR curves, CV comparisons, confusion matrices)
  - Calibration analysis and per-class performance
  - Platform transformation robustness analysis

Both reports include:
- ✅ Detailed figure captions explaining all visualizations
- ✅ Comprehensive performance comparisons across 14 model architectures
- ✅ Statistical analysis and significance testing
- ✅ Model-specific conclusions and recommendations
- ✅ Publication-ready formatting

Compile the reports using:
```bash
cd report
pdflatex final_report.tex
pdflatex submission_report.tex
```

## Documentation

### Core Documentation
- [**End-to-End Workflow**](docs/WORKFLOW.md) - Complete pipeline from archive to notebook generation
- [**Project Overview**](docs/PROJECT_OVERVIEW.md) - Comprehensive project documentation, architecture, and implementation details
- [**ML Methodology**](docs/METHODOLOGY.md) - Critical ML methodology fixes, best practices, and validation checks
- [**Optimizations**](docs/OPTIMIZATIONS.md) - Memory optimizations, MLOps improvements, and pipeline enhancements
- [**Production Guardrails**](docs/PRODUCTION_GUARDRAILS.md) - Comprehensive guardrail system for production reliability

### Feature Guides
- [**Autoencoder Scaling**](docs/AUTOENCODER_SCALING.md) - Guide for using Hugging Face autoencoders for video scaling
- [**Dashboard Usage**](docs/DASHBOARD_USAGE.md) - Interactive Streamlit results dashboard guide
- [**Paper Figure Generation**](docs/PAPER_FIGURE_GENERATION.md) - Generate publication-ready figures for IEEE papers

### Reference
- [**Architecture Diagrams**](presentation/ARCHITECTURE_DIAGRAM.md) - ASCII diagrams of pipeline architecture and data flow
- [**Changelog**](docs/CHANGELOG.md) - Project version history and notable changes
- [**Git Setup**](docs/GIT_SETUP.md) - Git repository information and setup instructions

### Component Documentation
- [**Scripts README**](src/scripts/README.md) - Individual stage scripts and SLURM batch scripts
- [**Notebooks README**](src/notebooks/README.md) - Jupyter notebook collection and usage
- [**Test README**](test/README.md) - Unit test suite documentation

### Executed Model Analysis Notebooks

The following notebooks contain complete analysis results with training curves, validation metrics, and performance visualizations:

- **[5a: Logistic Regression](src/notebooks/executed/5a_logistic_regression.ipynb)** - Logistic Regression baseline model analysis
- **[5alpha: Scikit-learn Logistic Regression](src/notebooks/executed/5alpha_sklearn_logreg.ipynb)** - Scikit-learn Logistic Regression analysis
- **[5b: Support Vector Machine](src/notebooks/executed/5b_svm.ipynb)** - SVM model analysis with fold-wise metrics
- **[5beta: Gradient Boosting](src/notebooks/executed/5beta_gradient_boosting.ipynb)** - Gradient Boosting (XGBoost, LightGBM, CatBoost) analysis
- **[5f: XGBoost + Pretrained Inception](src/notebooks/executed/5f_xgboost_pretrained_inception.ipynb)** - XGBoost with Pretrained Inception features
- **[5g: XGBoost + I3D](src/notebooks/executed/5g_xgboost_i3d.ipynb)** - XGBoost with I3D features
- **[5h: XGBoost + R(2+1)D](src/notebooks/executed/5h_xgboost_r2plus1d.ipynb)** - XGBoost with R(2+1)D features

Each notebook includes:
- ✅ Complete training curves (loss, accuracy, F1 score)
- ✅ Validation metrics across cross-validation folds
- ✅ MLflow experiment tracking integration
- ✅ DuckDB analytics queries
- ✅ ROC and Precision-Recall curves
- ✅ Comprehensive performance summaries with model-specific conclusions
- ✅ Detailed markdown introductions and data-driven analysis
- ✅ Training/validation loss plots for all models

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- `torch`, `torchvision`: Deep learning framework
- `polars`: Fast DataFrame operations
- `opencv-python`, `av`, `torchcodec`: Video decoding
- `papermill`, `ipykernel`: Notebook execution

## Model Architecture

- **Backbone**: 3D ResNet-18 (r3d_18) pretrained on Kinetics-400
- **Head**: Inception3DBlock → AdaptiveAvgPool3d → Dropout(0.5) → Linear(512, 1)
- **Input**: (N, C=3, T=16, H=224, W=224)
- **Output**: Binary logits (N, 1)

## Training Configuration

- **Frames per video**: 16 (uniformly sampled)
- **Input size**: 224x224 (fixed with letterboxing)
- **Batch size**: 32 (with progressive fallback: 16 → 8 → 4 → 2 → 1)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Mixed Precision**: Enabled (AMP)
- **Cross-Validation**: 5-fold stratified

## Results

Results are saved in multiple locations:

### Training Results
- **`data/stage5/`**: Model outputs organized by model type with:
  - Fold-wise metrics and visualizations (confusion matrices, ROC/PR curves, calibration plots)
  - Cross-validation comparison plots
  - Hyperparameter search results
  - Best model checkpoints and metadata

### Analysis Reports
- **`report/`**: LaTeX reports with comprehensive analysis:
  - `final_report.tex` - Complete detailed analysis
  - `submission_report.tex` - Concise submission version
  - Both include enhanced visualizations and detailed figure captions

### Experiment Tracking
- **`mlruns/`**: MLflow experiment tracking with:
  - Experiment configurations and hyperparameters
  - Metrics logs across all folds
  - Model artifacts and checkpoints
  - System metadata and git commit tracking

### Analysis Notebooks
- **`src/notebooks/executed/`**: Executed Jupyter notebooks with:
  - Complete analysis for each model
  - Training curves and validation metrics
  - Model-specific conclusions and recommendations

## Dataset Credits

This project uses the **Fake Video Corpus (FVC)** dataset provided by [MKLab-ITI](https://github.com/MKLab-ITI/fake-video-corpus). We gratefully acknowledge:

- **Dataset Source**: [fake-video-corpus](https://github.com/MKLab-ITI/fake-video-corpus)
- **Dataset Authors**: Olga Papadopoulou, Markos Zampoglou, Symeon Papadopoulos, and Ioannis Kompatsiaris
- **Citation**: 
  ```bibtex
  @article{papadopoulou2018corpus,
    author = "Papadopoulou, Olga and Zampoglou, Markos and Papadopoulos, Symeon and Kompatsiaris, Ioannis",
    title = "A corpus of debunked and verified user-generated videos",
    journal = "Online Information Review",
    doi = "10.1108/OIR-03-2018-0101",
    year={2018},
    publisher={Emerald Publishing Limited}
  }
  ```
- **License**: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Funding**: Supported by the InVID project, funded by the European Commission under contract number 687786

For questions about the dataset, please contact Olga Papadopoulou at olgapapa@iti.gr.

## License

This project is part of SI670 Applied Machine Learning course.

## Author

Horopter (santoshdesai12@hotmail.com)

