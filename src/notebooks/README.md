# FVC Deepfake Detection: Comprehensive Notebook Collection

This directory contains **presentation-quality notebooks** demonstrating the complete machine learning pipeline for deepfake video detection, from raw ZIP archives to production-ready models.

## Notebook Structure

### Master Pipeline Journey
- **`00_MASTER_PIPELINE_JOURNEY.ipynb`**: Complete end-to-end pipeline demonstration
  - Infrastructure & requirements
  - Data extraction & exploration
  - All 5 pipeline stages with technical rationale
  - MLOps integration (MLflow, DuckDB, Airflow)
  - Feature preprocessing (imputation, scaling, normalization)
  - Model evaluation & results

### Individual Model Notebooks (5c-5u)

Each model notebook includes:
- ✅ **Architecture Deep-Dive**: Mathematical foundations, implementation details
- ✅ **Training Methodology**: 5-fold CV, hyperparameter optimization, regularization
- ✅ **MLOps Integration**: MLflow tracking, DuckDB analytics, Airflow orchestration
- ✅ **Model Checkpoint Verification**: Checks for saved models before demonstration
- ✅ **Video Demonstrations**: Sample video loading and display
- ✅ **Commented Training Code**: Shows how to train (commented to prevent accidental training)
- ✅ **Results Visualization**: Performance metrics, confusion matrices, fold analysis

**Model Notebooks**:
- **5c**: `5c_naive_cnn.ipynb` - Naive CNN baseline
- **5d**: `5d_pretrained_inception.ipynb` - Pretrained Inception (R3D-18)
- **5e**: `5e_variable_ar_cnn.ipynb` - Variable AR CNN
- **5f**: `5f_xgboost_pretrained_inception.ipynb` - XGBoost + Pretrained Inception
- **5g**: `5g_xgboost_i3d.ipynb` - XGBoost + I3D
- **5h**: `5h_xgboost_r2plus1d.ipynb` - XGBoost + R(2+1)D
- **5i**: `5i_xgboost_vit_gru.ipynb` - XGBoost + ViT-GRU
- **5j**: `5j_xgboost_vit_transformer.ipynb` - XGBoost + ViT-Transformer
- **5k**: `5k_vit_gru.ipynb` - ViT-GRU
- **5l**: `5l_vit_transformer.ipynb` - ViT-Transformer
- **5m**: `5m_timesformer.ipynb` - TimeSformer
- **5n**: `5n_vivit.ipynb` - ViViT
- **5o**: `5o_i3d.ipynb` - I3D
- **5p**: `5p_r2plus1d.ipynb` - R(2+1)D
- **5q**: `5q_x3d.ipynb` - X3D
- **5r**: `5r_slowfast.ipynb` - SlowFast
- **5s**: `5s_slowfast_attention.ipynb` - SlowFast with Attention
- **5t**: `5t_slowfast_multiscale.ipynb` - Multi-Scale SlowFast
- **5u**: `5u_two_stream.ipynb` - Two-Stream

## Key Features

### Comprehensive Technical Coverage

**Data Engineering**:
- ZIP archive extraction and validation
- Data exploration with statistical analysis
- Video metadata extraction (duration, fps, resolution, codec)

**Feature Engineering**:
- Handcrafted features (noise residual, DCT, blur/sharpness, codec cues)
- Feature preprocessing (imputation, scaling, normalization, collinearity removal)
- Rationale for each feature type

**Model Architecture**:
- 23 diverse models from baselines to state-of-the-art
- Detailed architecture explanations
- Implementation locations and code references

**Training Methodology**:
- 5-fold stratified cross-validation (with rationale)
- Hyperparameter optimization (grid search on sample)
- Regularization (L2, dropout, batch norm, gradient clipping)
- Optimization (AdamW, cosine annealing, mixed precision, gradient accumulation)
- Activation functions (ReLU, GELU, Sigmoid)

**MLOps Infrastructure**:
- **MLflow**: Experiment tracking, model registry, artifact management
- **DuckDB**: Fast SQL queries on training results
- **Airflow**: Pipeline orchestration with dependency management
- **Custom MLOps**: ExperimentTracker, CheckpointManager, RunConfig

### Production-Grade Practices

- ✅ Error handling and validation
- ✅ Checkpointing and resume capability
- ✅ Reproducibility (fixed seeds, deterministic operations)
- ✅ Memory optimization (chunked processing, frame-by-frame decoding)
- ✅ Model verification (ensures proper implementations, no fallbacks)

## Usage

### Viewing Notebooks

```bash
# Start Jupyter
jupyter notebook src/notebooks/

# Or JupyterLab
jupyter lab src/notebooks/
```

### Running Demonstrations

1. **Start with Master Pipeline**: `00_MASTER_PIPELINE_JOURNEY.ipynb`
   - Complete overview of the entire pipeline
   - Technical rationale for all decisions
   - Infrastructure integration

2. **Explore Individual Models**: Open any `5*.ipynb` notebook
   - Architecture details
   - Training methodology
   - MLOps integration
   - Model checkpoint verification

### Training Models

**Note**: Notebooks show training code but it's commented out to prevent accidental training.

To actually train models, use:
```bash
# SLURM scripts
sbatch scripts/slurm_jobs/slurm_stage5c.sh  # For model 5c, etc.

# Or Python API (see commented code in notebooks)
```

## Technical Highlights

### Why These Notebooks Are Production-Quality

1. **Complete Journey**: From ZIP files → trained models → deployment
2. **Technical Depth**: Explains WHY, not just WHAT
3. **MLOps Integration**: MLflow, Airflow, DuckDB prominently featured
4. **Best Practices**: 5-fold CV, hyperparameter optimization, regularization
5. **Infrastructure**: Shows use of modern ML infrastructure
6. **Reproducibility**: Fixed seeds, deterministic operations
7. **Error Handling**: Robust validation and error messages
8. **Model Verification**: Ensures proper implementations (no fallbacks)

### What Makes This Professional

- **Mathematical Foundations**: Explains formulas and theory
- **Design Rationale**: Why each decision was made
- **Trade-off Analysis**: Pros/cons of different approaches
- **Industry Standards**: Follows ML best practices
- **Infrastructure Integration**: MLOps tools properly showcased
- **Production Considerations**: Deployment, monitoring, scalability

## Notebook Generation

Notebooks are generated using `generate_model_notebooks.py`:

```bash
python3 src/notebooks/generate_model_notebooks.py
```

This ensures consistency across all model notebooks and makes updates easy.

## Requirements

All notebooks require:
- Python 3.10+
- Jupyter/IPython
- Project dependencies (see `requirements.txt`)
- Trained models (for demonstration sections)

## Next Steps

1. **Review Master Pipeline**: Understand the complete journey
2. **Explore Model Notebooks**: Deep-dive into specific architectures
3. **Check MLflow UI**: Compare experiments across models
4. **Use DuckDB**: Run custom analytics queries
5. **Deploy Best Model**: Use MLflow Model Registry for production

---

**Target Audience**: ML Engineers, Research Scientists, Hiring Managers  
**Level**: Production-Grade, Research-Quality Implementation
