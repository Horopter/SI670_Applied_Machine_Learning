# Stage 5 Models: Comprehensive End-to-End Analysis

## Overview

This document provides a complete analysis of all Stage 5 models, covering the entire pipeline from SLURM job submission to final metrics, plots, and analytics integration.

## Table of Contents

1. [Model Inventory](#model-inventory)
2. [SLURM Job Structure](#slurm-job-structure)
3. [Python Training Pipeline](#python-training-pipeline)
4. [Metrics & Tracking](#metrics--tracking)
5. [MLflow Integration](#mlflow-integration)
6. [DuckDB Analytics](#duckdb-analytics)
7. [Airflow Orchestration](#airflow-orchestration)
8. [Plots & Visualizations](#plots--visualizations)
9. [Time Logs & Performance](#time-logs--performance)
10. [Data Flow & Connections](#data-flow--connections)

---

## Model Inventory

### All Stage 5 Models (21 Total)

#### Baseline Models (5a-5b)
1. **5a: logistic_regression** (`slurm_stage5a.sh`)
   - Type: Feature-based (sklearn)
   - Location: `data/stage5/logistic_regression/`
   - Status: ✅ Trained (5 folds + best_model)

2. **5b: svm** (`slurm_stage5b.sh`)
   - Type: Feature-based (sklearn)
   - Location: `data/stage5/svm/`
   - Status: ✅ Trained (5 folds + best_model)

#### PyTorch CNN Models (5c-5e)
3. **5c: naive_cnn** (`slurm_stage5c.sh`)
   - Type: Video-based (PyTorch)
   - Location: `data/stage5/naive_cnn/`
   - Status: ⚠️ Partial (fold_1 only)

4. **5d: pretrained_inception** (`slurm_stage5d.sh`)
   - Type: Video-based (PyTorch)
   - Location: `data/stage5/pretrained_inception/`
   - Status: ✅ Trained (5 folds + best_model)

5. **5e: variable_ar_cnn** (`slurm_stage5e.sh`)
   - Type: Video-based (PyTorch)
   - Location: `data/stage5/variable_ar_cnn/`
   - Status: ⚠️ No folds found

#### XGBoost Models (5f-5j)
6. **5f: xgboost_pretrained_inception** (`slurm_stage5f.sh`)
   - Type: Feature-based (XGBoost on pretrained features)
   - Location: `data/stage5/xgboost_pretrained_inception/`
   - Status: ✅ Trained (5 folds + best_model)

7. **5g: xgboost_i3d** (`slurm_stage5g.sh`)
   - Type: Feature-based (XGBoost on I3D features)
   - Location: `data/stage5/xgboost_i3d/`
   - Status: ✅ Trained (5 folds + best_model)

8. **5h: xgboost_r2plus1d** (`slurm_stage5h.sh`)
   - Type: Feature-based (XGBoost on R2+1D features)
   - Location: `data/stage5/xgboost_r2plus1d/`
   - Status: ✅ Trained (5 folds + best_model)

9. **5i: xgboost_vit_gru** (`slurm_stage5i.sh`)
   - Type: Feature-based (XGBoost on ViT-GRU features)
   - Location: `data/stage5/xgboost_vit_gru/`
   - Status: ✅ Trained (5 folds + best_model)

10. **5j: xgboost_vit_transformer** (`slurm_stage5j.sh`)
    - Type: Feature-based (XGBoost on ViT-Transformer features)
    - Location: `data/stage5/xgboost_vit_transformer/`
    - Status: ✅ Trained (5 folds)

#### Frame-Temporal Models (5k-5l)
11. **5k: vit_gru** (`slurm_stage5k.sh`)
    - Type: Video-based (PyTorch - ViT per frame + GRU)
    - Location: `data/stage5/vit_gru/`
    - Status: ⚠️ No folds found

12. **5l: vit_transformer** (`slurm_stage5l.sh`)
    - Type: Video-based (PyTorch - ViT per frame + Transformer)
    - Location: `data/stage5/vit_transformer/`
    - Status: ⚠️ No folds found

#### Video Transformers (5m-5n)
13. **5m: timesformer** (`slurm_stage5m.sh`)
    - Type: Video-based (PyTorch - Full video attention)
    - Location: `data/stage5/timesformer/`
    - Status: ⚠️ Not found in directory

14. **5n: vivit** (`slurm_stage5n.sh`)
    - Type: Video-based (PyTorch - Video Vision Transformer)
    - Location: `data/stage5/vivit/`
    - Status: ⚠️ Not found in directory

#### Spatiotemporal 3D Models (5o-5q)
15. **5o: i3d** (`slurm_stage5o.sh`)
    - Type: Video-based (PyTorch - 3D CNN)
    - Location: `data/stage5/i3d/`
    - Status: ⚠️ No folds found

16. **5p: r2plus1d** (`slurm_stage5p.sh`)
    - Type: Video-based (PyTorch - 3D CNN)
    - Location: `data/stage5/r2plus1d/`
    - Status: ⚠️ No folds found

17. **5q: x3d** (`slurm_stage5q.sh`)
    - Type: Video-based (PyTorch - 3D CNN)
    - Location: `data/stage5/x3d/`
    - Status: ✅ Trained (5 folds + best_model)

#### Advanced SlowFast Variants (5r-5t)
18. **5r: slowfast** (`slurm_stage5r.sh`)
    - Type: Video-based (PyTorch - Dual pathway)
    - Location: `data/stage5/slowfast/`
    - Status: ✅ Trained (5 folds + best_model)

19. **5s: slowfast_attention** (`slurm_stage5s.sh`)
    - Type: Video-based (PyTorch - SlowFast with attention)
    - Location: `data/stage5/slowfast_attention/`
    - Status: ⚠️ Not found in directory

20. **5t: slowfast_multiscale** (`slurm_stage5t.sh`)
    - Type: Video-based (PyTorch - SlowFast multiscale)
    - Location: `data/stage5/slowfast_multiscale/`
    - Status: ⚠️ Not found in directory

#### Two-Stream Models (5u)
21. **5u: two_stream** (`slurm_stage5u.sh`)
    - Type: Video-based (PyTorch - Dual streams + optical flow)
    - Location: `data/stage5/two_stream/`
    - Status: ⚠️ Not found in directory

---

## SLURM Job Structure

### Coordinator Script
**File**: `scripts/slurm_jobs/slurm_stage5_training.sh`

**Purpose**: Submits all 21 individual model training jobs

**Key Features**:
- Submits jobs for all models (5a-5u)
- Passes environment variables to child jobs
- Tracks submission status
- Provides monitoring commands

**Environment Variables**:
- `FVC_NUM_FRAMES`: Number of frames per video (default: 8)
- `FVC_N_SPLITS`: Number of k-fold splits (default: 5)
- `FVC_STAGE5_OUTPUT_DIR`: Output directory (default: data/training_results)
- `FVC_USE_TRACKING`: Enable/disable tracking (default: true)
- `FVC_DELETE_EXISTING`: Delete existing models (default: 0)
- `FVC_STAGE3_OUTPUT_DIR`: Stage 3 output directory
- `FVC_STAGE2_OUTPUT_DIR`: Stage 2 output directory
- `FVC_STAGE4_OUTPUT_DIR`: Stage 4 output directory

### Individual Model Scripts

**Template Structure** (example: `slurm_stage5a.sh`):

```bash
#SBATCH --job-name=fvc_stage5a
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/stage5/stage5a-%j.out
#SBATCH --error=logs/stage5/stage5a-%j.err
```

**Key Components**:
1. **Environment Setup**: Python, CUDA, virtual environment
2. **Prerequisites Check**: Python packages, Stage 2/3/4 outputs
3. **Import Validation**: `validate_stage5_imports.py`
4. **Feature Sanity Check**: `sanity_check_features.py`
5. **Training Execution**: Calls `src/scripts/run_stage5_training.py`
6. **Time Logging**: Tracks start/end times, calculates duration

**Time Logging**:
```bash
STAGE5_START=$(date +%s)
# ... training execution ...
STAGE5_END=$(date +%s)
STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
log "✓ Stage 5 ($MODEL_TYPE) completed successfully in ${STAGE5_DURATION}s ($((${STAGE5_DURATION} / 60)) minutes)"
```

**Log Files**:
- `logs/stage5/stage5a-{JOB_ID}.out`: Standard output
- `logs/stage5/stage5a-{JOB_ID}.err`: Standard error
- `logs/stage5/stage5a_{JOB_ID}.log`: Detailed training log

---

## Python Training Pipeline

### Entry Point
**File**: `src/scripts/run_stage5_training.py`

**Function**: `main()`

**Key Features**:
- Argument parsing for all training parameters
- Path validation and setup
- Memory statistics logging
- Time tracking
- Error handling with crash diagnostics

**Time Tracking**:
```python
stage_start = time.time()
# ... training ...
stage_duration = time.time() - stage_start
logger.info("Execution time: %.2f seconds (%.2f minutes)", 
           stage_duration, stage_duration / 60)
```

### Core Training Function
**File**: `lib/training/pipeline.py`

**Function**: `stage5_train_models()`

**Parameters**:
- `project_root`: Project root directory
- `scaled_metadata_path`: Path to scaled metadata (Stage 3)
- `features_stage2_path`: Path to Stage 2 features
- `features_stage4_path`: Path to Stage 4 features
- `model_types`: List of model types to train
- `n_splits`: Number of k-fold splits (default: 5)
- `num_frames`: Number of frames per video (default: 1000)
- `output_dir`: Output directory (default: "data/stage5")
- `use_tracking`: Enable experiment tracking (default: True)
- `use_mlflow`: Enable MLflow tracking (default: True)
- `train_ensemble`: Train ensemble model (default: False)
- `delete_existing`: Delete existing models (default: False)
- `resume`: Resume training (skip completed folds) (default: True)

**Training Flow**:
1. **Validation**: Check prerequisites (Stage 2/3/4 outputs)
2. **Metadata Loading**: Load scaled metadata and features
3. **Data Integrity**: Validate data (min 3000 rows)
4. **Model Training Loop**:
   - For each model type:
     - Create model output directory
     - Hyperparameter grid search (if configured)
     - K-fold cross-validation:
       - For each fold (1-5):
         - Check if fold already exists (resume mode)
         - Split data (train/validation)
         - Train model (baseline/PyTorch/XGBoost)
         - Evaluate on validation set
         - Save model checkpoint
         - Log metrics (ExperimentTracker + MLflow)
     - Aggregate results across folds
     - Generate plots (CV fold comparison, hyperparameter search)
     - Save aggregated metrics

### Model Training Functions

#### Baseline Models
**Function**: `_train_baseline_model_fold()`

**Process**:
1. Load Stage 2 and Stage 4 features
2. Extract features from videos
3. Train sklearn model (LogisticRegression or SVM)
4. Evaluate on validation set
5. Save model (.joblib) and scaler
6. Log metrics

**Output Files**:
- `fold_{N}/model.joblib`: Trained model
- `fold_{N}/scaler.joblib`: Feature scaler
- `fold_{N}/metadata.json`: Fold metrics
- `metrics.json`: Aggregated metrics across folds

#### PyTorch Models
**Function**: `_train_pytorch_model_fold()`

**Process**:
1. Create VideoDataset for train/validation
2. Create DataLoader with appropriate batch size
3. Initialize model from config
4. Train with `fit()` function:
   - Multiple epochs
   - Training loop with gradient accumulation
   - Validation after each epoch
   - Early stopping
   - Learning rate scheduling
5. Save best model checkpoint
6. Log metrics to ExperimentTracker and MLflow

**Output Files**:
- `fold_{N}/model.pt` or `fold_{N}/checkpoint.pt`: Model checkpoint
- `fold_{N}/metrics.jsonl`: Training history (epoch-by-epoch)
- `fold_{N}/metadata.json`: Fold summary metrics
- `metrics.json`: Aggregated metrics

#### XGBoost Models
**Function**: `_train_xgboost_model_fold()`

**Process**:
1. Load pretrained model (I3D, R2+1D, ViT-GRU, etc.)
2. Extract features from videos using pretrained model
3. Train XGBoost classifier on extracted features
4. Evaluate on validation set
5. Save XGBoost model and metadata

**Output Files**:
- `fold_{N}/xgboost_model.json`: XGBoost model
- `fold_{N}/metadata.json`: Fold metrics
- `metrics.json`: Aggregated metrics

---

## Metrics & Tracking

### ExperimentTracker
**File**: `lib/mlops/config.py`

**Class**: `ExperimentTracker`

**Purpose**: Local file-based experiment tracking

**Metrics Stored**:
- Training metrics: `loss`, `accuracy`, `f1`, `precision`, `recall`
- Validation metrics: Same as training
- Per-class metrics: `precision_class0`, `recall_class0`, `f1_class0`, etc.
- Epoch-by-epoch history

**Storage Format**:
- `metrics.jsonl`: JSON Lines format, one entry per metric log
- `config.json`: Model configuration
- `metadata.json`: Run metadata (run_id, timestamp, etc.)

**Metrics File Structure** (`metrics.jsonl`):
```json
{"run_id": "...", "timestamp": "...", "step": 1, "epoch": 1, "phase": "train", "metric": "loss", "value": 0.6234}
{"run_id": "...", "timestamp": "...", "step": 1, "epoch": 1, "phase": "val", "metric": "loss", "value": 0.5123}
```

**Location**: `data/stage5/{model_type}/fold_{N}/metrics.jsonl`

### Aggregated Metrics
**File**: `data/stage5/{model_type}/metrics.json`

**Content**:
- Mean and std of metrics across folds
- Best fold identification
- Hyperparameter search results (if applicable)

**Example Structure**:
```json
{
  "model_type": "logistic_regression",
  "n_splits": 5,
  "fold_results": [
    {"fold": 1, "val_acc": 0.85, "val_f1": 0.82, ...},
    ...
  ],
  "mean_val_acc": 0.84,
  "std_val_acc": 0.02,
  "best_fold": 2
}
```

---

## MLflow Integration

### MLflowTracker
**File**: `lib/mlops/mlflow_tracker.py`

**Class**: `MLflowTracker`

**Purpose**: MLflow-based experiment tracking with UI

### Integration Points

#### 1. Tracker Creation
**Location**: `lib/training/pipeline.py` (line ~728)

```python
if use_mlflow and MLFLOW_AVAILABLE:
    mlflow_tracker = create_mlflow_tracker(
        experiment_name=f"{model_type}", 
        use_mlflow=True
    )
    mlflow_tracker.log_config(model_config)
    mlflow_tracker.set_tag("fold", str(fold_idx + 1))
    mlflow_tracker.set_tag("model_type", model_type)
```

#### 2. Metrics Logging
**Location**: `lib/training/pipeline.py` (line ~1001)

```python
mlflow_metrics = {
    "val_loss": val_loss,
    "val_acc": val_acc,
    "val_f1": val_f1,
    "val_precision": val_precision,
    "val_recall": val_recall,
    "val_precision_class0": ...,
    "val_recall_class0": ...,
    "val_f1_class0": ...,
}
mlflow_tracker.log_metrics(mlflow_metrics, step=fold_idx + 1)
```

#### 3. Artifact Logging
**Location**: `lib/training/pipeline.py` (line ~1018)

```python
if model_path.exists() and model_path.stat().st_size > 0:
    mlflow_tracker.log_artifact(str(model_path), artifact_path="models")
```

#### 4. Run Management
**Location**: `lib/training/pipeline.py` (line ~1075)

```python
if mlflow_tracker is not None:
    mlflow.flush()
    mlflow_tracker.end_run()
```

### MLflow Storage
**Location**: `mlruns/` directory

**Structure**:
```
mlruns/
  {experiment_id}/
    {run_id}/
      artifacts/
        models/
          model.pt
      metrics/
        val_acc
        val_f1
        ...
      params/
        learning_rate
        batch_size
        ...
      tags/
        fold
        model_type
        ...
```

### Accessing MLflow UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

**Features**:
- Compare experiments across models
- View metrics over time
- Download model artifacts
- Filter by tags (model_type, fold, etc.)

---

## DuckDB Analytics

### DuckDBAnalytics Class
**File**: `lib/utils/duckdb_analytics.py`

**Purpose**: Fast SQL queries on training results

### Usage in Notebooks
**Location**: `src/notebooks/executed/*.ipynb`

**Example**:
```python
from lib.utils.duckdb_analytics import DuckDBAnalytics

analytics = DuckDBAnalytics()

# Register metrics file
analytics.register_parquet('results', 'data/stage5/xgboost_i3d/metrics.json')

# Query metrics
result = analytics.query("""
    SELECT 
        fold,
        AVG(val_f1) as avg_f1,
        STDDEV(val_f1) as std_f1
    FROM results
    GROUP BY fold
""")
```

### Available Analytics Functions

#### 1. Video Statistics
```python
analytics.get_video_statistics(metadata_table="metadata")
```

#### 2. Feature Statistics
```python
analytics.get_feature_statistics(features_table="features")
```

#### 3. Training Results Summary
```python
analytics.get_training_results_summary(results_table="training_results")
```

**Query**:
```sql
SELECT 
    model_type,
    AVG(val_acc) as avg_val_acc,
    STDDEV(val_acc) as std_val_acc,
    AVG(val_loss) as avg_val_loss,
    STDDEV(val_loss) as std_val_loss,
    COUNT(*) as fold_count
FROM training_results
GROUP BY model_type
ORDER BY avg_val_acc DESC
```

### Supported Data Sources
- Polars DataFrames: `register_dataframe()`
- Arrow files: `register_arrow()`
- Parquet files: `register_parquet()`
- JSON files: Convert to Parquet first

---

## Airflow Orchestration

### DAG Definition
**File**: `airflow/dags/fvc_pipeline_dag.py`

**DAG Name**: `fvc_pipeline`

### Stage 5 Task
**Function**: `stage5_training()`

**Configuration**:
```python
stage5_train_models(
    project_root=str(project_root),
    scaled_metadata_path="data/scaled_videos/scaled_metadata.arrow",
    features_stage2_path="data/features_stage2/features_metadata.arrow",
    features_stage4_path="data/features_stage4/features_metadata.arrow",
    model_types=["logistic_regression", "svm", "i3d"],
    n_splits=5,
    output_dir="data/training_results"
)
```

### Task Dependencies
```
stage1_augmentation 
  → stage2_features 
  → [stage3_scaling, stage4_scaled_features] 
  → stage5_training
```

### Airflow Features
- **Retry Logic**: 1 retry with 5-minute delay
- **Email Notifications**: On failure (if configured)
- **Manual Trigger**: `schedule_interval=None`
- **DAG Tags**: `['fvc', 'video-classification', 'ml']`

### Accessing Airflow UI
```bash
# Start Airflow webserver
airflow webserver --port 8080

# Start Airflow scheduler
airflow scheduler
```

**URL**: http://localhost:8080

---

## Plots & Visualizations

### Plot Generation
**Note**: Plot generation is now handled by Jupyter notebooks in `src/notebooks/executed/` and utility functions in `src/notebooks/notebook_utils.py`.

**Purpose**: Generate comprehensive plots from trained models

### Generated Plots

#### 1. Training Curves
**File**: `data/stage5/{model_type}/plots/training_curves.png`

**Content**:
- Training/Validation Loss over epochs
- Training/Validation Accuracy over epochs
- Training/Validation F1 Score over epochs
- Loss comparison overlay

**Source**: `metrics.jsonl` files from each fold

#### 2. ROC/PR Curves
**File**: `data/stage5/{model_type}/plots/roc_pr_curves.png`

**Content**:
- ROC Curve with AUC score
- Precision-Recall Curve with AP score

**Metrics**:
- AUC-ROC: Area under ROC curve
- AP: Average Precision

#### 3. Confusion Matrix
**File**: `data/stage5/{model_type}/plots/confusion_matrix.png`

**Content**: Heatmap of true vs predicted labels

#### 4. Prediction Distribution
**File**: `data/stage5/{model_type}/plots/prediction_distribution.png`

**Content**:
- Histogram of predictions by class
- Box plot of prediction distributions

#### 5. CV Fold Comparison
**File**: `data/stage5/{model_type}/plots/cv_fold_comparison.png`

**Generated By**: `lib/training/visualization.py`

**Function**: `plot_cv_fold_comparison()`

**Content**: Comparison of metrics across 5 folds

#### 6. Hyperparameter Search
**File**: `data/stage5/{model_type}/plots/hyperparameter_search.png`

**Generated By**: `lib/training/visualization.py`

**Function**: `plot_hyperparameter_search()`

**Content**: Visualization of hyperparameter search results

### Plot Generation Process

1. **Load Models**: Load trained models from each fold
2. **Generate Predictions**: Run inference on validation sets
3. **Aggregate Results**: Combine predictions across folds
4. **Generate Plots**: Create all visualization plots
5. **Save Metrics Summary**: `metrics_summary.json`

### Running Plot Generation

Plots are generated automatically when running the Jupyter notebooks:

```bash
# Run notebooks to generate plots
cd src/notebooks/executed
jupyter nbconvert --to notebook --execute --inplace 5a_logistic_regression.ipynb
```

Or use the validation script:
```bash
python validate_notebooks_comprehensive.py
```

---

## Time Logs & Performance

### Time Tracking Locations

#### 1. SLURM Scripts
**Location**: `scripts/slurm_jobs/slurm_stage5*.sh`

**Tracking**:
```bash
STAGE5_START=$(date +%s)
# ... training ...
STAGE5_END=$(date +%s)
STAGE5_DURATION=$((STAGE5_END - STAGE5_START))
log "Execution time: ${STAGE5_DURATION}s ($((${STAGE5_DURATION} / 60)) minutes)"
```

**Log Files**: `logs/stage5/stage5{model}-{JOB_ID}.out`

#### 2. Python Training Script
**Location**: `src/scripts/run_stage5_training.py`

**Tracking**:
```python
stage_start = time.time()
# ... training ...
stage_duration = time.time() - stage_start
logger.info("Execution time: %.2f seconds (%.2f minutes)", 
           stage_duration, stage_duration / 60)
```

**Log Files**: `logs/stage5_training_{timestamp}.log`

#### 3. Training Pipeline
**Location**: `lib/training/pipeline.py`

**Tracking**: Per-fold and per-model timing

### Performance Metrics from Logs

Based on `report/final_report.tex`:

#### Training Times
- **ViT-GRU**: ~72,883 seconds (~20.25 hours) for 5-fold CV
  - Feature extraction: ~18-20 hours (400 frames/video)
- **Gradient Boosting**: 1-2 hours
- **XGBoost on Pretrained Features**: 3-6 hours per model
  - Feature extraction is primary bottleneck

#### Inference Times
- **Baselines**: <10 ms per video
- **XGBoost**: 50-200 ms (including feature extraction)
- **Deep Learning**: 100-500 ms on GPU

#### Memory Usage
- **Feature-based models**: GPU: 16 GB, RAM: 80 GB
- **XGBoost on pretrained**: GPU: 16 GB, RAM: 80 GB
- **Deep learning models**: GPU: 16 GB, RAM: 80 GB
- **X3D**: Full GPU memory (16 GB) before OOM errors

### Extracting Time Logs

#### From SLURM Logs
```bash
# Extract execution times from all stage5 logs
grep "Execution time" logs/stage5/stage5*.out

# Extract specific model times
grep "Execution time" logs/stage5/stage5a-*.out
```

#### From Python Logs
```bash
# Extract execution times from training logs
grep "Execution time" logs/stage5_training_*.log

# Extract per-fold times
grep "Fold.*completed" logs/stage5_training_*.log
```

---

## Data Flow & Connections

### Input Data Sources

#### 1. Scaled Videos (Stage 3)
**Path**: `data/scaled_videos/scaled_metadata.arrow`

**Format**: Arrow/Parquet/CSV

**Required For**: All models

**Columns**:
- `video_path`: Path to scaled video file
- `label`: Binary label (0=real, 1=fake)
- Additional metadata columns

#### 2. Stage 2 Features
**Path**: `data/features_stage2/features_metadata.arrow`

**Required For**: Baseline models (logistic_regression, svm)

**Content**: Handcrafted features extracted from videos

#### 3. Stage 4 Features
**Path**: `data/features_stage4/features_scaled_metadata.arrow`

**Required For**: Models using `*_stage2_stage4` variants

**Content**: Features extracted from scaled videos

### Output Data Structure

```
data/stage5/
  {model_type}/
    fold_1/
      model.pt (or model.joblib, xgboost_model.json)
      metrics.jsonl
      metadata.json
      scaler.joblib (if applicable)
    fold_2/
      ...
    fold_5/
      ...
    best_model/
      (best performing fold's model)
    plots/
      cv_fold_comparison.png
      hyperparameter_search.png
      training_curves.png (if generated)
      roc_pr_curves.png (if generated)
    metrics.json
```

### Data Connections Diagram

```
┌─────────────────┐
│  Stage 3 Output │
│ (Scaled Videos) │
└────────┬────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────┐              ┌─────────────────┐
│  Stage 2        │              │  Stage 4        │
│  Features       │              │  Features       │
└────────┬────────┘              └────────┬────────┘
         │                                 │
         └──────────┬──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Stage 5 Training   │
         │  Pipeline           │
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│  Experiment     │   │  MLflow         │
│  Tracker        │   │  Tracking       │
│  (metrics.jsonl)│   │  (mlruns/)      │
└─────────────────┘   └─────────────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  DuckDB Analytics   │
         │  (SQL Queries)       │
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Plot Generation     │
         │  (PNG files)         │
         └──────────────────────┘
```

### Integration Points

1. **SLURM → Python**: Environment variables passed to training script
2. **Python → Training Pipeline**: Function calls with parameters
3. **Training Pipeline → ExperimentTracker**: Metrics logging
4. **Training Pipeline → MLflow**: Metrics and artifacts
5. **Output Files → DuckDB**: Metrics files registered for analytics
6. **Output Files → Plot Generation**: Models and metrics loaded for visualization
7. **Airflow → All Stages**: Orchestrates entire pipeline

---

## Summary

### Complete Models (✅)
- logistic_regression (5a)
- svm (5b)
- pretrained_inception (5d)
- xgboost_pretrained_inception (5f)
- xgboost_i3d (5g)
- xgboost_r2plus1d (5h)
- xgboost_vit_gru (5i)
- xgboost_vit_transformer (5j)
- x3d (5q)
- slowfast (5r)

### Partial Models (⚠️)
- naive_cnn (5c): Only fold_1
- variable_ar_cnn (5e): No folds
- vit_gru (5k): No folds
- vit_transformer (5l): No folds
- i3d (5o): No folds
- r2plus1d (5p): No folds

### Missing Models (❌)
- timesformer (5m)
- vivit (5n)
- slowfast_attention (5s)
- slowfast_multiscale (5t)
- two_stream (5u)

### Key Integrations
- ✅ SLURM job submission and monitoring
- ✅ Python training pipeline with resume capability
- ✅ ExperimentTracker for local metrics
- ✅ MLflow for experiment tracking UI
- ✅ DuckDB for analytics queries
- ✅ Airflow for pipeline orchestration
- ✅ Comprehensive plot generation
- ✅ Time logging at multiple levels

### Next Steps
1. Complete training for partial models
2. Generate plots for all trained models
3. Extract and analyze time logs
4. Query metrics using DuckDB
5. Compare models in MLflow UI
6. Document any missing connections or issues

