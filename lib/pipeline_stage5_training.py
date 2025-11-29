"""
Stage 5: Training Pipeline

Train models using:
- Downscaled videos (from Stage 3)
- M handcrafted features (from Stage 2)
- P additional features (from Stage 4)

No augmentation or feature generation in this stage - only training.

Borrows from previous pipeline:
- Training utilities (fit_with_tracking, train_one_epoch, evaluate)
- Metrics (basic_classification_metrics, confusion_matrix, roc_auc)
- MLOps infrastructure (ExperimentTracker, CheckpointManager)
- Model factory and baseline models
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .video_data import stratified_kfold
from .baseline_models import LogisticRegressionBaseline, SVMBaseline
from .model_factory import create_model, get_model_config, is_pytorch_model
from .video_training import (
    fit_with_tracking,
    train_one_epoch,
    evaluate,
    build_optimizer,
    build_scheduler,
    OptimConfig,
    TrainConfig,
    EarlyStopping
)
from .video_metrics import (
    collect_logits_and_labels,
    basic_classification_metrics,
    confusion_matrix,
    roc_auc
)
from .mlops_core import ExperimentTracker, CheckpointManager, RunConfig, create_run_directory
from .mlops_utils import aggressive_gc, log_memory_stats, safe_execute

logger = logging.getLogger(__name__)


class CombinedVideoFeatureDataset(Dataset):
    """
    Dataset that combines downscaled videos with handcrafted features.
    
    Returns:
        (video_tensor, features, label)
    """
    
    def __init__(
        self,
        metadata_df: pl.DataFrame,
        project_root: str,
        downscaled_metadata_path: str,
        features_stage2_path: str,
        features_stage4_path: str,
        num_frames: int = 8,
        train: bool = True
    ):
        self.metadata_df = metadata_df
        self.project_root = Path(project_root)
        self.num_frames = num_frames
        self.train = train
        
        # Load feature metadata
        self.features_stage2 = pl.read_csv(features_stage2_path)
        self.features_stage4 = pl.read_csv(features_stage4_path)
        
        # Create lookup dictionaries
        self.features_stage2_lookup = {
            row["video_path"]: row["feature_file"]
            for row in self.features_stage2.iter_rows(named=True)
        }
        self.features_stage4_lookup = {
            row["video_path"]: row["feature_file"]
            for row in self.features_stage4.iter_rows(named=True)
        }
        
        # Label mapping
        labels = sorted(self.metadata_df["label"].unique().to_list())
        self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    
    def __len__(self):
        return self.metadata_df.height
    
    def __getitem__(self, idx):
        row = self.metadata_df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        label_idx = self.label_to_idx[label]
        
        # Load downscaled video
        from .video_modeling import _read_video_wrapper, uniform_sample_indices
        from .video_paths import resolve_video_path
        
        video_path = resolve_video_path(video_rel, self.project_root)
        video = _read_video_wrapper(video_path)
        
        # Sample frames
        indices = uniform_sample_indices(video.shape[0], self.num_frames)
        frames = [video[i] for i in indices]
        video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
        
        # Load M features (Stage 2)
        features_m = np.zeros(50)  # Default size
        if video_rel in self.features_stage2_lookup:
            feature_file = self.project_root / self.features_stage2_lookup[video_rel]
            if feature_file.exists():
                features_m = np.load(str(feature_file))
        
        # Load P features (Stage 4)
        features_p = np.zeros(20)  # Default size
        if video_rel in self.features_stage4_lookup:
            feature_file = self.project_root / self.features_stage4_lookup[video_rel]
            if feature_file.exists():
                features_p = np.load(str(feature_file))
        
        # Concatenate features
        all_features = np.concatenate([features_m, features_p])
        features_tensor = torch.from_numpy(all_features).float()
        
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return video_tensor, features_tensor, label_tensor


def stage5_train_models(
    project_root: str,
    downscaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: str,
    model_types: List[str],
    n_splits: int = 5,
    num_frames: int = 8,
    output_dir: str = "data/training_results",
    use_tracking: bool = True
) -> Dict:
    """
    Stage 5: Train models using downscaled videos + M + P features.
    
    Args:
        project_root: Project root directory
        downscaled_metadata_path: Path to Stage 3 metadata CSV
        features_stage2_path: Path to Stage 2 features metadata CSV
        features_stage4_path: Path to Stage 4 features metadata CSV
        model_types: List of model types to train
        n_splits: Number of k-fold splits
        num_frames: Number of frames to sample per video
        output_dir: Directory to save training results
    
    Returns:
        Dictionary with training results
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Stage 5: Starting training pipeline...")
    logger.info(f"Stage 5: Models to train: {model_types}")
    logger.info(f"Stage 5: K-fold splits: {n_splits}")
    logger.info(f"Stage 5: Experiment tracking: {use_tracking}")
    
    # Create experiment tracker for Stage 5 (if tracking enabled)
    if use_tracking:
        from .mlops_core import create_run_directory
        run_dir, run_id = create_run_directory(str(output_dir), "stage5_training")
        stage5_tracker = ExperimentTracker(str(run_dir), run_id)
        logger.info(f"Stage 5: Run ID: {run_id}")
    
    # Load metadata
    metadata_df = pl.read_csv(downscaled_metadata_path)
    logger.info(f"Stage 5: Loaded {metadata_df.height} videos")
    
    # Create k-fold splits
    folds = stratified_kfold(metadata_df, n_splits=n_splits, random_state=42)
    logger.info(f"Stage 5: Created {len(folds)} folds")
    
    all_results = {}
    
    # Train each model
    for model_type in model_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage 5: Training model: {model_type}")
        logger.info(f"{'='*80}")
        
        model_results = []
        
        # Train on each fold
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            logger.info(f"\nStage 5: Fold {fold_idx + 1}/{n_splits}")
            
            # Create train/val splits
            train_df = metadata_df[train_indices]
            val_df = metadata_df[val_indices]
            
            # Create datasets
            train_dataset = CombinedVideoFeatureDataset(
                train_df,
                str(project_root),
                downscaled_metadata_path,
                features_stage2_path,
                features_stage4_path,
                num_frames=num_frames,
                train=True
            )
            
            val_dataset = CombinedVideoFeatureDataset(
                val_df,
                str(project_root),
                downscaled_metadata_path,
                features_stage2_path,
                features_stage4_path,
                num_frames=num_frames,
                train=False
            )
            
            # Get model config
            model_config = get_model_config(model_type)
            batch_size = model_config.get("batch_size", 8)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Conservative
                pin_memory=torch.cuda.is_available()
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            # Train model
            if is_pytorch_model(model_type):
                # PyTorch model training - use existing training utilities
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = create_model(model_type, model_config)
                model = model.to(device)
                
                # Create optimizer and scheduler
                optim_cfg = OptimConfig(
                    lr=model_config.get("learning_rate", 1e-4),
                    weight_decay=model_config.get("weight_decay", 1e-4)
                )
                train_cfg = TrainConfig(
                    num_epochs=model_config.get("num_epochs", 20),
                    device=device,
                    use_amp=model_config.get("use_amp", True),
                    gradient_accumulation_steps=model_config.get("gradient_accumulation_steps", 1),
                    early_stopping_patience=model_config.get("early_stopping_patience", 5)
                )
                
                # Create tracker and checkpoint manager for this fold
                fold_output_dir = output_dir / model_type / f"fold_{fold_idx + 1}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)
                
                tracker = ExperimentTracker(str(fold_output_dir))
                ckpt_manager = CheckpointManager(str(fold_output_dir))
                
                logger.info(f"Training PyTorch model {model_type} on fold {fold_idx + 1}...")
                
                # Use existing fit_with_tracking for training
                try:
                    trained_model = safe_execute(
                        lambda: fit_with_tracking(
                            model,
                            train_loader,
                            val_loader,
                            optim_cfg,
                            train_cfg,
                            tracker,
                            ckpt_manager
                        ),
                        context=f"training {model_type} fold {fold_idx + 1}",
                        oom_retry=True,
                        max_retries=1
                    )
                    
                    # Evaluate final model
                    val_loss, val_acc = evaluate(trained_model, val_loader, device)
                    
                    # Collect predictions for detailed metrics
                    logits, labels = collect_logits_and_labels(trained_model, val_loader, device)
                    metrics = basic_classification_metrics(logits, labels)
                    metrics["val_loss"] = val_loss
                    metrics["val_accuracy"] = val_acc
                    
                    # Add confusion matrix and ROC AUC
                    cm = confusion_matrix(logits, labels)
                    metrics["confusion_matrix"] = cm.tolist()
                    
                    roc_auc_score = roc_auc(logits, labels)
                    if not np.isnan(roc_auc_score):
                        metrics["roc_auc"] = roc_auc_score
                    
                    model_results.append({
                        "fold": fold_idx + 1,
                        "model_type": model_type,
                        **metrics
                    })
                    
                    logger.info(f"✓ Fold {fold_idx + 1} results: {metrics}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type} on fold {fold_idx + 1}: {e}", exc_info=True)
                    model_results.append({
                        "fold": fold_idx + 1,
                        "model_type": model_type,
                        "error": str(e)
                    })
                
            else:
                # Baseline model (sklearn)
                logger.info(f"Training baseline model {model_type} on fold {fold_idx + 1}...")
                
                # Extract features and labels
                X_train = []
                y_train = []
                for video_tensor, features_tensor, label_tensor in train_loader:
                    # Use features only for baseline models
                    X_train.append(features_tensor.numpy())
                    y_train.append(label_tensor.numpy())
                
                X_train = np.concatenate(X_train, axis=0)
                y_train = np.concatenate(y_train, axis=0)
                
                # Train baseline
                if model_type == "logistic_regression":
                    model = LogisticRegressionBaseline()
                elif model_type == "svm":
                    model = SVMBaseline()
                else:
                    logger.warning(f"Unknown baseline model: {model_type}")
                    continue
                
                model.fit(X_train, y_train)
                
                # Evaluate
                X_val = []
                y_val = []
                for video_tensor, features_tensor, label_tensor in val_loader:
                    X_val.append(features_tensor.numpy())
                    y_val.append(label_tensor.numpy())
                
                X_val = np.concatenate(X_val, axis=0)
                y_val = np.concatenate(y_val, axis=0)
                
                # Evaluate using existing metrics
                logger.info(f"Evaluating {model_type} on fold {fold_idx + 1}...")
                
                predictions = model.predict(X_val)
                probabilities = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Convert to tensors for metrics
                if probabilities is not None:
                    pred_tensor = torch.from_numpy(probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.flatten()).float()
                else:
                    pred_tensor = torch.from_numpy(predictions).float()
                label_tensor = torch.from_numpy(y_val).long()
                
                # Use existing metrics
                metrics = basic_classification_metrics(pred_tensor, label_tensor)
                cm = confusion_matrix(pred_tensor, label_tensor)
                metrics["confusion_matrix"] = cm.tolist()
                
                roc_auc_score = roc_auc(pred_tensor, label_tensor)
                if not np.isnan(roc_auc_score):
                    metrics["roc_auc"] = roc_auc_score
                
                model_results.append({
                    "fold": fold_idx + 1,
                    "model_type": model_type,
                    **metrics
                })
                
                logger.info(f"✓ Fold {fold_idx + 1} results: {metrics}")
            
            # Clear memory
            aggressive_gc(clear_cuda=True)
        
        all_results[model_type] = model_results
    
    logger.info(f"\n✓ Stage 5 complete: Trained {len(model_types)} models")
    return all_results

