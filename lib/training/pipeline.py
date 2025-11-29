"""
Model training pipeline.

Trains models using downscaled videos and extracted features.
Supports multiple model types and k-fold cross-validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional
import polars as pl
import torch
from torch.utils.data import DataLoader

from lib.data import stratified_kfold, load_metadata
from lib.models import VideoConfig, VideoDataset
from lib.mlops.config import ExperimentTracker, CheckpointManager
from lib.training.trainer import OptimConfig, TrainConfig, fit
from lib.training.model_factory import create_model, is_pytorch_model, get_model_config

logger = logging.getLogger(__name__)


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
    Stage 5: Train models using downscaled videos and features.
    
    Args:
        project_root: Project root directory
        downscaled_metadata_path: Path to downscaled metadata CSV
        features_stage2_path: Path to Stage 2 features metadata CSV
        features_stage4_path: Path to Stage 4 features metadata CSV
        model_types: List of model types to train
        n_splits: Number of k-fold splits
        num_frames: Number of frames per video
        output_dir: Directory to save training results
        use_tracking: Whether to use experiment tracking
    
    Returns:
        Dictionary of training results
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    logger.info("Stage 5: Loading metadata...")
    downscaled_df = pl.read_csv(downscaled_metadata_path)
    features2_df = pl.read_csv(features_stage2_path) if Path(features_stage2_path).exists() else None
    features4_df = pl.read_csv(features_stage4_path) if Path(features_stage4_path).exists() else None
    
    logger.info(f"Stage 5: Found {downscaled_df.height} downscaled videos")
    
    # Create video config
    video_config = VideoConfig(
        num_frames=num_frames,
        fixed_size=224,
    )
    
    results = {}
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage 5: Training model: {model_type}")
        logger.info(f"{'='*80}")
        
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model config
        model_config = get_model_config(model_type)
        
        # K-fold cross-validation
        fold_results = []
        
        for fold_idx in range(n_splits):
            logger.info(f"\nTraining {model_type} - Fold {fold_idx + 1}/{n_splits}")
            
            # Create fold splits
            splits = stratified_kfold(
                downscaled_df,
                n_splits=n_splits,
                fold_idx=fold_idx,
                random_seed=42
            )
            
            train_df = splits["train"]
            val_df = splits["val"]
            
            # Create datasets
            train_dataset = VideoDataset(
                train_df,
                project_root=str(project_root),
                config=video_config,
            )
            val_dataset = VideoDataset(
                val_df,
                project_root=str(project_root),
                config=video_config,
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_config.get("batch_size", 8),
                shuffle=True,
                num_workers=0,  # Conservative for memory
                pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=model_config.get("batch_size", 8),
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            # Train model
            if is_pytorch_model(model_type):
                # PyTorch model training
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
                    device=str(device),
                    use_amp=model_config.get("use_amp", True),
                    gradient_accumulation_steps=model_config.get("gradient_accumulation_steps", 1),
                    early_stopping_patience=model_config.get("early_stopping_patience", 5)
                )
                
                # Create tracker and checkpoint manager
                fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)
                
                if use_tracking:
                    tracker = ExperimentTracker(str(fold_output_dir))
                    ckpt_manager = CheckpointManager(str(fold_output_dir))
                else:
                    tracker = None
                    ckpt_manager = None
                
                logger.info(f"Training PyTorch model {model_type} on fold {fold_idx + 1}...")
                
                # Train
                try:
                    model = fit(
                        model,
                        train_loader,
                        val_loader,
                        optim_cfg,
                        train_cfg,
                    )
                    
                    # Evaluate final model
                    from lib.training.trainer import evaluate
                    val_loss, val_acc = evaluate(model, val_loader, device=str(device))
                    
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    
                    logger.info(f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training fold {fold_idx + 1}: {e}", exc_info=True)
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": float('nan'),
                        "val_acc": float('nan'),
                    })
                
                # Clear model
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                # Baseline model training (sklearn)
                logger.warning(f"Baseline model training not yet implemented in stage5")
                fold_results.append({
                    "fold": fold_idx + 1,
                    "val_loss": float('nan'),
                    "val_acc": float('nan'),
                })
        
        # Aggregate results
        if fold_results:
            avg_val_loss = sum(r["val_loss"] for r in fold_results if not (isinstance(r["val_loss"], float) and (r["val_loss"] != r["val_loss"]))) / len([r for r in fold_results if not (isinstance(r["val_loss"], float) and (r["val_loss"] != r["val_loss"]))])
            avg_val_acc = sum(r["val_acc"] for r in fold_results if not (isinstance(r["val_acc"], float) and (r["val_acc"] != r["val_acc"]))) / len([r for r in fold_results if not (isinstance(r["val_acc"], float) and (r["val_acc"] != r["val_acc"]))])
            
            results[model_type] = {
                "fold_results": fold_results,
                "avg_val_loss": avg_val_loss,
                "avg_val_acc": avg_val_acc,
            }
            
            logger.info(f"\n{model_type} - Avg Val Loss: {avg_val_loss:.4f}, Avg Val Acc: {avg_val_acc:.4f}")
    
    return results

