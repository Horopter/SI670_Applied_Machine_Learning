"""
Unified feature-based training pipeline for all models.

All models use Stage 2/4 features (not videos).
Implements: 60-20-20 split, imputation, scaling, normalization, stratified splits,
no data leaks, OOM resistant, 5-fold CV, hyperparameter tuning, F1/BCE evaluation,
ROC and PR curves.
"""

from __future__ import annotations

import logging
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt

from lib.training.feature_pipeline import (
    FeatureDataset, FeaturePreprocessor, create_stratified_splits,
    train_model_with_cv, evaluate_model, plot_roc_pr_curves
)
from lib.training.feature_models import create_feature_model
from lib.training.feature_preprocessing import load_and_combine_features
from lib.utils.paths import load_metadata_flexible
from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


# Model type mappings to feature-based architectures
MODEL_TYPE_MAPPING = {
    # Baseline models -> MLP
    "logistic_regression": "mlp",
    "logistic_regression_stage2": "mlp",
    "logistic_regression_stage2_stage4": "mlp",
    "svm": "mlp",  # SVM implemented as MLP with hinge loss equivalent
    "svm_stage2": "mlp",
    "svm_stage2_stage4": "mlp",
    
    # CNN models -> 1D CNN
    "naive_cnn": "cnn1d",
    "variable_ar_cnn": "cnn1d",
    
    # Transformer models -> Transformer
    "vit_transformer": "transformer",
    "timesformer": "transformer",
    "vivit": "transformer",
    
    # LSTM/GRU models -> LSTM
    "vit_gru": "lstm",
    
    # Complex models -> ResNet-style
    "slowfast": "resnet",
    "x3d": "resnet",
    "i3d": "resnet",
    "r2plus1d": "resnet",
    "pretrained_inception": "resnet",
    "two_stream": "resnet",
    "slowfast_attention": "resnet",
    "slowfast_multiscale": "resnet",
    
    # XGBoost models -> MLP (we'll handle XGBoost separately)
    "xgboost_i3d": "mlp",
    "xgboost_r2plus1d": "mlp",
    "xgboost_vit_gru": "mlp",
    "xgboost_vit_transformer": "mlp",
    "xgboost_pretrained_inception": "mlp",
}


# Hyperparameter grids for each model type
# All grids reduced to <50 combinations
HYPERPARAMETER_GRIDS = {
    "mlp": {
        "hidden_dims": [[256, 128, 64], [512, 256, 128], [128, 64], [256, 128]],  # 4 values (4*4*2*2 = 64 combinations max)
        "dropout": [0.3, 0.5, 0.7, 0.4],  # 4 values
        "learning_rate": [1e-4, 1e-3],  # 2 values
        "weight_decay": [1e-5, 1e-4],  # 2 values
    },
    "cnn1d": {
        "num_filters": [[64, 128, 256], [32, 64, 128]],  # 2*2*2*2*2 = 32 combinations (OK)
        "kernel_sizes": [[3, 3, 3], [5, 5, 5]],
        "dropout": [0.3, 0.5],
        "learning_rate": [1e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
    },
    "transformer": {
        "d_model": [128, 256],  # 2*2*2*2*2*2 = 64 -> 2*2*2*2*2*1 = 32 combinations
        "nhead": [4, 8],
        "num_layers": [2, 4],
        "dim_feedforward": [512, 1024],
        "dropout": [0.1, 0.2],
        "learning_rate": [1e-4, 1e-3],
        "weight_decay": [1e-5],  # Reduced from 2 to 1 value
    },
    "lstm": {
        "hidden_dim": [128, 256],  # 2*2*2*2*2 = 32 combinations (OK)
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5],
        "learning_rate": [1e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
    },
    "resnet": {
        "hidden_dims": [[256, 512, 256], [128, 256, 128]],  # 2*2*2*2 = 16 combinations (OK)
        "dropout": [0.3, 0.5],
        "learning_rate": [1e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
    },
}


def load_features_for_training(
    features_stage2_path: Optional[str],
    features_stage4_path: Optional[str],
    video_paths: List[str],
    project_root: str
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Load and combine features for training.
    
    Collinearity removal is done ONCE here before any splits to avoid data leakage.
    
    Args:
        features_stage2_path: Path to Stage 2 features metadata (optional)
        features_stage4_path: Path to Stage 4 features metadata (optional)
        video_paths: List of video paths to match features
        project_root: Project root directory
    
    Returns:
        Tuple of (features, feature_names, valid_video_indices)
        - features: Feature matrix (n_samples, n_features)
        - feature_names: List of feature names
        - valid_video_indices: Indices of videos that have valid features (None if all videos are valid)
    
    Raises:
        ValueError: If feature loading fails or no valid features found
        FileNotFoundError: If required feature files are missing
    """
    """
    Load and combine features for training.
    
    Collinearity removal is done ONCE here before any splits to avoid data leakage.
    
    Returns:
        Tuple of (features, feature_names, valid_video_indices)
        - features: Feature matrix (n_samples, n_features)
        - feature_names: List of feature names
        - valid_video_indices: Indices of videos that have valid features (None if all videos are valid)
    """
    # Input validation
    if not video_paths or not isinstance(video_paths, list):
        raise ValueError(f"video_paths must be a non-empty list, got: {type(video_paths)}")
    if not project_root or not isinstance(project_root, str):
        raise ValueError(f"project_root must be a non-empty string, got: {type(project_root)}")
    
    try:
        # Load features WITHOUT collinearity removal first
        features, feature_names, _, valid_video_indices = load_and_combine_features(
            features_stage2_path=features_stage2_path,
            features_stage4_path=features_stage4_path,
            video_paths=video_paths,
            project_root=project_root,
            remove_collinearity=False,  # Do it separately below
            correlation_threshold=0.95
        )
        
        # Validate loaded features
        if features is None or len(features) == 0:
            raise ValueError("No features loaded from Stage 2/4")
        if feature_names is None or len(feature_names) == 0:
            raise ValueError("No feature names loaded")
        
        # Remove collinear features ONCE before any splits
        original_feature_count = len(feature_names)
        logger.info(f"Removing collinear features BEFORE data splits (to avoid data leakage)...")
        logger.info(f"  Original feature count: {original_feature_count}")
        from lib.training.feature_preprocessing import remove_collinear_features
        features, kept_feature_indices, feature_names = remove_collinear_features(
            features,
            feature_names=feature_names,
            correlation_threshold=0.95,
            method="correlation"
        )
        final_feature_count = len(feature_names)
        logger.info(f"  Final feature count after collinearity removal: {final_feature_count}/{original_feature_count} ({100*final_feature_count/original_feature_count:.1f}% retained)")
        
        # Validate after collinearity removal
        if len(features) == 0 or features.shape[1] == 0:
            raise ValueError("All features removed during collinearity removal - cannot proceed")
        
        # Return video valid indices (not feature indices!)
        return features, feature_names, valid_video_indices
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load features: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading features: {e}", exc_info=True)
        raise ValueError(f"Feature loading failed: {e}") from e


def train_feature_model(
    model_type: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    n_splits: int = 5,
    batch_size: int = 32,
    epochs: int = 100,
    use_gpu: bool = True,
    device: Optional[torch.device] = None,
    hyperparameter_grid: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Train a feature-based model with hyperparameter tuning and CV.
    
    Args:
        model_type: Original model type name
        features: Feature matrix
        labels: Label array
        feature_names: Feature names
        output_dir: Output directory
        n_splits: Number of CV folds
        batch_size: Batch size
        epochs: Max epochs
        use_gpu: Use GPU if available
        device: Torch device
        hyperparameter_grid: Hyperparameter grid (auto-selected if None)
    
    Returns:
        Training results dictionary
    """
    # Map model type to architecture
    architecture = MODEL_TYPE_MAPPING.get(model_type, "mlp")
    
    # Get hyperparameter grid
    if hyperparameter_grid is None:
        hyperparameter_grid = HYPERPARAMETER_GRIDS.get(architecture, HYPERPARAMETER_GRIDS["mlp"])
    
    # Setup device
    if device is None:
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU for {model_type}")
        else:
            device = torch.device("cpu")
            logger.info(f"Using CPU for {model_type} (GPU not available)")
    
    input_dim = features.shape[1]
    logger.info(f"Training {model_type} (architecture: {architecture}, input_dim: {input_dim})")
    
    # Collinearity removal already done in load_features_for_training
    # Now create 60-20-20 split AFTER collinearity removal
    logger.info("Creating 60-20-20 stratified train-val-test split...")
    train_idx, val_idx, test_idx = create_stratified_splits(
        features, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    
    X_train, X_val, X_test = features[train_idx], features[val_idx], features[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"✓ Features cleaned BEFORE splits: {len(feature_names)} features (collinearity already removed)")
    
    # OPTIMIZATION: Use 20% stratified sample for hyperparameter search (faster)
    # Final training will use full dataset for robustness
    from sklearn.model_selection import StratifiedShuffleSplit
    
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH: Using 20% stratified sample for efficiency")
    logger.info("=" * 80)
    
    # Sample 20% of train+val for hyperparameter search
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
    sample_indices, _ = next(sss.split(X_trainval, y_trainval))
    
    X_trainval_sample = X_trainval[sample_indices]
    y_trainval_sample = y_trainval[sample_indices]
    
    # Split sample into train/val for hyperparameter search
    from sklearn.model_selection import train_test_split
    X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
        X_trainval_sample, y_trainval_sample, test_size=0.2, random_state=42, stratify=y_trainval_sample
    )
    
    logger.info(f"Hyperparameter search sample: {len(X_trainval_sample)} rows ({100.0 * len(X_trainval_sample) / len(X_trainval):.1f}% of {len(X_trainval)} total)")
    logger.info(f"  Sample train: {len(X_train_sample)}, Sample val: {len(X_val_sample)}")
    
    # Hyperparameter search on 20% sample
    best_score = -1
    best_params = None
    best_model = None
    best_preprocessor = None
    grid_results = []
    
    param_grid = ParameterGrid(hyperparameter_grid)
    logger.info(f"Hyperparameter search: {len(param_grid)} combinations")
    
    for param_idx, params in enumerate(param_grid):
        logger.info(f"Grid search {param_idx + 1}/{len(param_grid)}: {params} (20% sample)")
        
        # Separate model params from training params
        model_params = {k: v for k, v in params.items() 
                       if k not in ["learning_rate", "weight_decay", "batch_size", "epochs"]}
        train_params = {k: v for k, v in params.items() 
                       if k in ["learning_rate", "weight_decay"]}
        
        # Create model
        model = create_feature_model(architecture, input_dim, **model_params)
        model = model.to(device)
        
        # Create preprocessor
        preprocessor = FeaturePreprocessor(
            imputation_strategy="mean",
            scaling_method="standard",
            normalize=True
        )
        
        # Preprocess 20% sample data
        X_train_sample_processed = preprocessor.fit_transform(X_train_sample)
        X_val_sample_processed = preprocessor.transform(X_val_sample)
        
        # Create datasets from 20% sample
        train_dataset = FeatureDataset(X_train_sample_processed, y_train_sample, feature_names)
        val_dataset = FeatureDataset(X_val_sample_processed, y_val_sample, feature_names)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_params.get("learning_rate", 1e-3),
            weight_decay=train_params.get("weight_decay", 1e-5)
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop with early stopping
        best_val_f1 = -1
        patience_counter = 0
        patience = 10
        best_model_state = None  # Initialize to None for safety
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).float()
                
                optimizer.zero_grad()
                logits = model(batch_features)
                loss = criterion(logits.squeeze(), batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_probs = []
            val_labels_list = []
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    logits = model(batch_features)
                    probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                    val_probs.extend(probs)
                    val_labels_list.extend(batch_labels.numpy())
            
            val_probs = np.array(val_probs)
            val_labels_array = np.array(val_labels_list)
            val_preds = (val_probs > 0.5).astype(int)
            val_f1 = f1_score(val_labels_array, val_preds)
            val_acc = accuracy_score(val_labels_array, val_preds)
            
            # Calculate train metrics only when logging (every 10 epochs or first epoch)
            should_log = (epoch + 1) % 10 == 0 or epoch == 0
            if should_log:
                model.train()
                train_probs = []
                train_labels_list = []
                with torch.no_grad():
                    for batch_features, batch_labels in train_loader:
                        batch_features = batch_features.to(device)
                        logits = model(batch_features)
                        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                        train_probs.extend(probs)
                        train_labels_list.extend(batch_labels.numpy())
                
                train_probs = np.array(train_probs)
                train_labels_array = np.array(train_labels_list)
                train_preds = (train_probs > 0.5).astype(int)
                train_f1 = f1_score(train_labels_array, train_preds)
                train_acc = accuracy_score(train_labels_array, train_preds)
                
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_loss/len(train_loader):.4f}, "
                    f"train_f1={train_f1:.4f}, "
                    f"train_acc={train_acc:.4f}, "
                    f"val_f1={val_f1:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            logger.warning(f"No best model state found for params {params}, using final model state")
            best_model_state = model.state_dict().copy()
            best_val_f1 = val_f1  # Use final epoch's F1 if no improvement
        
        grid_results.append({
            "params": params,
            "val_f1": best_val_f1
        })
        
        if best_val_f1 > best_score:
            best_score = best_val_f1
            best_params = params.copy()
            # Use create_feature_model to ensure input_dim is passed correctly
            best_model = create_feature_model(architecture, input_dim, **model_params)
            best_model.load_state_dict(best_model_state)
            best_preprocessor = preprocessor
        
        # Cleanup
        del model, train_dataset, val_dataset, train_loader, val_loader
        aggressive_gc(clear_cuda=use_gpu)
    
    # Safety check: ensure we have valid best_params
    if best_params is None:
        raise ValueError(
            "No valid hyperparameters found during grid search. "
            "This may indicate all models failed to train or all validation F1 scores were invalid."
        )
    
    logger.info(f"Best hyperparameters from 20% sample: {best_params} (val_f1: {best_score:.4f})")
    
    # FINAL TRAINING: Train on full dataset with best hyperparameters
    logger.info("=" * 80)
    logger.info("FINAL TRAINING: Using full dataset with best hyperparameters")
    logger.info("=" * 80)
    logger.info(f"Full dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Train final model with best params on full training+val set with 5-fold CV
    X_trainval_full = np.vstack([X_train, X_val])
    y_trainval_full = np.concatenate([y_train, y_val])
    
    # Separate model and training params
    model_params = {k: v for k, v in best_params.items() 
                   if k not in ["learning_rate", "weight_decay", "batch_size", "epochs"]}
    train_params = {k: v for k, v in best_params.items() 
                   if k in ["learning_rate", "weight_decay"]}
    
    final_model = create_feature_model(architecture, input_dim, **model_params)
    # Move model to device and ensure all parameters/buffers are on device
    final_model = final_model.to(device)
    # Explicitly verify all parameters are on device
    for param in final_model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    for buffer in final_model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    final_preprocessor = FeaturePreprocessor()
    
    # 5-fold CV on train+val
    def model_factory(input_dim):
        return create_feature_model(architecture, input_dim, **model_params)
    
    # 5-fold CV on train+val (validates best hyperparameters found in grid search)
    # CV trains models on folds of train+val, providing validation metrics
    cv_results = train_model_with_cv(
        model_factory,
        input_dim,
        X_trainval_full,
        y_trainval_full,
        feature_names,
        n_splits=n_splits,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=train_params.get("learning_rate", 1e-3),
        weight_decay=train_params.get("weight_decay", 1e-5),
        device=device,
        preprocessor=final_preprocessor,
        use_gpu=use_gpu
    )
    
    # Train final model once on full train+val set with best hyperparameters
    # (CV validated hyperparameters, now train on full data for final test evaluation)
    logger.info("Training final model on full training+validation set with best hyperparameters...")
    final_preprocessor.fit(X_trainval_full)
    X_trainval_processed = final_preprocessor.transform(X_trainval_full)
    
    # Create dataset and loader for final training (full dataset)
    trainval_dataset = FeatureDataset(X_trainval_processed, y_trainval_full, feature_names)
    trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=train_params.get("learning_rate", 1e-3),
        weight_decay=train_params.get("weight_decay", 1e-5)
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Train final model (simplified - no redundant validation since CV already validated)
    final_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_features, batch_labels in trainval_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).float()
            
            optimizer.zero_grad()
            logits = final_model(batch_features)
            loss = criterion(logits.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Final model training epoch {epoch + 1}/{epochs}: loss={epoch_loss/len(trainval_loader):.4f}")
    
    # Ensure model is on device
    final_model = final_model.to(device)
    for param in final_model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    for buffer in final_model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    
    # Final evaluation on test set
    test_results = evaluate_model(
        final_model,
        X_test,
        y_test,
        feature_names,
        final_preprocessor,
        batch_size=batch_size,
        device=device,
        use_gpu=use_gpu
    )
    
    # Save results
    results = {
        "model_type": model_type,
        "architecture": architecture,
        "best_params": best_params,
        "best_val_f1": best_score,
        "cv_results": cv_results,
        "test_results": test_results,
        "grid_results": grid_results
    }
    
    # Save model and results
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "model_params": model_params,
        "architecture": architecture,
        "input_dim": input_dim,
        "feature_names": feature_names
    }, output_dir / "model.pt")
    
    # Save preprocessor
    import joblib
    joblib.dump(final_preprocessor, output_dir / "preprocessor.joblib")
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_val_f1": float(best_score),
            "cv_val_f1": float(cv_results["cv_val_f1"]),
            "cv_val_auc": float(cv_results["cv_val_auc"]),
            "test_f1": float(test_results["f1"]),
            "test_auc": float(test_results["auc"]),
            "test_ap": float(test_results["ap"]),
        }, f, indent=2)
    
    # Plot ROC/PR curves
    plot_roc_pr_curves(
        test_results,
        output_dir / "roc_pr_curves.png",
        title=f"{model_type} - ROC and PR Curves"
    )
    
    logger.info(f"Training complete for {model_type}")
    logger.info(f"  CV F1: {cv_results['cv_val_f1']:.4f}")
    logger.info(f"  Test F1: {test_results['f1']:.4f}")
    logger.info(f"  Test AUC: {test_results['auc']:.4f}")
    
    return results

