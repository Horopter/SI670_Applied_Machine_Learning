"""
Hyperparameter grid search configuration.

Defines hyperparameter grids for different model types.
Grid search is performed on a sample of data for efficiency,
then best hyperparameters are used for final training on full dataset.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
from itertools import product
import numpy as np

logger = logging.getLogger(__name__)


def get_hyperparameter_grid(model_type: str) -> Dict[str, List[Any]]:
    """
    Get hyperparameter grid for a model type.
    
    Args:
        model_type: Model type identifier
    
    Returns:
        Dictionary mapping parameter names to lists of values to try
    """
    grids = {
        # Baseline models (feature-based)
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000]
        },
        "logistic_regression_stage2": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000]
        },
        "logistic_regression_stage2_stage4": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000]
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"]
        },
        "svm_stage2": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"]
        },
        "svm_stage2_stage4": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"]
        },
        # PyTorch models
        "naive_cnn": {
            "learning_rate": [1e-4, 5e-4, 1e-3],  # Reduced from 4 to 3 values (3*3*2*2 = 36 combinations)
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [1],  # Capped at 1 to prevent OOM (processes 1000 frames at full resolution)
            "num_epochs": [20, 30]  # Reduced from 3 to 2 values
        },
        "pretrained_inception": {
            "learning_rate": [5e-5, 1e-4, 5e-4],  # Reduced from 4 to 3 values (3*2*2*2*2 = 48 combinations)
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]  # Capped at 2 to prevent OOM (large pretrained model processing many frames)
        },
        "variable_ar_cnn": {
            "learning_rate": [1e-4, 5e-4],  # Reduced from 3 to 2 values (2*2*2 = 8 combinations)
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]  # Capped at 2 to prevent OOM (processes variable-length videos with many frames)
        },
        "vit_gru": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values (2*2*2*2*2 = 32 combinations)
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "vit_transformer": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "slowfast": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            # batch_size removed - x3d requires batch_size=1 to prevent OOM (enforced in pipeline.py)
            # gradient_accumulation_steps will be adjusted to maintain effective batch size
        },
        "x3d": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "i3d": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        "r2plus1d": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        "r3d_18": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "timesformer": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values (2*2*2*2*2 = 32 combinations)
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "vivit": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "two_stream": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values (2*2*2*2*1 = 16 combinations)
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        "slowfast_attention": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        "slowfast_multiscale": {
            "learning_rate": [5e-5, 1e-4],  # Reduced from 3 to 2 values (2*2*2*2*2 = 32 combinations)
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        # XGBoost models - MINIMAL GRID SEARCH for fastest results
        # Reduced to single values for speed (can be expanded later if needed)
        # Total combinations: 1*1*1*1*1 = 1 (no grid search, just default params)
        "xgboost_pretrained_inception": {
            "n_estimators": [100],  # Single value for speed (was [100, 200])
            "max_depth": [5],  # Single value for speed (was [3, 5, 7])
            "learning_rate": [0.1],  # Single value for speed (was [0.01, 0.1])
            "subsample": [0.8],  # Single value for speed (was [0.8, 1.0])
            "colsample_bytree": [0.8]  # Single value for speed (was [0.8, 1.0])
        },
        "xgboost_i3d": {
            "n_estimators": [100],  # Single value for speed
            "max_depth": [5],  # Single value for speed
            "learning_rate": [0.1],  # Single value for speed
            "subsample": [0.8],  # Single value for speed
            "colsample_bytree": [0.8]  # Single value for speed
        },
        "xgboost_r2plus1d": {
            "n_estimators": [100],  # Single value for speed
            "max_depth": [5],  # Single value for speed
            "learning_rate": [0.1],  # Single value for speed
            "subsample": [0.8],  # Single value for speed
            "colsample_bytree": [0.8]  # Single value for speed
        },
        "xgboost_vit_gru": {
            "n_estimators": [100],  # Single value for speed
            "max_depth": [5],  # Single value for speed
            "learning_rate": [0.1],  # Single value for speed
            "subsample": [0.8],  # Single value for speed
            "colsample_bytree": [0.8]  # Single value for speed
        },
        "xgboost_vit_transformer": {
            "n_estimators": [100],  # Single value for speed
            "max_depth": [5],  # Single value for speed
            "learning_rate": [0.1],  # Single value for speed
            "subsample": [0.8],  # Single value for speed
            "colsample_bytree": [0.8]  # Single value for speed
        }
    }
    
    return grids.get(model_type, {})


def generate_parameter_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from a grid.
    
    Args:
        grid: Dictionary mapping parameter names to lists of values
    
    Returns:
        List of parameter dictionaries
    """
    if not grid:
        return [{}]
    
    keys = list(grid.keys())
    values = list(grid.values())
    
    combinations = []
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)
    
    return combinations


def select_best_hyperparameters(
    model_type: str,
    grid_results: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Select best hyperparameters based on cross-validation results.
    
    Args:
        model_type: Model type identifier
        grid_results: List of grid search results, each containing:
            - "mean_f1": mean F1 score across folds
            - "mean_acc": mean accuracy across folds
            - "fold_results": list of fold results
            - hyperparameter keys: the actual hyperparameter values
    
    Returns:
        Dictionary of best hyperparameters, or None if no valid results
    """
    if not grid_results:
        return None
    
    # Filter out invalid results (those with NaN or missing mean_f1)
    valid_results = [
        r for r in grid_results
        if isinstance(r.get("mean_f1", None), (int, float)) and not np.isnan(r.get("mean_f1", 0))
    ]
    
    if not valid_results:
        return None
    
    # Find best parameter combination (highest mean F1 score)
    best_params = None
    best_mean_f1 = -1
    
    for result in valid_results:
        mean_f1 = result.get("mean_f1", 0)
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            # Extract hyperparameters (exclude metrics and fold_results)
            best_params = {k: v for k, v in result.items() 
                          if k not in ["mean_f1", "mean_acc", "fold_results"]}
    
    logger.info(f"Best hyperparameters (mean F1: {best_mean_f1:.4f}): {best_params}")
    
    return best_params
