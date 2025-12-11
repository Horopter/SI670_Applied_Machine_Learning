"""
Grid search hyperparameter tuning for all model types.

Performs grid search with cross-validation to find best hyperparameters.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional
from itertools import product
import numpy as np

logger = logging.getLogger(__name__)


def get_hyperparameter_grid(model_type: str) -> Dict[str, List[Any]]:
    """
    Get hyperparameter grid for grid search based on model type.
    
    Args:
        model_type: Model type identifier
    
    Returns:
        Dictionary mapping hyperparameter names to lists of values to try
    """
    grids = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "max_iter": [1000, 2000]
        },
        "logistic_regression_stage2": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "max_iter": [1000, 2000]
        },
        "logistic_regression_stage2_stage4": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "max_iter": [1000, 2000]
        },
        "svm": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "degree": [2, 3, 4]  # Only used for poly kernel
        },
        "svm_stage2": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "max_iter": [1000, 2000]
        },
        "svm_stage2_stage4": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "max_iter": [1000, 2000]
        },
        "naive_cnn": {
            "learning_rate": [1e-5, 1e-4, 5e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [4, 8, 16],
            "num_epochs": [20, 30, 40]
        },
        "pretrained_inception": {
            "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
            "backbone_lr": [1e-6, 5e-6, 1e-5],
            "head_lr": [1e-4, 5e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [4, 8]
        },
        "variable_ar_cnn": {
            "learning_rate": [1e-5, 1e-4, 5e-4],
            "weight_decay": [1e-5, 1e-4],
            "batch_size": [2, 4]
        },
        "i3d": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "r2plus1d": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "x3d": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "slowfast": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "slowfast_attention": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        "slowfast_multiscale": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        "two_stream": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        "vit_gru": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "vit_transformer": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2, 4]
        },
        "timesformer": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        "vivit": {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "backbone_lr": [1e-6, 5e-6],
            "head_lr": [1e-4, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [2]
        },
        # XGBoost models
        "xgboost_pretrained_inception": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },
        "xgboost_i3d": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },
        "xgboost_r2plus1d": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },
        "xgboost_vit_gru": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },
        "xgboost_vit_transformer": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
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
    fold_results: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Select best hyperparameters based on cross-validation results.
    
    Args:
        model_type: Model type identifier
        fold_results: List of results from each fold, each containing hyperparameters and metrics
    
    Returns:
        Dictionary of best hyperparameters, or None if no valid results
    """
    if not fold_results:
        return None
    
    # Filter out invalid results
    valid_results = [
        r for r in fold_results
        if isinstance(r.get("val_f1", None), (int, float)) and not np.isnan(r.get("val_f1", 0))
    ]
    
    if not valid_results:
        return None
    
    # Group by hyperparameter combination
    param_groups = {}
    for result in valid_results:
        # Extract hyperparameters (exclude metrics)
        params = {k: v for k, v in result.items() 
                 if k not in ["fold", "val_loss", "val_acc", "val_f1", "val_precision", 
                             "val_recall", "val_f1_class0", "val_precision_class0", 
                             "val_recall_class0", "val_f1_class1", "val_precision_class1", 
                             "val_recall_class1"]}
        
        # Create hashable key from params
        param_key = tuple(sorted(params.items()))
        
        if param_key not in param_groups:
            param_groups[param_key] = {
                "params": params,
                "f1_scores": [],
                "acc_scores": []
            }
        
        param_groups[param_key]["f1_scores"].append(result.get("val_f1", 0))
        param_groups[param_key]["acc_scores"].append(result.get("val_acc", 0))
    
    # Find best parameter combination (highest mean F1 score)
    best_params = None
    best_mean_f1 = -1
    
    for param_key, group in param_groups.items():
        mean_f1 = np.mean(group["f1_scores"])
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_params = group["params"]
    
    logger.info(f"Best hyperparameters (mean F1: {best_mean_f1:.4f}): {best_params}")
    
    return best_params

