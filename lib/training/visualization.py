"""
Visualization utilities for model training results.

Generates graphs for:
- Learning curves (train/val loss and accuracy)
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Cross-validation fold comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Learning Curves"
) -> None:
    """
    Plot learning curves (loss and accuracy over epochs).
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Training accuracy per epoch (optional)
        val_accs: Validation accuracy per epoch (optional)
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=(15, 5))
    if train_accs is None:
        axes = [axes]
    
    # Plot loss
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot accuracy if provided
    if train_accs is not None:
        ax = axes[1]
        ax.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved learning curves to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
            transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve"
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    title: str = "Precision-Recall Curve"
) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
        title: Plot title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Precision-Recall curve to {save_path}")


def plot_cv_fold_comparison(
    fold_results: List[Dict[str, Any]],
    save_path: Path,
    title: str = "Cross-Validation Fold Comparison"
) -> None:
    """
    Plot comparison of metrics across CV folds.
    
    Args:
        fold_results: List of results from each fold
        save_path: Path to save the plot
        title: Plot title
    """
    if not fold_results:
        logger.warning("No fold results to plot")
        return
    
    # Extract metrics
    folds = [r.get("fold", i+1) for i, r in enumerate(fold_results)]
    val_accs = [r.get("val_acc", 0) for r in fold_results]
    val_f1s = [r.get("val_f1", 0) for r in fold_results]
    val_precisions = [r.get("val_precision", 0) for r in fold_results]
    val_recalls = [r.get("val_recall", 0) for r in fold_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = [
        (val_accs, "Accuracy", axes[0, 0]),
        (val_f1s, "F1 Score", axes[0, 1]),
        (val_precisions, "Precision", axes[1, 0]),
        (val_recalls, "Recall", axes[1, 1])
    ]
    
    for values, metric_name, ax in metrics:
        ax.bar(folds, values, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Fold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for fold, value in zip(folds, values):
            if not np.isnan(value):
                ax.text(fold, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        # Add mean line
        mean_val = np.nanmean(values)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CV fold comparison to {save_path}")


def plot_hyperparameter_search(
    search_results: List[Dict[str, Any]],
    save_path: Path,
    title: str = "Hyperparameter Search Results"
) -> None:
    """
    Plot hyperparameter search results.
    
    Args:
        search_results: List of results from grid search
        save_path: Path to save the plot
        title: Plot title
    """
    if not search_results:
        logger.warning("No search results to plot")
        return
    
    # Extract F1 scores and parameter combinations
    f1_scores = [r.get("mean_f1", 0) for r in search_results]
    
    # Create bar plot of top N results
    top_n = min(10, len(search_results))
    sorted_results = sorted(search_results, key=lambda x: x.get("mean_f1", 0), reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(top_n)
    scores = [r.get("mean_f1", 0) for r in sorted_results]
    labels = [f"Config {i+1}" for i in range(top_n)]
    
    bars = ax.barh(x_pos, scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean F1 Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, i, f'{score:.4f}', 
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved hyperparameter search results to {save_path}")


def generate_all_plots(
    model_type: str,
    fold_results: List[Dict[str, Any]],
    output_dir: Path,
    train_history: Optional[Dict[str, List[float]]] = None,
    predictions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Generate all plots for a model.
    
    Args:
        model_type: Model type identifier
        fold_results: List of results from each fold
        output_dir: Directory to save plots
        train_history: Training history (train_loss, val_loss, train_acc, val_acc)
        predictions: Tuple of (y_true, y_pred, y_proba) for best fold
        class_names: List of class names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if class_names is None:
        class_names = ["Real", "Fake"]
    
    # Plot CV fold comparison
    plot_cv_fold_comparison(
        fold_results,
        output_dir / "cv_fold_comparison.png",
        title=f"{model_type} - Cross-Validation Results"
    )
    
    # Plot learning curves if available
    if train_history:
        plot_learning_curves(
            train_history.get("train_loss", []),
            train_history.get("val_loss", []),
            train_history.get("train_acc"),
            train_history.get("val_acc"),
            output_dir / "learning_curves.png",
            title=f"{model_type} - Learning Curves"
        )
    
    # Plot confusion matrix, ROC, PR if predictions available
    if predictions:
        y_true, y_pred, y_proba = predictions
        
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            output_dir / "confusion_matrix.png",
            title=f"{model_type} - Confusion Matrix"
        )
        
        if y_proba is not None and len(y_proba.shape) > 1:
            # Multi-class: use positive class probabilities
            y_proba_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
        else:
            y_proba_pos = y_proba
        
        plot_roc_curve(
            y_true, y_proba_pos,
            output_dir / "roc_curve.png",
            title=f"{model_type} - ROC Curve"
        )
        
        plot_precision_recall_curve(
            y_true, y_proba_pos,
            output_dir / "precision_recall_curve.png",
            title=f"{model_type} - Precision-Recall Curve"
        )
    
    logger.info(f"Generated all plots for {model_type} in {output_dir}")


def plot_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    fold_output_dir: Path,
    fold_num: int,
    model_type: str,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Generate plots for a single fold: ROC curve, PR curve, and confusion matrix.
    
    Args:
        y_true: True labels (1D array)
        y_pred: Predicted labels (1D array)
        y_probs: Predicted probabilities (2D array with shape (n_samples, n_classes) or 1D for binary)
        fold_output_dir: Directory to save plots
        fold_num: Fold number
        model_type: Model type identifier
        class_names: List of class names (default: ["Real", "Fake"])
    """
    if class_names is None:
        class_names = ["Real", "Fake"]
    
    fold_output_dir = Path(fold_output_dir)
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract positive class probabilities for binary classification
        if y_probs.ndim == 2:
            if y_probs.shape[1] == 2:
                y_proba_pos = y_probs[:, 1]  # Probability of positive class
            else:
                # Multi-class: use max probability class
                y_proba_pos = np.max(y_probs, axis=1)
        else:
            # Already 1D
            y_proba_pos = y_probs
        
        # Combined ROC and PR curves plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
        avg_precision = average_precision_score(y_true, y_proba_pos)
        ax2.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        
        # Add random guess baseline (horizontal line at positive class prevalence)
        positive_prevalence = np.mean(y_true)
        ax2.axhline(y=positive_prevalence, color='navy', lw=2, linestyle='--', 
                   label=f'Random Guess (prevalence = {positive_prevalence:.4f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_type} - Fold {fold_num} - ROC and PR Curves', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fold_output_dir / "roc_pr_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{model_type} - Fold {fold_num} - Confusion Matrix', 
                     fontsize=14, fontweight='bold')
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
                transform=ax.transAxes, ha='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(fold_output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated fold {fold_num} plots: ROC/PR curves and confusion matrix")
        
    except Exception as e:
        logger.warning(f"Failed to generate plots for fold {fold_num}: {e}", exc_info=True)


def plot_feature_importance(
    feature_importance: Dict[str, float],
    save_path: Path,
    top_n: int = 20,
    title: str = "Feature Importance"
) -> None:
    """
    Plot feature importance for tree-based models (XGBoost, Gradient Boosting).
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        save_path: Path to save the plot
        top_n: Number of top features to display
        title: Plot title
    """
    if not feature_importance:
        logger.warning("No feature importance data provided")
        return
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features) if top_features else ([], [])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to {save_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Plot)"
) -> None:
    """
    Plot calibration curve (reliability plot) to assess probability calibration.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
        n_bins: Number of bins for calibration
        title: Plot title
    """
    from sklearn.calibration import calibration_curve
    
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba.flatten() if y_proba.ndim > 1 else y_proba
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba_pos, n_bins=n_bins, strategy='uniform'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model', linewidth=2, markersize=8)
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved calibration curve to {save_path}")


def plot_per_class_metrics(
    metrics_per_class: Dict[str, Dict[str, float]],
    save_path: Path,
    title: str = "Per-Class Metrics Breakdown"
) -> None:
    """
    Plot per-class metrics (precision, recall, F1) breakdown.
    
    Args:
        metrics_per_class: Dictionary with class names as keys and metrics dict as values
        save_path: Path to save the plot
        title: Plot title
    """
    if not metrics_per_class:
        logger.warning("No per-class metrics data provided")
        return
    
    classes = list(metrics_per_class.keys())
    metrics_names = ['precision', 'recall', 'f1']
    
    # Extract metrics for each class
    precision_vals = [metrics_per_class[cls].get('precision', 0) for cls in classes]
    recall_vals = [metrics_per_class[cls].get('recall', 0) for cls in classes]
    f1_vals = [metrics_per_class[cls].get('f1', 0) for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
    ax.bar(x, recall_vals, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_vals, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-class metrics plot to {save_path}")


def plot_hyperparameter_sensitivity(
    hyperparam_results: List[Dict[str, Any]],
    param_name: str,
    save_path: Path,
    metric_name: str = 'val_f1',
    title: Optional[str] = None
) -> None:
    """
    Plot hyperparameter sensitivity analysis.
    
    Args:
        hyperparam_results: List of dictionaries with hyperparameter values and metrics
        param_name: Name of hyperparameter to analyze
        save_path: Path to save the plot
        metric_name: Metric to plot (default: 'val_f1')
        title: Plot title (auto-generated if None)
    """
    if not hyperparam_results:
        logger.warning("No hyperparameter results provided")
        return
    
    param_values = []
    metric_values = []
    
    for result in hyperparam_results:
        if param_name in result.get('params', {}):
            param_values.append(result['params'][param_name])
            metric_values.append(result.get(metric_name, 0))
    
    if not param_values:
        logger.warning(f"Hyperparameter '{param_name}' not found in results")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(title or f"Hyperparameter Sensitivity: {param_name}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved hyperparameter sensitivity plot to {save_path}")


def plot_model_comparison_matrix(
    model_results: Dict[str, Dict[str, float]],
    metric_name: str = 'val_f1',
    save_path: Path = None,
    title: str = "Model Comparison Matrix"
) -> None:
    """
    Plot model comparison matrix heatmap.
    
    Args:
        model_results: Dictionary with model names as keys and metrics dict as values
        metric_name: Metric to compare (default: 'val_f1')
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    if not model_results:
        logger.warning("No model results provided")
        return
    
    model_names = list(model_results.keys())
    metric_values = [model_results[model].get(metric_name, 0) for model in model_names]
    
    # Create a matrix (1 row, multiple columns for comparison)
    matrix = np.array(metric_values).reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.5), 4))
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt='.4f', 
        cmap='YlOrRd',
        xticklabels=model_names,
        yticklabels=[metric_name.replace('_', ' ').title()],
        ax=ax,
        cbar_kws={'label': metric_name.replace('_', ' ').title()}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison matrix to {save_path}")
    plt.close()


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    title: str = "Error Analysis"
) -> None:
    """
    Plot error analysis showing misclassification patterns.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
        title: Plot title
    """
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba.flatten() if y_proba.ndim > 1 else y_proba
    
    errors = y_true != y_pred
    correct = ~errors
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Probability distribution for correct vs incorrect predictions
    ax = axes[0]
    ax.hist(y_proba_pos[correct], bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
    ax.hist(y_proba_pos[errors], bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Probability Distribution: Correct vs Incorrect', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error types breakdown
    ax = axes[1]
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    
    error_types = ['False Positives', 'False Negatives', 'True Positives', 'True Negatives']
    error_counts = [false_positives, false_negatives, true_positives, true_negatives]
    colors = ['red', 'orange', 'green', 'blue']
    
    ax.bar(error_types, error_counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Types Breakdown', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved error analysis plot to {save_path}")


def plot_statistical_significance(
    model_metrics: Dict[str, List[float]],
    save_path: Path,
    metric_name: str = 'val_f1',
    title: str = "Statistical Significance Test"
) -> None:
    """
    Plot statistical significance test results (e.g., t-test, Mann-Whitney U).
    
    Args:
        model_metrics: Dictionary with model names as keys and list of metric values (across folds) as values
        save_path: Path to save the plot
        metric_name: Metric name for display
        title: Plot title
    """
    from scipy import stats
    
    if len(model_metrics) < 2:
        logger.warning("Need at least 2 models for statistical significance test")
        return
    
    model_names = list(model_metrics.keys())
    n_models = len(model_names)
    
    # Perform pairwise comparisons
    p_values = np.zeros((n_models, n_models))
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                p_values[i, j] = 1.0
            else:
                # Perform Mann-Whitney U test (non-parametric)
                stat, p_val = stats.mannwhitneyu(
                    model_metrics[model1], 
                    model_metrics[model2], 
                    alternative='two-sided'
                )
                p_values[i, j] = p_val
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(p_values, cmap='RdYlGn', vmin=0, vmax=0.05)
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            text = ax.text(j, i, f'{p_values[i, j]:.4f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Model (Column)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model (Row)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('p-value', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved statistical significance plot to {save_path}")

