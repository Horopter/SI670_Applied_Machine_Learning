"""Critical tests for metrics computation."""
import pytest
import numpy as np
from lib.training.metrics_utils import compute_classification_metrics


class TestMetricsComputation:
    """Critical tests for metrics computation."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.15])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert metrics["val_acc"] == 1.0
        assert metrics["val_f1"] == 1.0
        assert metrics["val_precision"] == 1.0
        assert metrics["val_recall"] == 1.0
        assert not np.isnan(metrics["val_loss"])

    def test_worst_predictions(self):
        """Test metrics with worst predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.85])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert metrics["val_acc"] == 0.0
        assert metrics["val_f1"] == 0.0
        assert not np.isnan(metrics["val_loss"])

    def test_balanced_predictions(self):
        """Test metrics with balanced predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_probs = np.array([0.2, 0.8, 0.6, 0.4, 0.3, 0.7])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert 0.0 <= metrics["val_acc"] <= 1.0
        assert 0.0 <= metrics["val_f1"] <= 1.0
        assert 0.0 <= metrics["val_precision"] <= 1.0
        assert 0.0 <= metrics["val_recall"] <= 1.0
        assert not np.isnan(metrics["val_loss"])

    def test_all_zeros(self):
        """Test metrics when all predictions are 0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        y_probs = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.2])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert metrics["val_acc"] >= 0.0
        assert metrics["val_f1"] >= 0.0
        assert not np.isnan(metrics["val_loss"])

    def test_all_ones(self):
        """Test metrics when all predictions are 1."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        y_probs = np.array([0.9, 0.8, 0.85, 0.95, 0.9, 0.92])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert metrics["val_acc"] >= 0.0
        assert metrics["val_f1"] >= 0.0
        assert not np.isnan(metrics["val_loss"])

    def test_no_probabilities(self):
        """Test metrics without probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        metrics = compute_classification_metrics(y_true, y_pred, None)
        
        assert metrics["val_acc"] == 1.0
        assert np.isnan(metrics["val_loss"])

    def test_per_class_metrics(self):
        """Test per-class metrics computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_probs = np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.4])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        
        assert "val_f1_class0" in metrics
        assert "val_f1_class1" in metrics
        assert "val_precision_class0" in metrics
        assert "val_precision_class1" in metrics
        assert "val_recall_class0" in metrics
        assert "val_recall_class1" in metrics

    def test_empty_inputs(self):
        """Test metrics with empty inputs."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            compute_classification_metrics(y_true, y_pred, None)

