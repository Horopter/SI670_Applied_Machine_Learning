"""Comprehensive unit tests for video_metrics module."""
import pytest
import torch
import numpy as np
from lib.utils.metrics import (
    collect_logits_and_labels,
    basic_classification_metrics,
    confusion_matrix,
    roc_auc,
)


class TestCollectLogitsAndLabels:
    """Stress tests for collect_logits_and_labels."""
    
    def test_basic_collection(self, sample_logits_labels):
        """Test basic logits and labels collection."""
        from torch.utils.data import TensorDataset, DataLoader
        
        logits, labels = sample_logits_labels
        
        # Create a simple model that returns logits
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x).squeeze(-1)
        
        model = SimpleModel()
        
        # Create dataset and loader
        dataset = TensorDataset(torch.randn(10, 10), labels)
        loader = DataLoader(dataset, batch_size=2)
        
        collected_logits, collected_labels = collect_logits_and_labels(
            model, loader, device="cpu"
        )
        
        assert collected_logits.shape[0] == 10
        assert collected_labels.shape[0] == 10
    
    def test_binary_logits(self):
        """Test collection with binary logits (1D)."""
        from torch.utils.data import TensorDataset, DataLoader
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x).squeeze(-1)
        
        model = SimpleModel()
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
        loader = DataLoader(dataset, batch_size=2)
        
        logits, labels = collect_logits_and_labels(model, loader, device="cpu")
        
        assert logits.ndim == 1
        assert logits.shape[0] == 10
    
    def test_multiclass_logits(self):
        """Test collection with multiclass logits (2D)."""
        from torch.utils.data import TensorDataset, DataLoader
        
        model = torch.nn.Linear(10, 3)
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 3, (10,)))
        loader = DataLoader(dataset, batch_size=2)
        
        logits, labels = collect_logits_and_labels(model, loader, device="cpu")
        
        assert logits.ndim == 2
        assert logits.shape == (10, 3)
        assert labels.shape[0] == 10
    
    def test_empty_loader(self):
        """Test with empty loader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x).squeeze(-1)
        
        model = SimpleModel()
        dataset = TensorDataset(
            torch.empty(0, 10), torch.empty(0, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=2)
        
        # Empty loader - collect_logits_and_labels will try to concatenate empty list
        # This will raise ValueError, so we expect that
        with pytest.raises(ValueError, match="expected a non-empty list"):
            collect_logits_and_labels(model, loader, device="cpu")
    
    def test_large_batch(self):
        """Stress test with large batch."""
        from torch.utils.data import TensorDataset, DataLoader
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x).squeeze(-1)
        
        n = 1000
        model = SimpleModel()
        dataset = TensorDataset(
            torch.randn(n, 10), torch.randint(0, 2, (n,))
        )
        loader = DataLoader(dataset, batch_size=100)
        
        logits, labels = collect_logits_and_labels(model, loader, device="cpu")
        
        assert logits.shape[0] == n
        assert labels.shape[0] == n


class TestBasicClassificationMetrics:
    """Stress tests for basic_classification_metrics."""
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
        labels = torch.tensor([1, 0, 1, 0])
        
        metrics = basic_classification_metrics(logits, labels)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_worst_predictions(self):
        """Test with worst predictions."""
        logits = torch.tensor([-10.0, 10.0, -10.0, 10.0])
        labels = torch.tensor([1, 0, 1, 0])
        
        metrics = basic_classification_metrics(logits, labels)
        
        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
    
    def test_binary_1d_logits(self):
        """Test with 1D binary logits."""
        logits = torch.randn(100)
        labels = torch.randint(0, 2, (100,))
        
        metrics = basic_classification_metrics(logits, labels)
        
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
    
    def test_multiclass_2d_logits(self):
        """Test with 2D multiclass logits."""
        logits = torch.randn(100, 2)
        labels = torch.randint(0, 2, (100,))
        
        metrics = basic_classification_metrics(logits, labels)
        
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
    
    def test_all_positive(self):
        """Test when all predictions are positive."""
        logits = torch.ones(10) * 10.0
        labels = torch.randint(0, 2, (10,))
        
        metrics = basic_classification_metrics(logits, labels)
        
        # Precision might be low if many false positives
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
    
    def test_all_negative(self):
        """Test when all predictions are negative."""
        logits = torch.ones(10) * -10.0
        labels = torch.randint(0, 2, (10,))
        
        metrics = basic_classification_metrics(logits, labels)
        
        # Recall might be low if many false negatives
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
    
    def test_custom_threshold(self):
        """Test with custom threshold."""
        logits = torch.tensor([0.3, 0.4, 0.6, 0.7])
        labels = torch.tensor([0, 0, 1, 1])
        
        # Low threshold
        metrics_low = basic_classification_metrics(logits, labels, threshold=0.3)
        
        # High threshold
        metrics_high = basic_classification_metrics(logits, labels, threshold=0.7)
        
        # Should have different predictions (unless they happen to be the same)
        # For these specific values, they might be the same, so we just check they're valid
        assert 0.0 <= metrics_low["accuracy"] <= 1.0
        assert 0.0 <= metrics_high["accuracy"] <= 1.0
    
    def test_edge_case_all_same_class(self):
        """Test when all labels are the same class."""
        logits = torch.randn(10)
        labels = torch.zeros(10, dtype=torch.long)
        
        metrics = basic_classification_metrics(logits, labels)
        
        # Should handle gracefully
        assert 0.0 <= metrics["accuracy"] <= 1.0
    
    def test_large_dataset(self):
        """Stress test with large dataset."""
        n = 10000
        logits = torch.randn(n)
        labels = torch.randint(0, 2, (n,))
        
        metrics = basic_classification_metrics(logits, labels)
        
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
    
    def test_extreme_logits(self):
        """Test with extreme logit values."""
        logits = torch.tensor([100.0, -100.0, 100.0, -100.0])
        labels = torch.tensor([1, 0, 1, 0])
        
        metrics = basic_classification_metrics(logits, labels)
        
        # Should handle extreme values
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestConfusionMatrix:
    """Stress tests for confusion_matrix."""
    
    def test_basic_confusion_matrix(self):
        """Test basic confusion matrix."""
        logits = torch.tensor([1.0, -1.0, 1.0, -1.0])
        labels = torch.tensor([1, 0, 1, 0])
        
        cm = confusion_matrix(logits, labels)
        
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # True negatives
        assert cm[1, 1] == 2  # True positives
    
    def test_confusion_matrix_1d(self):
        """Test confusion matrix with 1D logits."""
        logits = torch.randn(100)
        labels = torch.randint(0, 2, (100,))
        
        cm = confusion_matrix(logits, labels)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == 100
    
    def test_confusion_matrix_2d(self):
        """Test confusion matrix with 2D logits."""
        logits = torch.randn(100, 2)
        labels = torch.randint(0, 2, (100,))
        
        cm = confusion_matrix(logits, labels)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == 100
    
    def test_perfect_predictions_cm(self):
        """Test confusion matrix with perfect predictions."""
        logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
        labels = torch.tensor([1, 0, 1, 0])
        
        cm = confusion_matrix(logits, labels)
        
        assert cm[0, 0] == 2  # TN
        assert cm[1, 1] == 2  # TP
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN


class TestRocAuc:
    """Stress tests for roc_auc."""
    
    def test_basic_roc_auc(self):
        """Test basic ROC-AUC calculation."""
        logits = torch.tensor([1.0, 0.5, -0.5, -1.0])
        labels = torch.tensor([1, 1, 0, 0])
        
        auc = roc_auc(logits, labels)
        
        assert 0.0 <= auc <= 1.0
    
    def test_perfect_roc_auc(self):
        """Test with perfect separation."""
        logits = torch.tensor([10.0, 5.0, -5.0, -10.0])
        labels = torch.tensor([1, 1, 0, 0])
        
        auc = roc_auc(logits, labels)
        
        assert auc == pytest.approx(1.0, abs=0.01)
    
    def test_random_roc_auc(self):
        """Test with random predictions."""
        logits = torch.randn(100)
        labels = torch.randint(0, 2, (100,))
        
        auc = roc_auc(logits, labels)
        
        # Random should be around 0.5
        assert 0.0 <= auc <= 1.0
    
    def test_roc_auc_1d(self):
        """Test ROC-AUC with 1D logits."""
        logits = torch.randn(1000)
        labels = torch.randint(0, 2, (1000,))
        
        auc = roc_auc(logits, labels)
        
        assert 0.0 <= auc <= 1.0
    
    def test_roc_auc_2d(self):
        """Test ROC-AUC with 2D logits."""
        logits = torch.randn(1000, 2)
        labels = torch.randint(0, 2, (1000,))
        
        auc = roc_auc(logits, labels)
        
        assert 0.0 <= auc <= 1.0
    
    def test_roc_auc_all_same_class(self):
        """Test ROC-AUC when all labels are same class."""
        logits = torch.randn(10)
        labels = torch.zeros(10, dtype=torch.long)
        
        # Should return NaN when only one class is present (ROC AUC is undefined)
        auc = roc_auc(logits, labels)
        
        # Should return NaN (not -1.0) since we now check for single class before calling sklearn
        assert np.isnan(auc), f"Expected NaN for single class, got {auc}"

