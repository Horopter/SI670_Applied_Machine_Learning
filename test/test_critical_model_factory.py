"""Critical tests for model factory."""
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from lib.training.model_factory import (
    create_model,
    is_pytorch_model,
    is_xgboost_model,
    get_model_config
)
from lib.mlops.config import RunConfig


class TestModelFactory:
    """Critical tests for model factory."""

    def test_create_logistic_regression(self, temp_dir):
        """Test creating logistic regression model."""
        config = RunConfig(
            run_id="test_123",
            experiment_name="test",
            project_root=str(temp_dir),
            model_type="logistic_regression"
        )
        model = create_model("logistic_regression", config)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_create_svm(self, temp_dir):
        """Test creating SVM model."""
        config = RunConfig(
            run_id="test_123",
            experiment_name="test",
            project_root=str(temp_dir),
            model_type="svm"
        )
        model = create_model("svm", config)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_is_pytorch_model(self):
        """Test PyTorch model detection."""
        assert not is_pytorch_model("logistic_regression")
        assert not is_pytorch_model("svm")
        assert is_pytorch_model("naive_cnn")
        assert is_pytorch_model("vit_gru")
        assert is_pytorch_model("slowfast")

    def test_is_xgboost_model(self):
        """Test XGBoost model detection."""
        assert not is_xgboost_model("logistic_regression")
        assert is_xgboost_model("xgboost_i3d")
        assert is_xgboost_model("xgboost_r2plus1d")
        assert is_xgboost_model("xgboost_pretrained_inception")

    def test_get_model_config(self):
        """Test getting model configuration."""
        config = get_model_config("logistic_regression")
        assert config is not None
        assert "batch_size" in config
        assert config["batch_size"] > 0

    def test_invalid_model_type(self, temp_dir):
        """Test invalid model type raises error."""
        config = RunConfig(
            run_id="test_123",
            experiment_name="test",
            project_root=str(temp_dir),
            model_type="invalid_model"
        )
        with pytest.raises((ValueError, KeyError)):
            create_model("invalid_model", config)

    def test_model_configs_exist(self):
        """Test all model types have configs."""
        model_types = [
            "logistic_regression", "svm", "naive_cnn",
            "vit_gru", "vit_transformer", "slowfast", "x3d",
            "xgboost_i3d", "xgboost_r2plus1d"
        ]
        for model_type in model_types:
            config = get_model_config(model_type)
            assert config is not None
            assert "batch_size" in config

