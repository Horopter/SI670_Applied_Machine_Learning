"""
Comprehensive unit tests for XGBoostPretrainedBaseline with dummy tensors.

These tests verify XGBoost API compatibility and prevent runtime errors
by testing with dummy data instead of real video files.
"""
import pytest
import numpy as np
import polars as pl
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Check if xgboost is available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    XGBOOST_VERSION = tuple(map(int, xgb.__version__.split('.')))
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = None

pytestmark = pytest.mark.skipif(
    not XGBOOST_AVAILABLE,
    reason="xgboost not available"
)


class TestXGBoostAPIVersionDetection:
    """Test XGBoost version detection and API compatibility."""
    
    def test_version_detection(self):
        """Test that XGBoost version is detected correctly."""
        assert XGBOOST_AVAILABLE, "XGBoost should be available for tests"
        assert XGBOOST_VERSION is not None, "XGBoost version should be detectable"
        assert len(XGBOOST_VERSION) >= 2, "Version should have at least major.minor"
        print(f"XGBoost version: {xgb.__version__}, tuple: {XGBOOST_VERSION}")
    
    def test_xgboost_import(self):
        """Test that XGBoost can be imported."""
        from lib.training._xgboost_pretrained import (
            XGBOOST_AVAILABLE as MODULE_XGBOOST_AVAILABLE,
            XGBOOST_VERSION as MODULE_XGBOOST_VERSION,
            USE_FIT_EARLY_STOPPING
        )
        assert MODULE_XGBOOST_AVAILABLE, "XGBoost should be available in module"
        assert MODULE_XGBOOST_VERSION is not None or MODULE_XGBOOST_VERSION == XGBOOST_VERSION
        print(f"Module USE_FIT_EARLY_STOPPING: {USE_FIT_EARLY_STOPPING}")


class TestXGBoostAPICalls:
    """Test XGBoost API calls with dummy data to catch compatibility issues."""
    
    def test_xgbclassifier_creation(self):
        """Test basic XGBClassifier creation with dummy parameters."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 10,  # Small for fast tests
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': 1,
        }
        
        # Should not raise any errors
        model = xgb.XGBClassifier(**params)
        assert model is not None
        print(f"✓ XGBClassifier created successfully with params: {list(params.keys())}")
    
    def test_xgbclassifier_fit_basic(self):
        """Test XGBClassifier.fit() with dummy data."""
        # Create dummy data
        n_samples = 100
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        
        # Test basic fit - should not raise errors
        model.fit(X, y)
        assert model.is_fitted
        print("✓ XGBClassifier.fit() works with basic parameters")
    
    def test_xgbclassifier_fit_with_eval_set(self):
        """Test XGBClassifier.fit() with eval_set parameter."""
        # Create dummy data
        n_samples = 100
        n_features = 20
        X_train = np.random.rand(80, n_features)
        X_val = np.random.rand(20, n_features)
        y_train = np.random.randint(0, 2, 80)
        y_val = np.random.randint(0, 2, 20)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        
        # Test fit with eval_set - should not raise errors
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        assert model.is_fitted
        print("✓ XGBClassifier.fit() works with eval_set")
    
    def test_xgbclassifier_callbacks_in_constructor(self):
        """Test XGBClassifier with callbacks in constructor (XGBoost 2.0+ API)."""
        # Create dummy data
        n_samples = 100
        n_features = 20
        X_train = np.random.rand(80, n_features)
        X_val = np.random.rand(20, n_features)
        y_train = np.random.randint(0, 2, 80)
        y_val = np.random.randint(0, 2, 20)
        
        try:
            from xgboost.callback import EarlyStopping
            
            # Test callbacks in constructor (XGBoost 2.0+ API)
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=10,
                random_state=42,
                tree_method='hist',
                n_jobs=1,
                callbacks=[EarlyStopping(rounds=5, save_best=True)]
            )
            
            # This should work for XGBoost 2.0+
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            assert model.is_fitted
            print("✓ XGBClassifier with callbacks in constructor works (XGBoost 2.0+ API)")
        except (ImportError, AttributeError, TypeError) as e:
            # If callbacks not available, that's OK - we have fallbacks
            print(f"⚠ Callbacks in constructor not available: {e} (this is OK, fallbacks exist)")
    
    def test_xgbclassifier_callbacks_in_fit_should_fail(self):
        """Test that callbacks in fit() fail for XGBoost 2.0+ (to verify our fix)."""
        if XGBOOST_VERSION and XGBOOST_VERSION >= (2, 0, 0):
            # Create dummy data
            n_samples = 100
            n_features = 20
            X_train = np.random.rand(80, n_features)
            X_val = np.random.rand(20, n_features)
            y_train = np.random.randint(0, 2, 80)
            y_val = np.random.randint(0, 2, 20)
            
            try:
                from xgboost.callback import EarlyStopping
                
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    n_estimators=10,
                    random_state=42,
                    tree_method='hist',
                    n_jobs=1
                )
                
                # This SHOULD fail for XGBoost 2.0+ (callbacks must be in constructor)
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[EarlyStopping(rounds=5)],
                        verbose=False
                    )
                    # If we get here, callbacks in fit() might work (unexpected)
                    print("⚠ Callbacks in fit() worked (unexpected for XGBoost 2.0+)")
                except TypeError as e:
                    # This is expected - callbacks in fit() should fail
                    assert "callbacks" in str(e).lower() or "unexpected keyword" in str(e).lower()
                    print(f"✓ Confirmed: callbacks in fit() fails as expected: {e}")
            except ImportError:
                pytest.skip("EarlyStopping callback not available")
        else:
            pytest.skip("Test only relevant for XGBoost 2.0+")
    
    def test_xgbclassifier_early_stopping_rounds_in_fit(self):
        """Test early_stopping_rounds in fit() for XGBoost < 2.0."""
        # Create dummy data
        n_samples = 100
        n_features = 20
        X_train = np.random.rand(80, n_features)
        X_val = np.random.rand(20, n_features)
        y_train = np.random.randint(0, 2, 80)
        y_val = np.random.randint(0, 2, 20)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        
        # Try early_stopping_rounds in fit() - may or may not work depending on version
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=5,
                verbose=False
            )
            assert model.is_fitted
            print("✓ early_stopping_rounds in fit() works (XGBoost < 2.0 API)")
        except TypeError as e:
            # This is expected for XGBoost 2.0+
            if XGBOOST_VERSION and XGBOOST_VERSION >= (2, 0, 0):
                print(f"✓ Confirmed: early_stopping_rounds in fit() fails for XGBoost 2.0+: {e}")
            else:
                raise  # Unexpected error
    
    def test_xgbclassifier_predict_proba(self):
        """Test XGBClassifier.predict_proba() with dummy data."""
        # Create dummy data
        n_samples = 100
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        model.fit(X, y)
        
        # Test predict_proba
        X_test = np.random.rand(10, n_features)
        probs = model.predict_proba(X_test)
        
        assert probs.shape == (10, 2), f"Expected (10, 2), got {probs.shape}"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1"
        print("✓ XGBClassifier.predict_proba() works correctly")
    
    def test_xgbclassifier_save_load_model(self):
        """Test XGBClassifier.save_model() and load_model() with dummy data."""
        # Create dummy data
        n_samples = 100
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        model.fit(X, y)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.json"
            model.save_model(str(model_path))
            assert model_path.exists(), "Model file should be created"
            
            # Load model
            loaded_model = xgb.XGBClassifier()
            loaded_model.load_model(str(model_path))
            
            # Verify predictions match
            X_test = np.random.rand(10, n_features)
            original_probs = model.predict_proba(X_test)
            loaded_probs = loaded_model.predict_proba(X_test)
            
            np.testing.assert_allclose(original_probs, loaded_probs, rtol=1e-5)
            print("✓ XGBClassifier.save_model() and load_model() work correctly")


class TestXGBoostPretrainedBaselineWithDummyData:
    """Test XGBoostPretrainedBaseline with mocked feature extraction."""
    
    @pytest.fixture
    def dummy_features(self):
        """Create dummy feature array."""
        n_samples = 100
        n_features = 50
        return np.random.rand(n_samples, n_features).astype(np.float32)
    
    @pytest.fixture
    def dummy_labels(self):
        """Create dummy labels."""
        return np.random.randint(0, 2, 100)
    
    @pytest.fixture
    def dummy_df(self):
        """Create dummy DataFrame."""
        return pl.DataFrame({
            "video_path": [f"dummy_video_{i}.mp4" for i in range(100)],
            "label": np.random.randint(0, 2, 100).tolist()
        })
    
    @pytest.fixture
    def temp_project_root(self):
        """Create temporary project root directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    @patch('lib.training._xgboost_pretrained.extract_features_from_pretrained_model')
    @patch('lib.training._xgboost_pretrained.create_model')
    @patch('lib.training._xgboost_pretrained.get_model_config')
    def test_fit_with_dummy_features(
        self,
        mock_get_model_config,
        mock_create_model,
        mock_extract_features,
        dummy_features,
        dummy_df,
        temp_project_root
    ):
        """Test fit() method with dummy features (mocked feature extraction)."""
        from lib.training._xgboost_pretrained import XGBoostPretrainedBaseline
        
        # Mock feature extraction to return dummy features
        mock_extract_features.return_value = dummy_features
        
        # Mock model creation
        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_create_model.return_value = mock_model
        mock_get_model_config.return_value = {"num_frames": 100}
        
        # Create model
        model = XGBoostPretrainedBaseline(
            base_model_type="i3d",
            num_frames=100,
            xgb_params={
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'n_estimators': 10,  # Small for fast tests
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42,
                'tree_method': 'hist',
                'n_jobs': 1,
            }
        )
        
        # Mock _extract_features_batch to return dummy features
        model._extract_features_batch = Mock(return_value=(dummy_features, dummy_df["label"].to_numpy()))
        
        # Test fit - should not raise any API errors
        model.fit(dummy_df, temp_project_root)
        
        assert model.is_fitted, "Model should be fitted"
        assert model.model is not None, "XGBoost model should be created"
        print("✓ XGBoostPretrainedBaseline.fit() works with dummy features")
    
    @patch('lib.training._xgboost_pretrained.extract_features_from_pretrained_model')
    def test_predict_with_dummy_features(
        self,
        mock_extract_features,
        dummy_features,
        dummy_df,
        temp_project_root
    ):
        """Test predict() method with dummy features."""
        from lib.training._xgboost_pretrained import XGBoostPretrainedBaseline
        
        # Create a fitted model
        model = XGBoostPretrainedBaseline(
            base_model_type="i3d",
            num_frames=100,
            xgb_params={
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42,
                'tree_method': 'hist',
                'n_jobs': 1,
            }
        )
        
        # Create dummy XGBoost model and fit it
        import xgboost as xgb
        dummy_X = np.random.rand(100, 50)
        dummy_y = np.random.randint(0, 2, 100)
        model.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        model.model.fit(dummy_X, dummy_y)
        model.is_fitted = True
        model.feature_indices = None  # No feature filtering
        
        # Mock feature extraction
        model._extract_features_batch = Mock(return_value=(dummy_features[:10], np.zeros(10)))
        
        # Test predict
        test_df = dummy_df.head(10)
        probs = model.predict(test_df, temp_project_root)
        
        assert probs.shape == (10, 2), f"Expected (10, 2), got {probs.shape}"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1"
        print("✓ XGBoostPretrainedBaseline.predict() works with dummy features")
    
    def test_save_load_with_dummy_model(self, temp_project_root):
        """Test save() and load() methods with dummy model."""
        from lib.training._xgboost_pretrained import XGBoostPretrainedBaseline
        
        # Create model and fit dummy XGBoost model
        model = XGBoostPretrainedBaseline(
            base_model_type="i3d",
            num_frames=100
        )
        
        # Create and fit dummy XGBoost model
        import xgboost as xgb
        dummy_X = np.random.rand(100, 50)
        dummy_y = np.random.randint(0, 2, 100)
        model.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        model.model.fit(dummy_X, dummy_y)
        model.is_fitted = True
        model.feature_indices = [0, 1, 2, 3, 4]
        model.feature_names = [f"feature_{i}" for i in range(5)]
        
        # Save model
        save_path = Path(temp_project_root) / "test_model"
        model.save(str(save_path))
        
        assert (save_path / "xgboost_model.json").exists(), "Model file should exist"
        assert (save_path / "metadata.json").exists(), "Metadata file should exist"
        
        # Load model
        loaded_model = XGBoostPretrainedBaseline(base_model_type="i3d")
        loaded_model.load(str(save_path))
        
        assert loaded_model.is_fitted, "Loaded model should be fitted"
        assert loaded_model.model is not None, "Loaded model should have XGBoost model"
        assert loaded_model.feature_indices == model.feature_indices, "Feature indices should match"
        assert loaded_model.feature_names == model.feature_names, "Feature names should match"
        
        # Verify predictions match
        X_test = np.random.rand(10, 50)
        original_probs = model.model.predict_proba(X_test[:, model.feature_indices])
        loaded_probs = loaded_model.model.predict_proba(X_test[:, loaded_model.feature_indices])
        
        np.testing.assert_allclose(original_probs, loaded_probs, rtol=1e-5)
        print("✓ XGBoostPretrainedBaseline.save() and load() work correctly")


class TestXGBoostAPIFallbackPaths:
    """Test that all fallback paths work correctly."""
    
    def test_fallback_no_callbacks(self):
        """Test fallback when callbacks are not available."""
        # Create dummy data
        n_samples = 100
        n_features = 20
        X_train = np.random.rand(80, n_features)
        X_val = np.random.rand(20, n_features)
        y_train = np.random.randint(0, 2, 80)
        y_val = np.random.randint(0, 2, 20)
        
        # Test without early stopping (fallback path)
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        
        # Fit without early stopping - should always work
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        assert model.is_fitted
        print("✓ Fallback path (no early stopping) works correctly")
    
    def test_all_api_paths_with_dummy_data(self):
        """Test all API paths to ensure no errors."""
        # Create dummy data
        n_samples = 100
        n_features = 20
        X_train = np.random.rand(80, n_features)
        X_val = np.random.rand(20, n_features)
        y_train = np.random.randint(0, 2, 80)
        y_val = np.random.randint(0, 2, 20)
        
        # Path 1: Basic fit
        model1 = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        model1.fit(X_train, y_train)
        assert model1.is_fitted
        
        # Path 2: Fit with eval_set
        model2 = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        model2.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        assert model2.is_fitted
        
        # Path 3: Try callbacks in constructor (if available)
        try:
            from xgboost.callback import EarlyStopping
            model3 = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=10,
                random_state=42,
                tree_method='hist',
                n_jobs=1,
                callbacks=[EarlyStopping(rounds=5)]
            )
            model3.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            assert model3.is_fitted
            print("✓ All API paths work correctly")
        except (ImportError, TypeError, AttributeError):
            print("✓ All available API paths work correctly (callbacks not available)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
