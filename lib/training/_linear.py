"""
Logistic Regression baseline models using Stage 2 and Stage 4 features.

Two versions:
- logistic_regression_stage2: Uses only Stage 2 features
- logistic_regression_stage2_stage4: Uses Stage 2 + Stage 4 features combined
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from lib.training.feature_preprocessing import load_and_combine_features, remove_collinear_features

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline using Stage 2 and/or Stage 4 features.
    
    Supports two modes:
    - stage2_only: Use only Stage 2 features
    - stage2_stage4: Use Stage 2 + Stage 4 features combined
    """
    
    def __init__(
        self,
        features_stage2_path: Optional[str] = None,
        features_stage4_path: Optional[str] = None,
        use_stage2_only: bool = False,
        cache_dir: Optional[str] = None,
        num_frames: int = 1000
    ):
        """
        Initialize baseline model.
        
        Args:
            features_stage2_path: Path to Stage 2 features metadata (REQUIRED for training in Stage 5)
            features_stage4_path: Path to Stage 4 features metadata (optional, for stage2_stage4 models)
            use_stage2_only: If True, use only Stage 2 features; if False, combine Stage 2 + Stage 4
            cache_dir: Directory to cache extracted features (unused, kept for compatibility)
            num_frames: Number of frames to sample (used only if extracting features during prediction)
        """
        self.num_frames = num_frames
        self.features_stage2_path = features_stage2_path
        self.features_stage4_path = features_stage4_path
        self.use_stage2_only = use_stage2_only
        self.scaler = StandardScaler()
        # Use solver that supports warm_start for epoch-wise training
        self.model = LogisticRegression(
            max_iter=1000, 
            random_state=42,
            solver='saga',  # Supports warm_start for iterative training
            warm_start=True  # Enable iterative training for epoch-wise curves
        )
        self.is_fitted = False
        self.tracker = None  # Will be set if epoch-wise training is requested
        self.feature_indices = None  # Indices of kept features after collinearity removal
        self.feature_names = None  # Names of kept features
        self.project_root = None  # Store project root for prediction
    
    def fit(self, df: pl.DataFrame, project_root: str, output_dir: Optional[str] = None) -> None:
        """
        Train the model.
        
        Args:
            df: DataFrame with video_path and label columns
            project_root: Project root directory
            output_dir: Optional output directory for metrics logging (defaults to project_root/data/stage5/{model_type}/fold_1)
        """
        self.project_root = project_root
        
        video_paths = df["video_path"].to_list()
        labels = df["label"].to_list()
        
        # Determine which features to load
        stage2_path = self.features_stage2_path
        stage4_path = None if self.use_stage2_only else self.features_stage4_path
        
        # CRITICAL: Stage 2 features are REQUIRED for all Logistic Regression models
        # Stage 5 only trains - features must be pre-extracted in Stage 2/4
        if not stage2_path:
            raise ValueError(
                f"Stage 2 features path is REQUIRED for {self.__class__.__name__}. "
                f"Features must be pre-extracted in Stage 2. "
                f"Do NOT re-extract features during training. "
                f"Please provide features_stage2_path in model configuration. "
                f"Expected Stage 2 features metadata file."
            )
        
        # Validate Stage 2 path exists and is not empty
        from pathlib import Path
        from lib.utils.paths import load_metadata_flexible
        stage2_path_obj = Path(stage2_path)
        if not stage2_path_obj.exists():
            raise FileNotFoundError(
                f"Stage 2 features metadata file does not exist: {stage2_path}. "
                f"Features must be pre-extracted in Stage 2. "
                f"Please run Stage 2 feature extraction first."
            )
        
        # Check if file is not empty
        test_df = load_metadata_flexible(stage2_path)
        if test_df is None or test_df.height == 0:
            raise ValueError(
                f"Stage 2 features metadata file is empty: {stage2_path}. "
                f"Features must be pre-extracted in Stage 2. "
                f"Please run Stage 2 feature extraction first."
            )
        
        logger.info(f"Using Stage 2 features from: {stage2_path} ({test_df.height} rows)")
        
        # Validate Stage 4 path if provided (for stage2_stage4 models)
        if stage4_path:
            stage4_path_obj = Path(stage4_path)
            if not stage4_path_obj.exists():
                raise FileNotFoundError(
                    f"Stage 4 features metadata file does not exist: {stage4_path}. "
                    f"Stage 4 is required for {self.__class__.__name__} when use_stage2_only=False. "
                    f"Please run Stage 4 scaled feature extraction first."
                )
            
            # Check if file is not empty
            test_df = load_metadata_flexible(stage4_path)
            if test_df is None or test_df.height == 0:
                raise ValueError(
                    f"Stage 4 features metadata file is empty: {stage4_path}. "
                    f"Stage 4 is required for {self.__class__.__name__} when use_stage2_only=False. "
                    f"Please run Stage 4 scaled feature extraction first."
                )
            logger.info(f"Using Stage 4 features from: {stage4_path} ({test_df.height} rows)")
        
        # Load and combine features (for both stage2_only and stage2_stage4 modes)
        logger.info(
            f"Loading features for {len(video_paths)} videos "
            f"(Stage 2: {stage2_path}, Stage 4: {stage4_path if stage4_path else 'not used'})..."
        )
        logger.info(f"Stage 2 path: {stage2_path}")
        logger.info(f"Stage 4 path: {stage4_path if stage4_path else 'not used'}")
        sys.stdout.flush()
        
        # NOTE: Collinearity removal should already be done before splits in the main pipeline
        # We load without removing collinearity here to avoid doing it multiple times
        try:
            features, feature_names, kept_indices, valid_video_indices = load_and_combine_features(
                features_stage2_path=stage2_path,
                features_stage4_path=stage4_path,
                video_paths=video_paths,
                project_root=project_root,
                remove_collinearity=False,  # Already done before splits in main pipeline
                correlation_threshold=0.95,
                collinearity_method="correlation"
            )
            logger.info(f"✓ Loaded {len(feature_names)} features (collinearity already removed)")
            logger.info(f"Feature matrix shape: {features.shape if features is not None else 'None'}")
            sys.stdout.flush()
        except MemoryError as e:
            logger.critical(f"Memory error during feature loading: {e}")
            logger.critical("This may indicate insufficient memory or corrupted feature files")
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load features: {e}", exc_info=True)
            if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                logger.critical("CRITICAL: Possible crash during feature loading")
                logger.critical("This may indicate corrupted feature files or library incompatibility")
                logger.critical(f"Check feature files: Stage 2: {stage2_path}, Stage 4: {stage4_path}")
            logger.error(
                "Make sure Stage 2/4 features are already extracted. "
                "Do NOT re-extract features during training."
            )
            raise
        
        # Filter to valid videos if needed
        if valid_video_indices is not None and len(valid_video_indices) > 0:
            original_count = len(video_paths)
            features = features[valid_video_indices]
            video_paths = [video_paths[i] for i in valid_video_indices]
            labels = [labels[i] for i in valid_video_indices]
            logger.info(f"Filtered to {len(video_paths)}/{original_count} videos with valid features")
        
        # Validate features
        if features is None or features.size == 0:
            raise ValueError("No features loaded from Stage 2/4 metadata")
        if len(features.shape) != 2:
            raise ValueError(f"Invalid feature shape: {features.shape}, expected 2D array")
        if features.shape[0] != len(video_paths):
            raise ValueError(
                f"Feature count mismatch: {features.shape[0]} features for {len(video_paths)} videos"
            )
        
        # Store feature information
        self.feature_indices = kept_indices
        self.feature_names = feature_names
        
        logger.info(f"Using {len(feature_names)} features after collinearity removal")
        
        # Convert labels to binary (0/1)
        label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        if len(label_map) < 2:
            raise ValueError(f"Need at least 2 classes, found {len(label_map)}: {label_map}")
        y = np.array([label_map[label] for label in labels])
        
        # Validate labels
        if len(y) != features.shape[0]:
            raise ValueError(f"Label count mismatch: {len(y)} labels for {features.shape[0]} samples")
        
        # Scale features
        logger.info("Scaling features...")
        logger.info(f"Feature matrix shape: {features.shape}, dtype: {features.dtype}")
        logger.info(f"Feature stats: min={np.nanmin(features):.4f}, max={np.nanmax(features):.4f}, mean={np.nanmean(features):.4f}")
        sys.stdout.flush()
        
        # Check for corrupted features before scaling
        if np.any(np.isnan(features)):
            nan_count = np.isnan(features).sum()
            logger.warning(f"Found {nan_count} NaN values in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0)
        if np.any(np.isinf(features)):
            inf_count = np.isinf(features).sum()
            logger.warning(f"Found {inf_count} Inf values in features, replacing with 0")
            features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
        
        try:
            features_scaled = self.scaler.fit_transform(features)
            logger.info(f"Feature scaling completed. Scaled shape: {features_scaled.shape}")
        except MemoryError as e:
            logger.critical(f"Memory error during feature scaling: {e}")
            logger.critical(f"Feature matrix size: {features.nbytes / 1024**2:.2f} MB")
            raise
        except Exception as e:
            logger.error(f"Failed to scale features: {e}", exc_info=True)
            logger.error(f"Feature matrix shape: {features.shape}, dtype: {features.dtype}")
            raise
        
        # Validate scaled features
        if np.any(np.isnan(features_scaled)) or np.any(np.isinf(features_scaled)):
            logger.warning("NaN or Inf values in scaled features, replacing with 0")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train model
        logger.info("Training Logistic Regression...")
        logger.info(f"Training samples: {features_scaled.shape[0]}, Features: {features_scaled.shape[1]}")
        logger.info(f"Label distribution: {np.bincount(y)}")
        sys.stdout.flush()
        
        # Check if we should do epoch-wise training (if tracker is available or output_dir is provided)
        # This allows capturing training/validation metrics per iteration
        num_epochs = 100  # Default number of iterations for epoch-wise training
        
        # Create tracker if output_dir is provided but tracker is not set
        if self.tracker is None and output_dir is not None:
            try:
                from lib.mlops.config import ExperimentTracker
                from pathlib import Path
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                self.tracker = ExperimentTracker(output_path)
                logger.info(f"Created tracker for epoch-wise training at {output_dir}")
            except Exception as e:
                logger.debug(f"Could not create tracker from output_dir {output_dir}: {e}")
        
        # Default output_dir if not provided and tracker is not available
        if self.tracker is None and output_dir is None:
            try:
                from lib.mlops.config import ExperimentTracker
                from pathlib import Path
                # Default to project_root/data/stage5/logistic_regression/fold_1
                default_output_dir = Path(project_root) / "data" / "stage5" / "logistic_regression" / "fold_1"
                default_output_dir.mkdir(parents=True, exist_ok=True)
                self.tracker = ExperimentTracker(default_output_dir)
                logger.info(f"Created tracker with default output_dir: {default_output_dir}")
            except Exception as e:
                logger.debug(f"Could not create tracker with default output_dir: {e}")
        
        do_epoch_wise = self.tracker is not None
        
        try:
            if do_epoch_wise:
                # Train iteratively to capture epoch-wise metrics
                logger.info(f"Training Logistic Regression with {num_epochs} iterations (epoch-wise)...")
                
                # Split data for validation during training
                from sklearn.model_selection import train_test_split
                X_train_iter, X_val_iter, y_train_iter, y_val_iter = train_test_split(
                    features_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train iteratively
                for epoch in range(num_epochs):
                    # Fit with max_iter=1 and warm_start=True for iterative training
                    self.model.max_iter = epoch + 1
                    self.model.fit(X_train_iter, y_train_iter)
                    
                    # Evaluate on validation set
                    from sklearn.metrics import log_loss, accuracy_score, f1_score
                    val_probs = self.model.predict_proba(X_val_iter)
                    val_preds = self.model.predict(X_val_iter)
                    
                    val_loss = log_loss(y_val_iter, val_probs)
                    val_acc = accuracy_score(y_val_iter, val_preds)
                    val_f1 = f1_score(y_val_iter, val_preds, average='binary', zero_division=0)
                    
                    # Log validation metrics
                    self.tracker.log_epoch_metrics(
                        epoch + 1,
                        {
                            "loss": float(val_loss),
                            "accuracy": float(val_acc),
                            "f1": float(val_f1)
                        },
                        phase="val"
                    )
                    
                    # Also evaluate on training set periodically
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        train_probs = self.model.predict_proba(X_train_iter)
                        train_preds = self.model.predict(X_train_iter)
                        train_loss = log_loss(y_train_iter, train_probs)
                        train_acc = accuracy_score(y_train_iter, train_preds)
                        train_f1 = f1_score(y_train_iter, train_preds, average='binary', zero_division=0)
                        
                        self.tracker.log_epoch_metrics(
                            epoch + 1,
                            {
                                "loss": float(train_loss),
                                "accuracy": float(train_acc),
                                "f1": float(train_f1)
                            },
                            phase="train"
                        )
                
                # Final fit on full dataset
                self.model.max_iter = num_epochs
                self.model.fit(features_scaled, y)
                logger.info(f"✓ Logistic Regression training completed ({num_epochs} iterations)")
            else:
                # Standard training (no epoch-wise tracking)
                self.model.fit(features_scaled, y)
                logger.info("✓ Logistic Regression training completed")
        except MemoryError as e:
            logger.critical(f"Memory error during Logistic Regression training: {e}")
            logger.critical(f"Feature matrix size: {features_scaled.nbytes / 1024**2:.2f} MB")
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to train Logistic Regression: {e}", exc_info=True)
            if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                logger.critical("CRITICAL: Possible crash during sklearn LogisticRegression.fit()")
                logger.critical("This may indicate corrupted features, memory issue, or sklearn library incompatibility")
            raise
        
        self.is_fitted = True
        logger.info("✓ Logistic Regression trained")
    
    def predict(self, df: pl.DataFrame, project_root: Optional[str] = None) -> np.ndarray:
        """
        Predict labels for videos.
        
        Args:
            df: DataFrame with video_path column
            project_root: Project root directory (uses stored value if None)
        
        Returns:
            Predicted probabilities (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if project_root is None:
            project_root = self.project_root
        if project_root is None:
            raise ValueError("project_root must be provided either in fit() or predict()")
        
        video_paths = df["video_path"].to_list()
        
        # Determine which features to load
        stage2_path = self.features_stage2_path
        stage4_path = None if self.use_stage2_only else self.features_stage4_path
        
        # CRITICAL: Stage 2 features are REQUIRED for prediction
        # Features must be pre-extracted in Stage 2/4 - no in-prediction extraction
        if not stage2_path:
            raise ValueError(
                f"Stage 2 features path is REQUIRED for {self.__class__.__name__}.predict(). "
                f"Features must be pre-extracted in Stage 2. "
                f"Do NOT re-extract features during prediction. "
                f"Please provide features_stage2_path in model configuration."
            )
        
        # Validate Stage 2 path exists
        from pathlib import Path
        stage2_path_obj = Path(stage2_path)
        if not stage2_path_obj.exists():
            raise FileNotFoundError(
                f"Stage 2 features metadata file does not exist: {stage2_path}. "
                f"Features must be pre-extracted in Stage 2. "
                f"Please run Stage 2 feature extraction first."
            )
        
        # Load and combine features
        logger.info(f"Loading features for prediction ({len(video_paths)} videos)...")
        sys.stdout.flush()
        
        try:
            features, _, _, _ = load_and_combine_features(
                features_stage2_path=stage2_path,
                features_stage4_path=stage4_path,
                video_paths=video_paths,
                project_root=project_root,
                remove_collinearity=False  # Don't remove collinearity again, use same features as training
            )
            logger.info(f"✓ Loaded features for prediction (shape: {features.shape if features is not None else 'None'})")
        except MemoryError as e:
            logger.critical(f"Memory error during feature loading in predict(): {e}")
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load features in predict(): {e}", exc_info=True)
            if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                logger.critical("CRITICAL: Possible crash during feature loading in predict()")
                logger.critical(f"Check feature files: Stage 2: {stage2_path}, Stage 4: {stage4_path}")
            raise
        
        # Apply same feature filtering as during training
        if self.feature_indices is not None:
            features = features[:, self.feature_indices]
            logger.debug(f"Applied feature filtering: {len(self.feature_indices)} features")
        
        # Scale features
        logger.info(f"Scaling features for prediction (shape: {features.shape})...")
        sys.stdout.flush()
        
        # Check for corrupted features before scaling
        if np.any(np.isnan(features)):
            nan_count = np.isnan(features).sum()
            logger.warning(f"Found {nan_count} NaN values in prediction features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0)
        if np.any(np.isinf(features)):
            inf_count = np.isinf(features).sum()
            logger.warning(f"Found {inf_count} Inf values in prediction features, replacing with 0")
            features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
        
        try:
            features_scaled = self.scaler.transform(features)
        except MemoryError as e:
            logger.critical(f"Memory error during feature scaling in predict(): {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to scale features in predict(): {e}", exc_info=True)
            raise
        
        # Predict probabilities
        logger.info("Running LogisticRegression.predict_proba()...")
        sys.stdout.flush()
        
        try:
            probs = self.model.predict_proba(features_scaled)
            logger.info(f"✓ Prediction completed (shape: {probs.shape})")
        except MemoryError as e:
            logger.critical(f"Memory error during LogisticRegression.predict_proba(): {e}")
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to predict probabilities: {e}", exc_info=True)
            if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                logger.critical("CRITICAL: Possible crash during sklearn LogisticRegression.predict_proba()")
                logger.critical("This may indicate corrupted features, memory issue, or sklearn library incompatibility")
            raise
        
        return probs
    
    def save(self, save_dir: str) -> None:
        """Save model and scaler."""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_dir, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.joblib"))
        # Save feature metadata
        import json
        metadata = {
            "feature_indices": self.feature_indices,
            "feature_names": self.feature_names,
            "use_stage2_only": self.use_stage2_only,
            "features_stage2_path": self.features_stage2_path,
            "features_stage4_path": self.features_stage4_path,
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved Logistic Regression model to %s", save_dir)
    
    def load(self, load_dir: str) -> None:
        """Load model and scaler."""
        self.model = joblib.load(os.path.join(load_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(load_dir, "scaler.joblib"))
        # Load feature metadata
        import json
        metadata_path = os.path.join(load_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.feature_indices = metadata.get("feature_indices")
            self.feature_names = metadata.get("feature_names")
            self.use_stage2_only = metadata.get("use_stage2_only", False)
            self.features_stage2_path = metadata.get("features_stage2_path")
            self.features_stage4_path = metadata.get("features_stage4_path")
        self.is_fitted = True
        logger.info("Loaded Logistic Regression model from %s", load_dir)


__all__ = ["LogisticRegressionBaseline"]
