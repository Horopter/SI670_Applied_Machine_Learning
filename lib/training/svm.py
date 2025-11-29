"""
Linear SVM baseline model using handcrafted features.
"""

from __future__ import annotations

import os
import logging
from typing import Optional
import numpy as np
import polars as pl
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib

from lib.features.handcrafted import HandcraftedFeatureExtractor

logger = logging.getLogger(__name__)


class SVMBaseline:
    """
    Linear SVM baseline using handcrafted features.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, num_frames: int = 8):
        """
        Initialize baseline model.
        
        Args:
            cache_dir: Directory to cache extracted features
            num_frames: Number of frames to sample per video
        """
        self.feature_extractor = HandcraftedFeatureExtractor(cache_dir, num_frames)
        self.scaler = StandardScaler()
        self.model = LinearSVC(max_iter=1000, random_state=42)
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, project_root: str) -> None:
        """
        Train the model.
        
        Args:
            df: DataFrame with video_path and label columns
            project_root: Project root directory
        """
        logger.info("Extracting handcrafted features for %d videos...", df.height)
        
        video_paths = df["video_path"].to_list()
        labels = df["label"].to_list()
        
        # Extract features (extreme conservative batch size for OOM safety)
        from lib.utils.memory import aggressive_gc
        features = self.feature_extractor.extract_batch(
            video_paths,
            project_root,
            batch_size=1,  # Extreme conservative: reduced from 3 to minimize memory
        )
        
        # Aggressive GC after feature extraction
        aggressive_gc(clear_cuda=False)
        
        # Convert labels to binary (0/1)
        label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        y = np.array([label_map[label] for label in labels])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        logger.info("Training Linear SVM...")
        self.model.fit(features_scaled, y)
        self.is_fitted = True
        
        logger.info("✓ Linear SVM trained")
    
    def predict(self, df: pl.DataFrame, project_root: str) -> np.ndarray:
        """
        Predict labels for videos.
        
        Args:
            df: DataFrame with video_path column
            project_root: Project root directory
        
        Returns:
            Predicted probabilities (n_samples, 2)
            Note: SVM doesn't provide probabilities by default, so we use decision function
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        video_paths = df["video_path"].to_list()
        
        # Extract features
        features = self.feature_extractor.extract_batch(
            video_paths,
            project_root,
            batch_size=4  # Reduced from 10 for 80GB RAM constraint
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get decision function (distance from hyperplane)
        decision = self.model.decision_function(features_scaled)
        
        # Convert to probabilities using sigmoid
        # This is an approximation since LinearSVC doesn't provide probabilities
        probs_positive = 1 / (1 + np.exp(-decision))
        probs = np.column_stack([1 - probs_positive, probs_positive])
        
        return probs
    
    def save(self, save_dir: str) -> None:
        """Save model and scaler."""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_dir, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.joblib"))
        logger.info("Saved SVM model to %s", save_dir)
    
    def load(self, load_dir: str) -> None:
        """Load model and scaler."""
        self.model = joblib.load(os.path.join(load_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(load_dir, "scaler.joblib"))
        self.is_fitted = True
        logger.info("Loaded SVM model from %s", load_dir)


__all__ = ["SVMBaseline"]

