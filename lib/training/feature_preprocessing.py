"""
Feature preprocessing utilities for training.

Provides functions to:
- Remove collinear features
- Combine features from multiple stages
- Normalize and scale features
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import sklearn for VIF calculation
try:
    from sklearn.feature_selection import VarianceThreshold
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    SKLEARN_AVAILABLE = True
    STATSMODELS_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    STATSMODELS_AVAILABLE = False
    VarianceThreshold = None
    variance_inflation_factor = None
    add_constant = None


def remove_collinear_features(
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    correlation_threshold: float = 0.95,
    vif_threshold: float = 10.0,
    method: str = "correlation"
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Remove collinear features using correlation analysis or VIF.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        feature_names: Optional list of feature names
        correlation_threshold: Maximum correlation allowed between features (default: 0.95)
        vif_threshold: Maximum VIF allowed (default: 10.0, only used if method="vif")
        method: Method to use ("correlation" or "vif" or "both")
    
    Returns:
        Tuple of:
        - Filtered feature matrix (n_samples, n_features_filtered)
        - Indices of kept features
        - Names of kept features (or empty list if feature_names not provided)
    """
    if features.shape[1] == 0:
        return features, [], []
    
    n_samples, n_features = features.shape
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    if len(feature_names) != n_features:
        logger.warning(f"Feature names length ({len(feature_names)}) doesn't match features ({n_features})")
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Remove features with zero variance
    if SKLEARN_AVAILABLE:
        try:
            variance_selector = VarianceThreshold(threshold=0.0)
            features_var_filtered = variance_selector.fit_transform(features)
            kept_indices = variance_selector.get_support(indices=True).tolist()
            logger.info(f"Removed {n_features - len(kept_indices)} features with zero variance")
        except ValueError as e:
            # All features have zero variance - use manual filtering instead
            logger.warning(f"VarianceThreshold failed (all features may have zero variance): {e}")
            logger.warning("Falling back to manual variance filtering")
            variances = np.var(features, axis=0)
            kept_indices = np.where(variances > 1e-8)[0].tolist()
            if len(kept_indices) == 0:
                # All features have zero variance - keep all features as fallback
                logger.warning("All features have zero variance! Keeping all features as fallback.")
                kept_indices = list(range(n_features))
            features_var_filtered = features[:, kept_indices]
            logger.info(f"Removed {n_features - len(kept_indices)} features with zero variance (manual)")
    else:
        # Manual variance filtering
        variances = np.var(features, axis=0)
        kept_indices = np.where(variances > 1e-8)[0].tolist()
        if len(kept_indices) == 0:
            # All features have zero variance - keep all features as fallback
            logger.warning("All features have zero variance! Keeping all features as fallback.")
            kept_indices = list(range(n_features))
        features_var_filtered = features[:, kept_indices]
        logger.info(f"Removed {n_features - len(kept_indices)} features with zero variance")
    
    if len(kept_indices) == 0:
        logger.warning("All features removed due to zero variance! Using original features as fallback.")
        return features, list(range(n_features)), feature_names
    
    # Update feature names
    kept_feature_names = [feature_names[i] for i in kept_indices]
    
    # Remove collinear features based on correlation
    if method in ["correlation", "both"]:
        # Compute correlation matrix
        corr_matrix = np.corrcoef(features_var_filtered.T)
        
        # Find highly correlated feature pairs
        to_remove = set()
        for i in range(len(kept_indices)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(kept_indices)):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) >= correlation_threshold:
                    # Remove the feature with lower variance (less informative)
                    var_i = np.var(features_var_filtered[:, i])
                    var_j = np.var(features_var_filtered[:, j])
                    if var_i < var_j:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
                    logger.debug(
                        f"Removing collinear feature: {kept_feature_names[j if var_i < var_j else i]} "
                        f"(correlation={corr_matrix[i, j]:.3f})"
                    )
        
        # Filter features
        final_kept_indices = [kept_indices[i] for i in range(len(kept_indices)) if i not in to_remove]
        features_filtered = features_var_filtered[:, [i for i in range(len(kept_indices)) if i not in to_remove]]
        final_feature_names = [kept_feature_names[i] for i in range(len(kept_indices)) if i not in to_remove]
        
        logger.info(
            f"Removed {len(kept_indices) - len(final_kept_indices)} collinear features "
            f"(correlation >= {correlation_threshold})"
        )
        
        if method == "both":
            # Also apply VIF filtering
            features_filtered, final_kept_indices, final_feature_names = _remove_vif_collinear(
                features_filtered, final_kept_indices, final_feature_names, vif_threshold
            )
    elif method == "vif":
        # Use VIF only
        features_filtered, final_kept_indices, final_feature_names = _remove_vif_collinear(
            features_var_filtered, kept_indices, kept_feature_names, vif_threshold
        )
    else:
        logger.warning(f"Unknown method '{method}', using correlation method")
        return remove_collinear_features(
            features, feature_names, correlation_threshold, vif_threshold, method="correlation"
        )
    
    logger.info(
        f"Final feature count: {len(final_kept_indices)}/{n_features} "
        f"({100 * len(final_kept_indices) / n_features:.1f}% retained)"
    )
    
    return features_filtered, final_kept_indices, final_feature_names


def _remove_vif_collinear(
    features: np.ndarray,
    kept_indices: List[int],
    feature_names: List[str],
    vif_threshold: float
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Remove features with high Variance Inflation Factor (VIF).
    
    Args:
        features: Feature matrix (n_samples, n_features)
        kept_indices: Current list of kept feature indices
        feature_names: Current list of kept feature names
        vif_threshold: Maximum VIF allowed
    
    Returns:
        Tuple of (filtered_features, final_kept_indices, final_feature_names)
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available, skipping VIF filtering")
        return features, kept_indices, feature_names
    
    if features.shape[1] == 0:
        return features, kept_indices, feature_names
    
    # Calculate VIF for each feature
    try:
        # Add constant for VIF calculation
        features_with_const = add_constant(features, has_constant='skip')
        
        vif_scores = []
        for i in range(1, features_with_const.shape[1]):  # Skip constant column
            try:
                vif = variance_inflation_factor(features_with_const.values, i)
                vif_scores.append(vif if not np.isnan(vif) and np.isfinite(vif) else np.inf)
            except Exception as e:
                logger.debug(f"Error calculating VIF for feature {i}: {e}")
                vif_scores.append(np.inf)
        
        # Remove features with VIF > threshold
        to_keep = [i for i, vif in enumerate(vif_scores) if vif <= vif_threshold]
        
        if len(to_keep) < len(vif_scores):
            removed_count = len(vif_scores) - len(to_keep)
            logger.info(
                f"Removed {removed_count} features with VIF > {vif_threshold} "
                f"(max VIF: {max(vif_scores):.2f})"
            )
            
            final_features = features[:, to_keep]
            final_indices = [kept_indices[i] for i in to_keep]
            final_names = [feature_names[i] for i in to_keep]
            
            return final_features, final_indices, final_names
        else:
            logger.debug(f"All features have VIF <= {vif_threshold}")
            return features, kept_indices, feature_names
            
    except Exception as e:
        logger.warning(f"Error in VIF calculation: {e}, skipping VIF filtering")
        return features, kept_indices, feature_names


def load_and_combine_features(
    features_stage2_path: Optional[str],
    features_stage4_path: Optional[str],
    video_paths: List[str],
    project_root: str,
    remove_collinearity: bool = True,
    correlation_threshold: float = 0.95,
    vif_threshold: float = 10.0,
    collinearity_method: str = "correlation"
) -> Tuple[np.ndarray, List[str], Optional[List[int]]]:
    """
    Load and combine features from Stage 2 and Stage 4 metadata files.
    Optionally removes collinear features after combining.
    
    Args:
        features_stage2_path: Path to Stage 2 features metadata
        features_stage4_path: Path to Stage 4 features metadata
        video_paths: List of video paths to match features
        project_root: Project root directory
        remove_collinearity: Whether to remove collinear features (default: True)
        correlation_threshold: Maximum correlation allowed (default: 0.95)
        vif_threshold: Maximum VIF allowed (default: 10.0)
        collinearity_method: Method for collinearity removal ("correlation", "vif", or "both")
    
    Returns:
        Tuple of (combined_features, feature_names, kept_indices)
        - combined_features: Feature matrix (n_samples, n_features)
        - feature_names: List of feature names
        - kept_indices: Indices of kept features (None if remove_collinearity=False)
    """
    all_features = []
    all_feature_names = []
    
    # Load Stage 2 features
    if features_stage2_path and Path(features_stage2_path).exists():
        logger.info("Loading Stage 2 features...")
        try:
            path_obj = Path(features_stage2_path)
            if path_obj.suffix == '.arrow':
                df2 = pl.read_ipc(path_obj)
            elif path_obj.suffix == '.parquet':
                df2 = pl.read_parquet(path_obj)
            else:
                df2 = pl.read_csv(features_stage2_path)
            
            # Get feature columns (exclude metadata columns)
            metadata_cols = {'video_path', 'label', 'feature_path'}
            feature_cols = [col for col in df2.columns if col not in metadata_cols]
            
            if feature_cols:
                # Match features to video paths
                features_dict = {row['video_path']: [row[col] for col in feature_cols] 
                                for row in df2.iter_rows(named=True)}
                
                stage2_features = []
                for vpath in video_paths:
                    # Try to match video path
                    matched = None
                    for key, vals in features_dict.items():
                        if key in vpath or vpath in key:
                            matched = vals
                            break
                    
                    if matched is None:
                        logger.warning(f"No Stage 2 features found for {vpath}")
                        matched = [0.0] * len(feature_cols)
                    
                    stage2_features.append(matched)
                
                all_features.append(np.array(stage2_features))
                all_feature_names.extend([f"stage2_{col}" for col in feature_cols])
                logger.info(f"Loaded {len(feature_cols)} Stage 2 features")
        except Exception as e:
            logger.error(f"Error loading Stage 2 features: {e}")
    
    # Load Stage 4 features
    if features_stage4_path and Path(features_stage4_path).exists():
        logger.info("Loading Stage 4 features...")
        try:
            path_obj = Path(features_stage4_path)
            if path_obj.suffix == '.arrow':
                df4 = pl.read_ipc(path_obj)
            elif path_obj.suffix == '.parquet':
                df4 = pl.read_parquet(path_obj)
            else:
                df4 = pl.read_csv(features_stage4_path)
            
            # Get feature columns
            metadata_cols = {'video_path', 'label', 'feature_path'}
            feature_cols = [col for col in df4.columns if col not in metadata_cols]
            
            if feature_cols:
                features_dict = {row['video_path']: [row[col] for col in feature_cols] 
                                for row in df4.iter_rows(named=True)}
                
                stage4_features = []
                for vpath in video_paths:
                    matched = None
                    for key, vals in features_dict.items():
                        if key in vpath or vpath in key:
                            matched = vals
                            break
                    
                    if matched is None:
                        logger.warning(f"No Stage 4 features found for {vpath}")
                        matched = [0.0] * len(feature_cols)
                    
                    stage4_features.append(matched)
                
                all_features.append(np.array(stage4_features))
                all_feature_names.extend([f"stage4_{col}" for col in feature_cols])
                logger.info(f"Loaded {len(feature_cols)} Stage 4 features")
        except Exception as e:
            logger.error(f"Error loading Stage 4 features: {e}")
    
    if not all_features:
        logger.warning("No features loaded!")
        return np.array([]).reshape(len(video_paths), 0), [], None
    
    # Combine features
    combined_features = np.hstack(all_features)
    logger.info(f"Combined {len(all_feature_names)} features from {len(all_features)} stages")
    
    # Remove collinear features if requested
    kept_indices = None
    if remove_collinearity and combined_features.shape[1] > 0:
        logger.info("Removing collinear features from combined features...")
        combined_features, kept_indices, all_feature_names = remove_collinear_features(
            combined_features,
            feature_names=all_feature_names,
            correlation_threshold=correlation_threshold,
            vif_threshold=vif_threshold,
            method=collinearity_method
        )
        logger.info(f"Final feature count after collinearity removal: {len(all_feature_names)}")
    
    return combined_features, all_feature_names, kept_indices


__all__ = [
    "remove_collinear_features",
    "load_and_combine_features",
]

