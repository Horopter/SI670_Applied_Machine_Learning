"""Critical tests for data loading."""
import pytest
import polars as pl
from pathlib import Path
from unittest.mock import Mock, patch

from lib.utils.paths import load_metadata_flexible
from lib.data import stratified_kfold, load_metadata


class TestDataLoading:
    """Critical tests for data loading."""

    def test_load_metadata_flexible_csv(self, temp_dir):
        """Test loading CSV metadata."""
        csv_path = Path(temp_dir) / "test.csv"
        data = {
            "video_path": ["v1.mp4", "v2.mp4"],
            "label": [0, 1],
            "video_id": ["id1", "id2"]
        }
        df = pl.DataFrame(data)
        df.write_csv(csv_path)
        
        result = load_metadata_flexible(str(csv_path))
        assert result is not None
        assert result.height == 2

    def test_load_metadata_flexible_json(self, temp_dir):
        """Test loading JSON metadata."""
        json_path = Path(temp_dir) / "test.json"
        data = {
            "video_path": ["v1.mp4", "v2.mp4"],
            "label": [0, 1],
            "video_id": ["id1", "id2"]
        }
        import json
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        result = load_metadata_flexible(str(json_path))
        assert result is not None

    def test_stratified_kfold_basic(self, sample_video_df):
        """Test basic stratified k-fold."""
        splits = stratified_kfold(
            sample_video_df,
            n_splits=3
        )
        assert len(splits) == 3
        for train_df, val_df in splits:
            assert train_df.height > 0
            assert val_df.height > 0

    def test_stratified_kfold_preserves_labels(self, sample_video_df):
        """Test k-fold preserves label distribution."""
        splits = stratified_kfold(
            sample_video_df,
            n_splits=3
        )
        for train_df, val_df in splits:
            train_labels = train_df["label"].unique().to_list()
            val_labels = val_df["label"].unique().to_list()
            # Both should have both labels
            assert 0 in train_labels or 1 in train_labels
            assert 0 in val_labels or 1 in val_labels

    def test_stratified_kfold_no_overlap(self, sample_video_df):
        """Test k-fold has no overlap between train/val."""
        splits = stratified_kfold(
            sample_video_df,
            n_splits=3
        )
        for train_df, val_df in splits:
            train_ids = set(train_df["video_id"].to_list())
            val_ids = set(val_df["video_id"].to_list())
            assert len(train_ids & val_ids) == 0

    def test_stratified_kfold_duplicate_groups(self, sample_video_df_with_dups):
        """Test k-fold with duplicate groups."""
        splits = stratified_kfold(
            sample_video_df_with_dups,
            n_splits=2
        )
        assert len(splits) == 2
        for train_df, val_df in splits:
            train_groups = set(train_df["dup_group"].to_list())
            val_groups = set(val_df["dup_group"].to_list())
            # No overlap in duplicate groups
            assert len(train_groups & val_groups) == 0

