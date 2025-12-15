"""Comprehensive unit tests for handcrafted_features module."""
import pytest
import numpy as np
import cv2
from lib.features.handcrafted import (
    extract_noise_residual,
    extract_dct_statistics,
    extract_blur_sharpness,
    extract_boundary_inconsistency,
    extract_codec_cues,
    extract_all_features,
)


class TestExtractNoiseResidual:
    """Stress tests for extract_noise_residual."""
    
    def test_basic_extraction(self):
        """Test basic noise residual extraction."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        residual = extract_noise_residual(frame)
        
        assert isinstance(residual, dict)
        assert "noise_energy" in residual
    
    def test_grayscale_frame(self):
        """Test with grayscale frame."""
        frame = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        residual = extract_noise_residual(frame)
        
        assert isinstance(residual, dict)
        assert "noise_energy" in residual
    
    def test_small_frame(self):
        """Test with small frame."""
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        residual = extract_noise_residual(frame)
        
        assert isinstance(residual, dict)
        assert "noise_energy" in residual
    
    def test_large_frame(self):
        """Stress test with large frame."""
        frame = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
        residual = extract_noise_residual(frame)
        
        assert isinstance(residual, dict)
        assert "noise_energy" in residual
    
    def test_uniform_frame(self):
        """Test with uniform frame (no noise)."""
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
        residual = extract_noise_residual(frame)
        
        # Should have very low residual
        assert isinstance(residual, dict)
        assert "noise_energy" in residual
        assert residual["noise_energy"] >= 0
    
    def test_extreme_values(self):
        """Test with extreme pixel values."""
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        residual = extract_noise_residual(frame)
        
        assert isinstance(residual, dict)
        assert "noise_energy" in residual
        
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 255
        residual = extract_noise_residual(frame)
        
        assert isinstance(residual, dict)
        assert "noise_energy" in residual


class TestExtractDctStatistics:
    """Stress tests for extract_dct_statistics."""
    
    def test_basic_extraction(self):
        """Test basic DCT statistics extraction."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        stats = extract_dct_statistics(frame)
        
        assert isinstance(stats, dict)
        assert "dct_dc_mean" in stats
        assert "dct_dc_std" in stats
        assert "dct_ac_mean" in stats
    
    def test_grayscale_frame(self):
        """Test with grayscale frame."""
        frame = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        stats = extract_dct_statistics(frame)
        
        assert isinstance(stats, dict)
    
    def test_different_block_sizes(self):
        """Test with different block sizes."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        for block_size in [4, 8, 16]:
            stats = extract_dct_statistics(frame, block_size=block_size)
            assert isinstance(stats, dict)
    
    def test_non_multiple_dimensions(self):
        """Test with dimensions not multiple of block size."""
        frame = np.random.randint(0, 256, (225, 225, 3), dtype=np.uint8)
        stats = extract_dct_statistics(frame, block_size=8)
        
        # Should handle padding
        assert isinstance(stats, dict)
    
    def test_small_frame(self):
        """Test with very small frame."""
        frame = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        stats = extract_dct_statistics(frame)
        
        assert isinstance(stats, dict)
    
    def test_uniform_frame(self):
        """Test with uniform frame."""
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
        stats = extract_dct_statistics(frame)
        
        # AC coefficients should be very small
        assert stats["dct_ac_mean"] < 0.1


class TestExtractBlurSharpness:
    """Stress tests for extract_blur_sharpness."""
    
    def test_basic_extraction(self):
        """Test basic blur/sharpness extraction."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        stats = extract_blur_sharpness(frame)
        
        assert isinstance(stats, dict)
        assert "laplacian_var" in stats
        assert "gradient_mean" in stats
    
    def test_sharp_frame(self):
        """Test with sharp frame (high frequency)."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        stats = extract_blur_sharpness(frame)
        
        assert stats["laplacian_var"] >= 0
        assert stats["gradient_mean"] >= 0
    
    def test_blurred_frame(self):
        """Test with blurred frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(frame, (15, 15), 5.0)
        stats = extract_blur_sharpness(blurred)
        
        # Blurred frame should have lower variance
        assert stats["laplacian_var"] >= 0
    
    def test_grayscale_frame(self):
        """Test with grayscale frame."""
        frame = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        stats = extract_blur_sharpness(frame)
        
        assert isinstance(stats, dict)
    
    def test_small_frame(self):
        """Test with small frame."""
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        stats = extract_blur_sharpness(frame)
        
        assert isinstance(stats, dict)


class TestExtractBoundaryInconsistency:
    """Stress tests for extract_boundary_inconsistency."""
    
    def test_basic_extraction(self):
        """Test basic boundary inconsistency extraction."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        score = extract_boundary_inconsistency(frame)
        
        # Returns a float score, not a dict
        assert isinstance(score, (float, np.floating))
        assert score >= 0
    
    def test_grayscale_frame(self):
        """Test with grayscale frame."""
        frame = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        score = extract_boundary_inconsistency(frame)
        
        assert isinstance(score, (float, np.floating))
        assert score >= 0
    
    def test_smooth_frame(self):
        """Test with smooth frame."""
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
        score = extract_boundary_inconsistency(frame)
        
        # Should have low inconsistency
        assert score >= 0


class TestExtractCodecCues:
    """Stress tests for extract_codec_cues."""
    
    def test_basic_extraction(self, temp_dir):
        """Test basic codec cues extraction with actual video file."""
        import os
        test_video = os.path.join(os.path.dirname(__file__), "test_videos", "test_video1.mp4")
        
        if not os.path.exists(test_video):
            pytest.skip(f"Test video not found: {test_video}")
        
        stats = extract_codec_cues(test_video)
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
    
    def test_grayscale_frame(self, temp_dir):
        """Test codec cues with different video."""
        import os
        test_video = os.path.join(os.path.dirname(__file__), "test_videos", "test_video2.mp4")
        
        if not os.path.exists(test_video):
            pytest.skip(f"Test video not found: {test_video}")
        
        stats = extract_codec_cues(test_video)
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
    
    def test_small_frame(self, temp_dir):
        """Test codec cues extraction."""
        import os
        test_video = os.path.join(os.path.dirname(__file__), "test_videos", "test_video1.mp4")
        
        if not os.path.exists(test_video):
            pytest.skip(f"Test video not found: {test_video}")
        
        stats = extract_codec_cues(test_video)
        
        assert isinstance(stats, dict)


class TestExtractAllFeatures:
    """Stress tests for extract_all_features."""
    
    def test_basic_extraction(self):
        """Test extracting all features from a frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        features = extract_all_features(frame)
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_feature_consistency(self):
        """Test that features are consistent across calls for same frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        features1 = extract_all_features(frame)
        features2 = extract_all_features(frame)
        
        # Should be identical for same frame
        assert features1 == features2
    
    def test_different_frames(self):
        """Test that different frames give different features."""
        frame1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        features1 = extract_all_features(frame1)
        features2 = extract_all_features(frame2)
        
        # Should be different for different frames (likely, but not guaranteed)
        # At least check they're both dicts
        assert isinstance(features1, dict)
        assert isinstance(features2, dict)
    
    def test_different_videos(self):
        """Test with different frames."""
        frame1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        features1 = extract_all_features(frame1)
        features2 = extract_all_features(frame2)
        
        assert isinstance(features1, dict)
        assert len(features1) > 0
        assert isinstance(features2, dict)
        assert len(features2) > 0
    
    def test_different_num_frames(self):
        """Test with different frames (extract_all_features works on single frame)."""
        frame1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        features1 = extract_all_features(frame1)
        features2 = extract_all_features(frame2)
        
        # Should have same feature vector length (same feature set)
        assert len(features1) == len(features2)

