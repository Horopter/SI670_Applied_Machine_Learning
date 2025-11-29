"""Comprehensive unit tests for video_augmentations module."""
import pytest
import numpy as np
from PIL import Image
import torch
from lib.video_augmentations import (
    RandomRotation,
    RandomAffine,
    RandomGaussianNoise,
    RandomGaussianBlur,
    RandomCutout,
    LetterboxResize,
    build_comprehensive_frame_transforms,
)


class TestRandomRotation:
    """Stress tests for RandomRotation."""
    
    def test_basic_rotation(self):
        """Test basic rotation."""
        img = Image.new('RGB', (224, 224), color='red')
        transform = RandomRotation(degrees=10.0, p=1.0)
        
        result = transform(img)
        
        assert isinstance(result, Image.Image)
        assert result.size == img.size
    
    def test_no_rotation(self):
        """Test when p=0 (no rotation)."""
        img = Image.new('RGB', (224, 224), color='red')
        transform = RandomRotation(degrees=10.0, p=0.0)
        
        result = transform(img)
        
        assert result == img
    
    def test_extreme_angles(self):
        """Test with extreme rotation angles."""
        img = Image.new('RGB', (224, 224), color='red')
        transform = RandomRotation(degrees=180.0, p=1.0)
        
        result = transform(img)
        
        assert isinstance(result, Image.Image)
    
    def test_small_image(self):
        """Test with small image."""
        img = Image.new('RGB', (32, 32), color='red')
        transform = RandomRotation(degrees=10.0, p=1.0)
        
        result = transform(img)
        
        assert isinstance(result, Image.Image)


class TestRandomAffine:
    """Stress tests for RandomAffine."""
    
    def test_basic_affine(self):
        """Test basic affine transformation."""
        img = Image.new('RGB', (224, 224), color='red')
        transform = RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1), p=1.0)
        
        result = transform(img)
        
        assert isinstance(result, Image.Image)
    
    def test_no_affine(self):
        """Test when p=0 (no transformation)."""
        img = Image.new('RGB', (224, 224), color='red')
        transform = RandomAffine(p=0.0)
        
        result = transform(img)
        
        assert result == img
    
    def test_extreme_transforms(self):
        """Test with extreme transformation parameters."""
        img = Image.new('RGB', (224, 224), color='red')
        transform = RandomAffine(
            translate=(0.5, 0.5),
            scale=(0.5, 2.0),
            shear=45.0,
            p=1.0
        )
        
        result = transform(img)
        
        assert isinstance(result, Image.Image)


class TestLetterboxResize:
    """Stress tests for LetterboxResize."""
    
    def test_basic_resize(self):
        """Test basic letterbox resize."""
        img = Image.new('RGB', (1920, 1080), color='red')
        transform = LetterboxResize(fixed_size=224)
        
        result = transform(img)
        
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)
    
    def test_square_to_square(self):
        """Test square to square resize."""
        img = Image.new('RGB', (256, 256), color='red')
        transform = LetterboxResize(fixed_size=224)
        
        result = transform(img)
        
        assert result.size == (224, 224)
    
    def test_wide_to_square(self):
        """Test wide image to square."""
        img = Image.new('RGB', (1920, 1080), color='red')
        transform = LetterboxResize(fixed_size=224)
        
        result = transform(img)
        
        assert result.size == (224, 224)
    
    def test_tall_to_square(self):
        """Test tall image to square."""
        img = Image.new('RGB', (1080, 1920), color='red')
        transform = LetterboxResize(fixed_size=224)
        
        result = transform(img)
        
        assert result.size == (224, 224)
    
    def test_small_to_large(self):
        """Test small image to large."""
        img = Image.new('RGB', (32, 32), color='red')
        transform = LetterboxResize(fixed_size=224)
        
        result = transform(img)
        
        assert result.size == (224, 224)
    
    def test_same_size(self):
        """Test resize to same size."""
        img = Image.new('RGB', (224, 224), color='red')
        transform = LetterboxResize(fixed_size=224)
        
        result = transform(img)
        
        assert result.size == (224, 224)


class TestBuildComprehensiveFrameTransforms:
    """Stress tests for build_comprehensive_frame_transforms."""
    
    def test_basic_transforms(self):
        """Test building basic transforms."""
        transforms_compose, _ = build_comprehensive_frame_transforms(
            train=True,
            fixed_size=224,
        )
        
        assert transforms_compose is not None
        assert hasattr(transforms_compose, 'transforms')
    
    def test_train_transforms(self):
        """Test training transforms."""
        transforms_compose, _ = build_comprehensive_frame_transforms(
            train=True,
            fixed_size=224,
        )
        
        # Should include augmentations
        assert len(transforms_compose.transforms) > 2
    
    def test_val_transforms(self):
        """Test validation transforms."""
        transforms_compose, _ = build_comprehensive_frame_transforms(
            train=False,
            fixed_size=224,
        )
        
        # Should have fewer/no augmentations
        assert transforms_compose is not None
    
    def test_different_sizes(self):
        """Test with different sizes."""
        for size in [32, 64, 128, 224, 256]:
            transforms_compose, _ = build_comprehensive_frame_transforms(
                train=True,
                fixed_size=size,
            )
            
            assert transforms_compose is not None
    
    def test_no_fixed_size(self):
        """Test without fixed size."""
        transforms_compose, _ = build_comprehensive_frame_transforms(
            train=True,
            fixed_size=None,
        )
        
        assert transforms_compose is not None
    
    def test_apply_transforms(self):
        """Test applying transforms to image."""
        # Transform pipeline expects numpy array/tensor input (converts to PIL first)
        # So we pass a numpy array
        import numpy as np
        img_array = np.array(Image.new('RGB', (224, 224), color='red'))
        
        transforms_compose, post_tensor_transform = build_comprehensive_frame_transforms(
            train=True,
            fixed_size=224,
        )
        
        # Apply spatial transforms (converts array -> PIL -> PIL transforms -> tensor)
        result = transforms_compose(img_array)
        
        # Apply post-tensor transforms if any
        if post_tensor_transform is not None:
            result = post_tensor_transform(result)
        
        # Result should be Tensor after transforms
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # RGB channels
        assert result.shape[1] == 224  # Height
        assert result.shape[2] == 224  # Width
    
    def test_transforms_consistency(self):
        """Test that transforms are consistent."""
        img = Image.new('RGB', (224, 224), color='red')
        
        # Build transforms twice
        transforms1, _ = build_comprehensive_frame_transforms(
            train=True,
            fixed_size=224,
        )
        transforms2, _ = build_comprehensive_frame_transforms(
            train=True,
            fixed_size=224,
        )
        
        # Should have same structure
        assert len(transforms1.transforms) == len(transforms2.transforms)

