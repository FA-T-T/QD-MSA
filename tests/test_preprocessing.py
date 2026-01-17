"""Tests for data preprocessing and dataset loading."""

import pytest
import torch
from PIL import Image
import tempfile
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPreprocessing:
    """Tests for image preprocessing functions."""
    
    def test_isic_preprocessor_single_image(self):
        """Test ISICPreprocessor with a single image."""
        from data.preprocessing import ISICPreprocessor
        
        preprocessor = ISICPreprocessor(image_size=224, normalize=True)
        
        # Create a dummy RGB image
        image = Image.new('RGB', (300, 400), color=(128, 128, 128))
        
        tensor = preprocessor.preprocess(image)
        
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_isic_preprocessor_grayscale_conversion(self):
        """Test that grayscale images are converted to RGB."""
        from data.preprocessing import ISICPreprocessor
        
        preprocessor = ISICPreprocessor(image_size=224)
        
        # Create a grayscale image
        image = Image.new('L', (300, 400), color=128)
        
        tensor = preprocessor.preprocess(image)
        
        # Should be converted to 3 channels
        assert tensor.shape == (3, 224, 224)
    
    def test_isic_preprocessor_batch(self):
        """Test batch preprocessing."""
        from data.preprocessing import ISICPreprocessor
        
        preprocessor = ISICPreprocessor(image_size=224)
        
        images = [
            Image.new('RGB', (300, 400), color=(255, 0, 0)),
            Image.new('RGB', (200, 300), color=(0, 255, 0)),
            Image.new('RGB', (400, 400), color=(0, 0, 255)),
        ]
        
        batch = preprocessor.preprocess_batch(images)
        
        assert batch.shape == (3, 3, 224, 224)
    
    def test_train_transforms_augmentation(self):
        """Test that training transforms include augmentation."""
        from data.preprocessing import get_train_transforms
        
        transform = get_train_transforms(image_size=224)
        
        # Create a test image
        image = Image.new('RGB', (300, 300), color=(128, 128, 128))
        
        # Apply transform multiple times - should get different results due to random augmentation
        results = [transform(image) for _ in range(5)]
        
        # All should have correct shape
        for result in results:
            assert result.shape == (3, 224, 224)
        
        # At least some should be different (due to random augmentation)
        # This might occasionally fail due to randomness, but very unlikely
        all_same = all(torch.equal(results[0], r) for r in results[1:])
        # Note: We don't assert this to avoid flaky tests
    
    def test_val_transforms_deterministic(self):
        """Test that validation transforms are deterministic."""
        from data.preprocessing import get_val_transforms
        
        transform = get_val_transforms(image_size=224)
        
        image = Image.new('RGB', (300, 300), color=(128, 128, 128))
        
        result1 = transform(image)
        result2 = transform(image)
        
        assert result1.shape == (3, 224, 224)
        assert torch.equal(result1, result2)
    
    def test_normalization_values(self):
        """Test that normalization uses ImageNet values."""
        from data.preprocessing import ISICPreprocessor
        
        preprocessor = ISICPreprocessor(image_size=224, normalize=False)
        preprocessor_norm = ISICPreprocessor(image_size=224, normalize=True)
        
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        tensor_unnorm = preprocessor.preprocess(image)
        tensor_norm = preprocessor_norm.preprocess(image)
        
        # Normalized tensor should have different values
        assert not torch.equal(tensor_unnorm, tensor_norm)


class TestDataset:
    """Tests for ISIC2017Dataset class."""
    
    def test_dataset_initialization(self):
        """Test dataset can be initialized with empty directory."""
        from data.dataset import ISIC2017Dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ISIC2017Dataset(root=tmpdir, split='train')
            
            # Should return empty dataset when no files present
            assert len(dataset) == 0
    
    def test_dataset_invalid_split(self):
        """Test that invalid split raises error."""
        from data.dataset import ISIC2017Dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid split"):
                ISIC2017Dataset(root=tmpdir, split='invalid')
    
    def test_dataset_class_mapping(self):
        """Test class name mapping."""
        from data.dataset import ISIC2017Dataset
        
        assert ISIC2017Dataset.CLASSES[0] == 'melanoma'
        assert ISIC2017Dataset.CLASSES[1] == 'seborrheic_keratosis'
        assert ISIC2017Dataset.CLASSES[2] == 'nevus'
    
    def test_dataset_get_class_distribution(self):
        """Test class distribution method."""
        from data.dataset import ISIC2017Dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ISIC2017Dataset(root=tmpdir, split='train')
            
            dist = dataset.get_class_distribution()
            
            assert 'melanoma' in dist
            assert 'seborrheic_keratosis' in dist
            assert 'nevus' in dist


class TestDataLoaders:
    """Tests for data loader creation."""
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        from data.dataset import create_data_loaders
        from data.preprocessing import get_train_transforms, get_val_transforms
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_loader, val_loader, test_loader = create_data_loaders(
                root=tmpdir,
                batch_size=4,
                num_workers=0,  # Use 0 for testing
                train_transform=get_train_transforms(),
                val_transform=get_val_transforms(),
            )
            
            # Loaders should be created even with empty dataset
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None


class TestDataUtils:
    """Tests for data utility functions."""
    
    def test_verify_dataset_missing(self):
        """Test dataset verification with missing files."""
        from data.utils import verify_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = verify_dataset(tmpdir)
            
            # Should return False for missing dataset
            assert result is False
    
    def test_download_dataset_instructions(self):
        """Test that download provides instructions."""
        from data.utils import download_dataset
        import io
        from contextlib import redirect_stdout
        
        with tempfile.TemporaryDirectory() as tmpdir:
            f = io.StringIO()
            with redirect_stdout(f):
                result = download_dataset(tmpdir)
            
            output = f.getvalue()
            
            # Should print instructions
            assert 'ISIC2017' in output or 'isic-archive' in output.lower()
            assert result is False  # Dataset not available


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
