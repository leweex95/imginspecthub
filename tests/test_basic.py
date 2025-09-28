"""
Basic tests for imginspecthub functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw

from imginspecthub.models import get_available_models
from imginspecthub.utils.device import get_device_info
from imginspecthub.utils.image_preprocessing import load_image, preprocess_image


def create_test_image(path: str) -> None:
    """Create a simple test image."""
    img = Image.new('RGB', (300, 200), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), 'Test Image', fill='black')
    draw.rectangle([75, 75, 225, 125], outline='red', width=3)
    img.save(path)


class TestBasicFunctionality:
    """Test basic functionality."""
    
    def test_available_models(self):
        """Test that we have available models."""
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "clip" in models
    
    def test_device_info(self):
        """Test device information."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "recommended_device" in info
        assert isinstance(info["cuda_available"], bool)
    
    def test_image_loading(self):
        """Test image loading functionality."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            create_test_image(f.name)
            
            # Test loading
            img = load_image(f.name)
            assert isinstance(img, Image.Image)
            assert img.mode == "RGB"
            assert img.size == (300, 200)
            
            # Test preprocessing
            processed = preprocess_image(f.name, target_size=(224, 224))
            assert isinstance(processed, Image.Image)
            assert processed.size == (224, 224)
            
            # Cleanup
            os.unlink(f.name)
    
    def test_image_loading_nonexistent(self):
        """Test error handling for non-existent images."""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_image.jpg")


class TestImagePreprocessing:
    """Test image preprocessing utilities."""
    
    def test_preprocess_pil_image(self):
        """Test preprocessing PIL Image directly."""
        img = Image.new('RGB', (300, 200), color='blue')
        
        # Test without resizing
        processed = preprocess_image(img)
        assert isinstance(processed, Image.Image)
        assert processed.size == (300, 200)
        
        # Test with resizing
        processed = preprocess_image(img, target_size=(100, 100))
        assert processed.size == (100, 100)
    
    def test_get_image_info(self):
        """Test getting image information."""
        from imginspecthub.utils.image_preprocessing import get_image_info
        
        img = Image.new('RGB', (400, 300), color='green')
        info = get_image_info(img)
        
        assert isinstance(info, dict)
        assert info["width"] == 400
        assert info["height"] == 300
        assert info["size"] == (400, 300)
        assert info["mode"] == "RGB"


if __name__ == "__main__":
    pytest.main([__file__])