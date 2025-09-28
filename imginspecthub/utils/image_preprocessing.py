"""
Image preprocessing utilities.
"""

import os
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path)
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot load image {image_path}: {e}")


def preprocess_image(image: Union[str, Image.Image], 
                    target_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = True) -> Image.Image:
    """
    Preprocess image for model input.
    
    Args:
        image: Image path or PIL Image
        target_size: Target size as (width, height)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed PIL Image
    """
    # Load image if path provided
    if isinstance(image, str):
        pil_image = load_image(image)
    else:
        pil_image = image.convert("RGB")
    
    # Resize if target size specified
    if target_size:
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    
    return pil_image


def normalize_image(image: Union[Image.Image, np.ndarray], 
                   mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalize image with given mean and std (ImageNet defaults).
    
    Args:
        image: PIL Image or numpy array
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized numpy array
    """
    if isinstance(image, Image.Image):
        image_array = np.array(image).astype(np.float32) / 255.0
    else:
        image_array = image.astype(np.float32)
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
    
    # Normalize
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    normalized = (image_array - mean) / std
    
    return normalized


def resize_image_aspect_ratio(image: Union[str, Image.Image], 
                             max_size: int = 512) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Image path or PIL Image
        max_size: Maximum size for the larger dimension
        
    Returns:
        Resized PIL Image
    """
    if isinstance(image, str):
        pil_image = load_image(image)
    else:
        pil_image = image.convert("RGB")
    
    # Calculate new size maintaining aspect ratio
    width, height = pil_image.size
    
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def enhance_image(image: Union[str, Image.Image], 
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 saturation: float = 1.0) -> Image.Image:
    """
    Enhance image with brightness, contrast, and saturation adjustments.
    
    Args:
        image: Image path or PIL Image
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        saturation: Saturation factor (1.0 = no change)
        
    Returns:
        Enhanced PIL Image
    """
    from PIL import ImageEnhance
    
    if isinstance(image, str):
        pil_image = load_image(image)
    else:
        pil_image = image.convert("RGB")
    
    # Apply enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(saturation)
    
    return pil_image


def get_image_info(image: Union[str, Image.Image]) -> dict:
    """
    Get information about an image.
    
    Args:
        image: Image path or PIL Image
        
    Returns:
        Dictionary with image information
    """
    if isinstance(image, str):
        pil_image = load_image(image)
        file_size = os.path.getsize(image)
    else:
        pil_image = image
        file_size = None
    
    info = {
        "size": pil_image.size,
        "width": pil_image.size[0],
        "height": pil_image.size[1],
        "mode": pil_image.mode,
        "format": pil_image.format,
    }
    
    if file_size is not None:
        info["file_size_bytes"] = file_size
        info["file_size_mb"] = file_size / (1024 * 1024)
    
    return info