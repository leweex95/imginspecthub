"""
Base model interface for all image understanding models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch
from PIL import Image
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all image understanding models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            device: Device to run the model on (cuda, cpu, etc.)
        """
        self.model_name = model_name
        self.device = device or self._get_default_device()
        self.model = None
        self.processor = None
        self._is_loaded = False
    
    def _get_default_device(self) -> str:
        """Get the default device (cuda if available, else cpu)."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def get_description(self, image: Union[str, Image.Image], prompt: Optional[str] = None) -> str:
        """
        Get text description of the image.
        
        Args:
            image: Image path or PIL Image
            prompt: Optional text prompt for guided generation
            
        Returns:
            Text description of the image
        """
        pass
    
    def process_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        Process image input to PIL Image.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            PIL Image
        """
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "device": self.device,
            "loaded": self._is_loaded,
            "supports_description": True
        }