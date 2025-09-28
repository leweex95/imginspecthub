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
    
    @abstractmethod
    def get_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Get image embedding.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Image embedding as numpy array
        """
        pass
    
    def get_similarity_score(self, image1: Union[str, Image.Image], 
                           image2: Union[str, Image.Image]) -> float:
        """
        Get similarity score between two images.
        
        Args:
            image1: First image path or PIL Image
            image2: Second image path or PIL Image
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)
        
        # Cosine similarity
        dot_product = np.dot(emb1.flatten(), emb2.flatten())
        norm1 = np.linalg.norm(emb1.flatten())
        norm2 = np.linalg.norm(emb2.flatten())
        
        return float(dot_product / (norm1 * norm2))
    
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
            "supports_description": True,
            "supports_embedding": True,
            "supports_similarity": True
        }