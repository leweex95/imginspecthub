"""
Model implementations for various image understanding models.
"""

from .base import BaseModel
from .registry import ModelRegistry, get_available_models, get_model

# Import models to trigger registration
from .clip_model import CLIPModel

__all__ = ["BaseModel", "CLIPModel", "ModelRegistry", "get_available_models", "get_model"]