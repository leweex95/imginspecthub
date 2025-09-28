"""
Model registry for managing and discovering available models.
"""

from typing import Dict, List, Type, Optional
from .base import BaseModel


class ModelRegistry:
    """Registry for managing available models."""
    
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class.
        
        Args:
            name: Model name identifier
            model_class: Model class that inherits from BaseModel
        """
        cls._models[name] = model_class
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseModel]:
        """
        Get model class by name.
        
        Args:
            name: Model name identifier
            
        Returns:
            Model class
            
        Raises:
            KeyError: If model not found
        """
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not found. Available models: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls._models.keys())
    
    @classmethod
    def create_model(cls, name: str, device: Optional[str] = None) -> BaseModel:
        """
        Create model instance by name.
        
        Args:
            name: Model name identifier
            device: Device to run the model on
            
        Returns:
            Model instance
        """
        model_class = cls.get_model_class(name)
        return model_class(model_name=name, device=device)


def register_model(name: str):
    """Decorator for registering model classes."""
    def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
        ModelRegistry.register(name, model_class)
        return model_class
    return decorator


def get_available_models() -> List[str]:
    """Get list of available model names."""
    return ModelRegistry.get_available_models()


def get_model(name: str, device: Optional[str] = None) -> BaseModel:
    """
    Get model instance by name.
    
    Args:
        name: Model name identifier
        device: Device to run the model on
        
    Returns:
        Model instance
    """
    return ModelRegistry.create_model(name, device)