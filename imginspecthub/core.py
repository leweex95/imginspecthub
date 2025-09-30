"""
Core ImageInspector class that provides the main API interface.
"""

import time
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import numpy as np

from .models import get_model, get_available_models
from .models.base import BaseModel
from .utils.device import get_device, get_device_info
from .utils.logging import create_result_entry, log_results, setup_logger
from .utils.batch_processing import process_batch, process_directory, compare_models_on_image


class ImageInspector:
    """Main class for image inspection operations."""
    
    def __init__(self, model_name: str, device: Optional[str] = None, 
                 log_results_to_file: bool = False, log_file: Optional[str] = None):
        """
        Initialize ImageInspector.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on
            log_results_to_file: Whether to log results to file
            log_file: Optional log file path
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.log_results_to_file = log_results_to_file
        self.log_file = log_file
        
        # Set up logger
        self.logger = setup_logger("imginspecthub", log_file=log_file)
        
        # Initialize model
        self.model: Optional[BaseModel] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the specified model."""
        try:
            self.model = get_model(self.model_name, self.device)
            self.logger.info(f"Initialized {self.model_name} model on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize model {self.model_name}: {e}")
            raise
    
    def get_description(self, image: Union[str, Image.Image], 
                       prompt: Optional[str] = None,
                       log_result: bool = None) -> str:
        """
        Get text description of an image.
        
        Args:
            image: Image path or PIL Image
            prompt: Optional text prompt for guided generation
            log_result: Whether to log the result (overrides instance setting)
            
        Returns:
            Text description of the image
        """
        if not self.model or not self.model.is_loaded():
            self._load_model()
        
        start_time = time.time()
        
        try:
            description = self.model.get_description(image, prompt)
            execution_time = time.time() - start_time
            
            self.logger.info(f"Generated description in {execution_time:.2f}s")
            
            # Log result if enabled
            if log_result or (log_result is None and self.log_results_to_file):
                self._log_operation_result(
                    image, "description", description, execution_time, 
                    additional_info={"prompt": prompt}
                )
            
            return description
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Description generation failed: {e}")
            
            if log_result or (log_result is None and self.log_results_to_file):
                self._log_operation_result(
                    image, "description", None, execution_time,
                    additional_info={"prompt": prompt, "error": str(e)}
                )
            
            raise
    
    def process_batch(self, image_paths: List[str],
                     operations: List[str],
                     output_file: Optional[str] = None,
                     format_type: str = "json",
                     prompt: Optional[str] = None,
                     max_workers: int = 1,
                     show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            operations: List of operations ('description', 'embedding')
            output_file: Optional output file
            format_type: Output format ('json' or 'csv')
            prompt: Optional prompt for descriptions
            max_workers: Number of worker threads
            show_progress: Whether to show progress bar
            
        Returns:
            List of results
        """
        if not self.model or not self.model.is_loaded():
            self._load_model()
        
        return process_batch(
            model=self.model,
            image_paths=image_paths,
            operations=operations,
            output_file=output_file,
            format_type=format_type,
            prompt=prompt,
            max_workers=max_workers,
            show_progress=show_progress
        )
    
    def process_directory(self, directory: str,
                         operations: List[str],
                         output_file: Optional[str] = None,
                         format_type: str = "json",
                         prompt: Optional[str] = None,
                         max_workers: int = 1,
                         extensions: List[str] = None,
                         show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
            operations: List of operations
            output_file: Optional output file
            format_type: Output format
            prompt: Optional prompt for descriptions
            max_workers: Number of worker threads
            extensions: File extensions to include
            show_progress: Whether to show progress bar
            
        Returns:
            List of results
        """
        if not self.model or not self.model.is_loaded():
            self._load_model()
        
        return process_directory(
            model=self.model,
            directory=directory,
            operations=operations,
            output_file=output_file,
            format_type=format_type,
            prompt=prompt,
            max_workers=max_workers,
            extensions=extensions,
            show_progress=show_progress
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and system."""
        model_info = {}
        
        if self.model:
            model_info = self.model.get_model_info()
        
        device_info = get_device_info()
        
        return {
            "model": model_info,
            "device_info": device_info,
            "available_models": get_available_models()
        }
    
    def _log_operation_result(self, image: Union[str, Image.Image], 
                            operation: str, result: Any, 
                            execution_time: float,
                            additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Log operation result to file."""
        if not self.log_results_to_file:
            return
        
        image_path = image if isinstance(image, str) else "PIL_Image"
        
        result_entry = create_result_entry(
            model_name=self.model_name,
            image_path=image_path,
            operation=operation,
            result=result,
            execution_time=execution_time,
            device=self.device,
            additional_info=additional_info
        )
        
        # Default log file if not specified
        output_file = self.log_file or "imginspecthub_results.json"
        
        log_results(result_entry, output_file, "json")