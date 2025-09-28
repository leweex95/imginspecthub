"""
Batch processing utilities for handling multiple images.
"""

import os
import time
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..models.base import BaseModel
from .logging import create_result_entry, log_batch_results


def find_images(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Find all image files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    image_paths = []
    extensions = [ext.lower() for ext in extensions]
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)


def process_single_image(model: BaseModel, 
                        image_path: str,
                        operations: List[str],
                        prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single image with specified operations.
    
    Args:
        model: Model instance
        image_path: Path to image
        operations: List of operations to perform
        prompt: Optional prompt for description
        
    Returns:
        Results dictionary
    """
    results = {
        "image_path": image_path,
        "model_name": model.model_name,
        "device": model.device,
        "operations": {}
    }
    
    for operation in operations:
        start_time = time.time()
        
        try:
            if operation == "description":
                result = model.get_description(image_path, prompt)
            elif operation == "embedding":
                result = model.get_embedding(image_path).tolist()  # Convert to list for JSON
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            execution_time = time.time() - start_time
            
            results["operations"][operation] = {
                "result": result,
                "execution_time": execution_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            results["operations"][operation] = {
                "result": None,
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }
    
    return results


def process_batch(model: BaseModel,
                 image_paths: List[str],
                 operations: List[str],
                 output_file: Optional[str] = None,
                 format_type: str = "json",
                 prompt: Optional[str] = None,
                 max_workers: int = 1,
                 show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Process multiple images in batch.
    
    Args:
        model: Model instance
        image_paths: List of image paths
        operations: List of operations to perform
        output_file: Optional output file for results
        format_type: Output format ('json' or 'csv')
        prompt: Optional prompt for description
        max_workers: Number of worker threads (1 for sequential)
        show_progress: Whether to show progress bar
        
    Returns:
        List of result dictionaries
    """
    if not model.is_loaded():
        print(f"Loading model {model.model_name}...")
        model.load_model()
    
    results = []
    
    # Progress bar setup
    progress_bar = tqdm(total=len(image_paths), disable=not show_progress,
                       desc=f"Processing with {model.model_name}")
    
    def process_image_wrapper(image_path: str) -> Dict[str, Any]:
        """Wrapper for processing single image with progress update."""
        result = process_single_image(model, image_path, operations, prompt)
        progress_bar.update(1)
        return result
    
    if max_workers == 1:
        # Sequential processing
        for image_path in image_paths:
            result = process_image_wrapper(image_path)
            results.append(result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(process_single_image, model, path, operations, prompt): path 
                for path in image_paths
            }
            
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                progress_bar.update(1)
    
    progress_bar.close()
    
    # Save results if output file specified
    if output_file:
        log_batch_results(results, output_file, format_type)
        print(f"Results saved to {output_file}")
    
    return results


def process_directory(model: BaseModel,
                     directory: str,
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
        model: Model instance
        directory: Directory containing images
        operations: List of operations to perform
        output_file: Optional output file for results
        format_type: Output format ('json' or 'csv')
        prompt: Optional prompt for description
        max_workers: Number of worker threads
        extensions: List of file extensions to include
        show_progress: Whether to show progress bar
        
    Returns:
        List of result dictionaries
    """
    # Find all images in directory
    image_paths = find_images(directory, extensions)
    
    if not image_paths:
        print(f"No images found in directory: {directory}")
        return []
    
    print(f"Found {len(image_paths)} images in {directory}")
    
    # Process batch
    return process_batch(
        model=model,
        image_paths=image_paths,
        operations=operations,
        output_file=output_file,
        format_type=format_type,
        prompt=prompt,
        max_workers=max_workers,
        show_progress=show_progress
    )


def compare_models_on_image(models: List[BaseModel],
                          image_path: str,
                          operations: List[str],
                          prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare multiple models on a single image.
    
    Args:
        models: List of model instances
        image_path: Path to image
        operations: List of operations to perform
        prompt: Optional prompt for description
        
    Returns:
        Comparison results dictionary
    """
    comparison_results = {
        "image_path": image_path,
        "models": {},
        "summary": {
            "total_models": len(models),
            "operations": operations
        }
    }
    
    for model in models:
        if not model.is_loaded():
            model.load_model()
        
        model_results = process_single_image(model, image_path, operations, prompt)
        comparison_results["models"][model.model_name] = model_results
    
    return comparison_results


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary report from batch processing results.
    
    Args:
        results: List of processing results
        
    Returns:
        Summary report dictionary
    """
    if not results:
        return {"error": "No results to summarize"}
    
    total_images = len(results)
    successful_operations = {}
    failed_operations = {}
    total_time_by_operation = {}
    
    for result in results:
        for operation, op_result in result.get("operations", {}).items():
            if operation not in successful_operations:
                successful_operations[operation] = 0
                failed_operations[operation] = 0
                total_time_by_operation[operation] = 0.0
            
            if op_result.get("success", False):
                successful_operations[operation] += 1
            else:
                failed_operations[operation] += 1
            
            total_time_by_operation[operation] += op_result.get("execution_time", 0.0)
    
    summary = {
        "total_images": total_images,
        "operations_summary": {},
        "performance": {}
    }
    
    for operation in successful_operations:
        success_rate = successful_operations[operation] / total_images
        avg_time = total_time_by_operation[operation] / total_images
        
        summary["operations_summary"][operation] = {
            "successful": successful_operations[operation],
            "failed": failed_operations[operation],
            "success_rate": success_rate,
            "total_time": total_time_by_operation[operation],
            "average_time": avg_time
        }
    
    return summary