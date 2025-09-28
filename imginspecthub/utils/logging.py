"""
Logging utilities for results and system information.
"""

import json
import csv
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


def setup_logger(name: str = "imginspecthub", 
                level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_results(results: Dict[str, Any], 
               output_file: str,
               format_type: str = "json") -> None:
    """
    Log results to file in JSON or CSV format.
    
    Args:
        results: Results dictionary
        output_file: Output file path
        format_type: Format type ('json' or 'csv')
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Add timestamp
    results["timestamp"] = datetime.now().isoformat()
    
    if format_type.lower() == "json":
        _log_json_results(results, output_file)
    elif format_type.lower() == "csv":
        _log_csv_results(results, output_file)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def _log_json_results(results: Dict[str, Any], output_file: str) -> None:
    """Log results in JSON format."""
    # Check if file exists and load existing data
    existing_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        except (json.JSONDecodeError, Exception):
            existing_data = []
    
    # Append new results
    existing_data.append(results)
    
    # Write back to file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2, default=str)


def _log_csv_results(results: Dict[str, Any], output_file: str) -> None:
    """Log results in CSV format."""
    # Flatten nested dictionaries
    flattened = _flatten_dict(results)
    
    # Check if file exists
    file_exists = os.path.exists(output_file)
    
    # Write to CSV
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flattened.keys())
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(flattened)


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested items
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert list to string representation
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def create_result_entry(model_name: str,
                       image_path: str,
                       operation: str,
                       result: Any,
                       execution_time: float,
                       device: str,
                       additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized result entry.
    
    Args:
        model_name: Name of the model used
        image_path: Path to the input image
        operation: Operation performed (description, embedding, similarity)
        result: Operation result
        execution_time: Time taken for operation
        device: Device used for computation
        additional_info: Additional information to include
        
    Returns:
        Standardized result dictionary
    """
    entry = {
        "model_name": model_name,
        "image_path": image_path,
        "operation": operation,
        "result": result,
        "execution_time_seconds": execution_time,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_info:
        entry.update(additional_info)
    
    return entry


def log_batch_results(results: List[Dict[str, Any]],
                     output_file: str,
                     format_type: str = "json") -> None:
    """
    Log batch processing results.
    
    Args:
        results: List of result dictionaries
        output_file: Output file path
        format_type: Format type ('json' or 'csv')
    """
    if format_type.lower() == "json":
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format_type.lower() == "csv":
        if not results:
            return
        
        # Flatten all results and get all possible fieldnames
        flattened_results = [_flatten_dict(result) for result in results]
        all_fieldnames = set()
        for result in flattened_results:
            all_fieldnames.update(result.keys())
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
            writer.writeheader()
            writer.writerows(flattened_results)
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")