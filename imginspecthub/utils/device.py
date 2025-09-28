"""
Device management utilities.
"""

import torch
from typing import Optional


def get_device(preferred_device: Optional[str] = None) -> str:
    """
    Get the best available device.
    
    Args:
        preferred_device: Preferred device ('cuda', 'cpu', etc.)
        
    Returns:
        Device string
    """
    if preferred_device:
        if preferred_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif preferred_device == "cpu":
            return "cpu"
        else:
            print(f"Warning: Preferred device '{preferred_device}' not available, falling back to auto-selection")
    
    # Auto-select best device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


def ensure_device_available(device: str) -> str:
    """
    Ensure device is available, fallback to CPU if not.
    
    Args:
        device: Target device
        
    Returns:
        Available device (may fallback to CPU)
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS not available, falling back to CPU")
        return "cpu"
    
    return device


def get_device_info() -> dict:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "recommended_device": get_device()
    }
    
    if info["cuda_available"]:
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(info["cuda_device_count"])]
    
    return info