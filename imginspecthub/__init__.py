"""
imginspecthub: A unified hub for running and comparing image understanding models.
"""

__version__ = "0.1.0"
__author__ = "imginspecthub contributors"
__license__ = "GPL-3.0-or-later"

from .core import ImageInspector
from .models import get_available_models

__all__ = ["ImageInspector", "get_available_models"]