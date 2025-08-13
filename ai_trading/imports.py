"""
Centralized Import Management for AI Trading Dependencies

This module provides standardized imports for core dependencies.
All dependencies are now treated as hard requirements.

AI-AGENT-REF: Simplified to use hard dependencies only
"""

from __future__ import annotations

import importlib
import importlib.util
import logging

# Initialize logger for import messages
logger = logging.getLogger(__name__)

# Core dependencies - all required
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Availability flags - all True since these are hard dependencies
NUMPY_AVAILABLE = True
PANDAS_AVAILABLE = True
SKLEARN_AVAILABLE = True

# Lazy imports for optional dependencies - will be resolved on first use
_talib = None
_ta = None

def resolve_talib():
    """
    Package-safe TA-Lib import using find_spec -> import_module.
    No try/except blocks; clear logging; raises if missing.
    """
    # canonical Python package name is 'talib'
    if importlib.util.find_spec("talib") is None:
        logger.error("TA-Lib (Python package 'talib') not found in environment; please install talib.")
        # Raise the canonical exception so callers can handle/exit cleanly
        raise ModuleNotFoundError("No module named 'talib'")
    return importlib.import_module("talib")

def resolve_pandas_ta():
    """
    Package-safe pandas_ta import using find_spec -> import_module.
    """
    if importlib.util.find_spec("pandas_ta") is None:
        logger.error("pandas_ta not found in environment; please install pandas_ta.")
        raise ModuleNotFoundError("No module named 'pandas_ta'")
    return importlib.import_module("pandas_ta")

def get_talib():
    """Get TA-Lib module, importing on first use."""
    global _talib
    if _talib is None:
        _talib = resolve_talib()
    return _talib

def get_pandas_ta():
    """Get pandas_ta module, importing on first use."""
    global _ta
    if _ta is None:
        _ta = resolve_pandas_ta()
    return _ta

# Property-style accessors that import on first access
class _LazyImport:
    def __init__(self, resolver):
        self._resolver = resolver
        self._module = None
        
    def __getattr__(self, name):
        if self._module is None:
            self._module = self._resolver()
        return getattr(self._module, name)
    
    def __bool__(self):
        try:
            if self._module is None:
                self._module = self._resolver()
            return True
        except ModuleNotFoundError:
            return False

# Create lazy import objects
talib = _LazyImport(resolve_talib)
ta = _LazyImport(resolve_pandas_ta)

# Set availability flags based on module specs
TALIB_AVAILABLE = importlib.util.find_spec("talib") is not None
PANDAS_TA_AVAILABLE = importlib.util.find_spec("pandas_ta") is not None
TA_AVAILABLE = TALIB_AVAILABLE or PANDAS_TA_AVAILABLE

def get_ta_lib():
    """Get TA-Lib if available, otherwise None."""
    return talib if TALIB_AVAILABLE else None

# Re-export for compatibility
__all__ = [
    "np", "pd", "LinearRegression", "StandardScaler",
    "NUMPY_AVAILABLE", "PANDAS_AVAILABLE", "SKLEARN_AVAILABLE", 
    "TALIB_AVAILABLE", "PANDAS_TA_AVAILABLE", "TA_AVAILABLE",
    "talib", "ta", "get_ta_lib"
]