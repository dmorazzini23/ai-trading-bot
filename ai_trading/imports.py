"""
Centralized Import Management for AI Trading Dependencies

This module provides standardized imports for core dependencies.
All dependencies are now treated as hard requirements.

AI-AGENT-REF: Simplified to use hard dependencies only
"""

from __future__ import annotations

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

# Optional dependencies with feature flags
try:
    import talib
    TALIB_AVAILABLE = True
    logger.debug("TA-Lib imported successfully")
except ImportError:
    talib = None
    TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    logger.debug("pandas_ta imported successfully")  
except ImportError:
    ta = None
    PANDAS_TA_AVAILABLE = False
    logger.debug("pandas_ta not available")

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