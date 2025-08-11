"""
Centralized import management for ai_trading modules.

All dependencies are imported directly. Optional features are controlled
via Settings flags rather than import guards.
"""

import logging

# Hard imports - no fallbacks
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Availability flags - all True for hard dependencies
NUMPY_AVAILABLE = True
PANDAS_AVAILABLE = True
SKLEARN_AVAILABLE = True

# TA library for optimized technical analysis - hard dependency
import ta
TA_AVAILABLE = True
logger.info("TA library loaded successfully for enhanced technical analysis")


# Export commonly used items
__all__ = [
    "np",
    "pd", 
    "metrics",
    "RandomForestClassifier",
    "train_test_split",
    "ta",
    "NUMPY_AVAILABLE",
    "PANDAS_AVAILABLE", 
    "SKLEARN_AVAILABLE",
    "TA_AVAILABLE",
]
