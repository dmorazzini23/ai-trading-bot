"""Centralized import management for ai_trading modules.

All dependencies are imported directly. Optional features are controlled
via Settings flags rather than import guards.
"""
from ai_trading.logging import get_logger
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
logger = get_logger(__name__)
NUMPY_AVAILABLE = True
PANDAS_AVAILABLE = True
SKLEARN_AVAILABLE = True
import ta
TA_AVAILABLE = True
logger.info('TA library loaded successfully for enhanced technical analysis')
__all__ = ['np', 'pd', 'metrics', 'RandomForestClassifier', 'train_test_split', 'ta', 'NUMPY_AVAILABLE', 'PANDAS_AVAILABLE', 'SKLEARN_AVAILABLE', 'TA_AVAILABLE']