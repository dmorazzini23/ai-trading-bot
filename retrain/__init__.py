"""Compatibility wrapper exposing the public retraining helpers."""

from ai_trading.retrain import *  # noqa: F401,F403 - re-export public API
from ai_trading.retrain import main

__all__ = ["atomic_joblib_dump", "detect_regime", "build_parser", "main"]
