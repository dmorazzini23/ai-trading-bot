"""Compatibility wrapper exposing the public retraining helpers."""

from types import SimpleNamespace

from ai_trading.retrain import *  # noqa: F401,F403 - re-export public API
from ai_trading.retrain import main


joblib = SimpleNamespace(dump=atomic_joblib_dump)

__all__ = ["atomic_joblib_dump", "detect_regime", "build_parser", "main", "joblib"]
