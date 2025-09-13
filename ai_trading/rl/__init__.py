"""Compatibility shim for RL utilities."""
from .module import RLConfig, train, load, predict, _C

__all__ = ["RLConfig", "train", "load", "predict", "_C"]
