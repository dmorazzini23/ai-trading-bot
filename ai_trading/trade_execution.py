"""Compatibility wrapper for trade_execution module."""
from importlib import import_module as _import_module

_te = _import_module('trade_execution')

globals().update({k: getattr(_te, k) for k in dir(_te) if not k.startswith('_')})
