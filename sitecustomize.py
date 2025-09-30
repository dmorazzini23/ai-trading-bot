"""Minimal site hook for tests.

Ensures the repository root is on ``sys.path`` for subprocess imports and,
optionally, normalises python-dotenv when ``AI_TRADING_DOTENV_GUARD=1``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import pathlib
import posixpath  # noqa: F401 - ensure import for caching
import ntpath  # noqa: F401 - ensure import for caching
import unittest.mock
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("PYTEST_DONT_REWRITE", "1")
os.environ.setdefault("PYTEST_ADDOPTS", "--assert=plain")

if os.getenv("AI_TRADING_DOTENV_GUARD") == "1":
    try:
        _dotenv = importlib.import_module("dotenv")
        getattr(_dotenv, "dotenv_values")
    except Exception:
        pass
    else:
        sys.modules["dotenv"] = _dotenv


_ESSENTIAL_NAMES = (
    "sys",
    "builtins",
    "sitecustomize",
    "importlib",
    "importlib.machinery",
    "importlib.abc",
    "importlib.util",
    "importlib._bootstrap",
    "importlib._bootstrap_external",
    "pathlib",
    "posixpath",
    "ntpath",
    "logging",
    "logging.config",
    "logging.handlers",
    "unittest.mock",
)
_ESSENTIAL_MODULES = {
    name: sys.modules.get(name)
    for name in _ESSENTIAL_NAMES
}

_ORIG_CLEAR_DICT = unittest.mock._clear_dict


def _snapshot_essential_modules():
    for name in _ESSENTIAL_NAMES:
        module = sys.modules.get(name)
        if module is not None:
            _ESSENTIAL_MODULES[name] = module


def _resolve_essential(name):
    module = _ESSENTIAL_MODULES.get(name)
    if module is None:
        module = sys.modules.get(name)
        if module is not None:
            _ESSENTIAL_MODULES[name] = module
    if module is None:
        try:
            module = importlib.import_module(name)
        except Exception:
            return None
        else:
            _ESSENTIAL_MODULES[name] = module
    return module


def _safe_clear_dict(in_dict):  # pragma: no cover - simple shim
    snapshot_required = in_dict is sys.modules
    if snapshot_required:
        _snapshot_essential_modules()
    _ORIG_CLEAR_DICT(in_dict)
    if snapshot_required:
        for key in _ESSENTIAL_NAMES:
            module = _resolve_essential(key)
            if module is not None:
                in_dict[key] = module


unittest.mock._clear_dict = _safe_clear_dict

