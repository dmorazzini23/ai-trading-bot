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
    "pathlib",
    "ntpath",
    "posixpath",
    "importlib",
    "importlib.machinery",
    "importlib.abc",
    "importlib.util",
    "unittest.mock",
    "sys",
    "builtins",
)
_ESSENTIAL_MODULES = {
    name: sys.modules.get(name)
    for name in _ESSENTIAL_NAMES
    if sys.modules.get(name) is not None
}

_ORIG_CLEAR_DICT = unittest.mock._clear_dict


def _safe_clear_dict(in_dict):  # pragma: no cover - simple shim
    _ORIG_CLEAR_DICT(in_dict)
    if in_dict is sys.modules:
        for key, module in _ESSENTIAL_MODULES.items():
            if module is not None:
                in_dict.setdefault(key, module)


unittest.mock._clear_dict = _safe_clear_dict

