"""Meta-learning utilities."""
from __future__ import annotations

from ai_trading.utils.lazy_imports import load_pandas

# Provide a canonical ``pd`` attribute for tests to patch.
pd = load_pandas()

from .core import *  # noqa: F401,F403
from .core import _import_pandas  # re-export for tests

# Re-export private helpers needed by contract tests
from .bootstrap import _generate_bootstrap_training_data  # noqa: F401
from .recovery import (
    _implement_fallback_data_recovery,
    recover_dataframe,
)  # noqa: F401

__all__ = [
    "pd",
    "_import_pandas",
    "_generate_bootstrap_training_data",
    "recover_dataframe",
    "_implement_fallback_data_recovery",
]
