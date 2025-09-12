"""Meta-learning utilities."""
from __future__ import annotations

# Provide a ``pd`` attribute for tests to patch. Fall back to the internal
# pandas facade when pandas is unavailable.
try:  # pragma: no cover - exercised in environments with pandas installed
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from ai_trading.utils import pandas_facade as pd  # type: ignore

from .core import *  # noqa: F401,F403
from .core import _import_pandas  # re-export for tests

# Re-export private helpers needed by contract tests
from .bootstrap import _generate_bootstrap_training_data  # noqa: F401
from .recovery import _implement_fallback_data_recovery  # noqa: F401

__all__ = [
    "pd",
    "_import_pandas",
    "_generate_bootstrap_training_data",
    "_implement_fallback_data_recovery",
]
