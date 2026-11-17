"""Performance metrics for trading results with numerical stability."""
from __future__ import annotations

from typing import TYPE_CHECKING
from importlib.util import find_spec

HAS_PANDAS = find_spec("pandas") is not None
HAS_NUMPY = True

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

