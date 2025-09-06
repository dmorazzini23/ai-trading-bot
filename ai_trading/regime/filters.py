"""Helpers for working with trade regime data.

This module exposes a ``load_trades`` function that attempts to read a CSV
file containing historical trade data.  The function is defensive: missing
files or optional dependencies (``pandas``) result in an empty DataFrame
rather than an exception.  Callers can treat an empty result as "no data"
without needing to handle errors themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

log = logging.getLogger(__name__)


def load_trades(path: str | Path = "data/trades.csv", **read_csv_kwargs: Any):
    """Load trade data from ``path``.

    Parameters
    ----------
    path:
        Location of the CSV file.  Defaults to ``data/trades.csv`` relative to the
        repository root.
    **read_csv_kwargs:
        Extra keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        The parsed CSV contents.  If the file is missing or ``pandas`` is not
        installed, an empty DataFrame is returned.
    """

    try:  # Import inside function to keep pandas out of module import time
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - environment specific
        log.warning("pandas is required to load trade data: %s", exc)
        return None

    file_path = Path(path)
    if not file_path.exists():
        log.warning("trade file not found at %s", file_path)
        return pd.DataFrame()

    # Default kwargs mirror usage in tests but allow overrides
    kwargs = {"engine": "python", "on_bad_lines": "skip", "skip_blank_lines": True}
    kwargs.update(read_csv_kwargs)

    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as exc:  # pragma: no cover - unexpected parsing issues
        log.warning("failed to read trade file %s: %s", file_path, exc)
        return pd.DataFrame()
