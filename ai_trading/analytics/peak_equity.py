"""Utilities for tracking peak equity.

This module provides helpers to read the peak equity value from disk while
handling permission errors gracefully.
"""

from __future__ import annotations

from pathlib import Path

from ai_trading.logging import logger

_PEAK_EQUITY_PERMISSION_LOGGED = False


def read_peak_equity(path: str | Path) -> float:
    """Return the peak equity stored at ``path``.

    Returns ``0.0`` when the file cannot be read. If the file is not
    accessible due to permissions, a warning is logged (once) containing
    "permission denied".
    """
    global _PEAK_EQUITY_PERMISSION_LOGGED
    p = Path(path)
    try:
        return float(p.read_text().strip() or 0.0)
    except PermissionError:
        if not _PEAK_EQUITY_PERMISSION_LOGGED:
            logger.warning(
                "PEAK_EQUITY_FILE %s permission denied; skipping peak equity tracking",
                p,
            )
            _PEAK_EQUITY_PERMISSION_LOGGED = True
        return 0.0
    except (FileNotFoundError, IsADirectoryError, ValueError, OSError):
        return 0.0


__all__ = ["read_peak_equity"]
