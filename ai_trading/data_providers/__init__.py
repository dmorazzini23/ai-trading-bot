"""Optional data provider helpers."""

from __future__ import annotations


def get_yfinance():
    """Return the ``yfinance`` module if installed, else ``None``."""
    try:
        import yfinance  # type: ignore
    except ImportError:
        return None
    return yfinance


def has_yfinance() -> bool:
    """Return ``True`` if ``yfinance`` is available."""
    try:
        import yfinance  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["get_yfinance", "has_yfinance"]

