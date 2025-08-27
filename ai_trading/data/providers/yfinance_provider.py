from __future__ import annotations


def get_yfinance():
    """Return the ``yfinance`` module if installed, else ``None``."""
    try:
        import yfinance  # type: ignore  # local import
    except ImportError:
        return None
    return yfinance


def has_yfinance() -> bool:
    """Return ``True`` if ``yfinance`` is available."""
    try:
        import yfinance  # noqa: F401  # local import
    except ImportError:
        return False
    return True


class Provider:
    def __init__(self, session=None):
        self._session = session

    @staticmethod
    def _yf():
        yf = get_yfinance()
        if yf is None:
            raise ImportError("Provider 'yfinance' requested but the 'yfinance' package is not installed")
        return yf

    def fetch_ohlcv(self, symbol: str, interval: str = "1d", **kwargs):
        yf = self._yf()
        t = yf.Ticker(symbol)
        return t.history(period=kwargs.get("period", "1y"), interval=interval)


__all__ = ["Provider", "get_yfinance", "has_yfinance"]
