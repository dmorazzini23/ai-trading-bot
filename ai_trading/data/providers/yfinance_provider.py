from __future__ import annotations

import types


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

    def get_bars(self, symbol: str, limit: int):
        """Return recent OHLCV bars for ``symbol``.

        Parameters
        ----------
        symbol: str
            The ticker symbol to fetch.
        limit: int
            Number of bars to return.
        """
        yf = self._yf()
        t = yf.Ticker(symbol)
        try:
            df = t.history(period=f"{int(limit)}d", interval="1d").tail(limit)
        except Exception:  # pragma: no cover - network/third-party errors
            return []
        bars = []
        for _, row in df.iterrows():
            bar = types.SimpleNamespace(
                o=float(row.get("Open", 0.0)),
                h=float(row.get("High", 0.0)),
                l=float(row.get("Low", 0.0)),
                c=float(row.get("Close", 0.0)),
                v=float(row.get("Volume", 0.0)),
            )
            bars.append(bar)
        return bars


__all__ = ["Provider", "get_yfinance", "has_yfinance"]
