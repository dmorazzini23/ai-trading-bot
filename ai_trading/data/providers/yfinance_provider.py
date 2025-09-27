from __future__ import annotations

import types

from ai_trading.data.fetch.normalize import normalize_ohlcv_df


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
        df = t.history(period=kwargs.get("period", "1y"), interval=interval)
        df = normalize_ohlcv_df(df)
        if len(df) == 0:
            return []
        return df

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
            df_raw = t.history(period=f"{int(limit)}d", interval="1d")
        except Exception:  # pragma: no cover - network/third-party errors
            return []
        df = normalize_ohlcv_df(df_raw)
        if len(df) == 0:
            return []
        df = df.tail(limit)
        bars = []
        for _, row in df.iterrows():
            bar = types.SimpleNamespace(
                o=float(row.get("open", row.get("Open", 0.0)) or 0.0),
                h=float(row.get("high", row.get("High", 0.0)) or 0.0),
                l=float(row.get("low", row.get("Low", 0.0)) or 0.0),
                c=float(row.get("close", row.get("Close", 0.0)) or 0.0),
                v=float(row.get("volume", row.get("Volume", 0.0)) or 0.0),
            )
            bars.append(bar)
        return bars


__all__ = ["Provider", "get_yfinance", "has_yfinance"]
