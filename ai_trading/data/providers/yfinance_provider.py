from __future__ import annotations

import types

from ai_trading.data.fetch_yf import fetch_yf_batched


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

    def fetch_ohlcv(self, symbol: str, interval: str = "1d", **kwargs):
        params: dict[str, object] = {
            "period": kwargs.get("period", "1y"),
            "interval": interval,
        }
        if "start" in kwargs or "end" in kwargs:
            params["start"] = kwargs.get("start")
            params["end"] = kwargs.get("end")

        df_map = fetch_yf_batched([symbol], **params)
        df = df_map.get(symbol)
        if df is None or df.empty:
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
        period = f"{int(limit)}d"
        df_map = fetch_yf_batched([symbol], period=period, interval="1d")
        df = df_map.get(symbol)
        if df is None or df.empty:
            return []
        df = df.tail(limit)
        bars = []
        for _, row in df.iterrows():
            bar = types.SimpleNamespace(
                o=float(row.get("open", 0.0) or 0.0),
                h=float(row.get("high", 0.0) or 0.0),
                l=float(row.get("low", 0.0) or 0.0),
                c=float(row.get("close", 0.0) or 0.0),
                v=float(row.get("volume", 0.0) or 0.0),
            )
            bars.append(bar)
        return bars


__all__ = ["Provider", "get_yfinance", "has_yfinance"]
