"""Finnhub optional dependency handling."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ai_trading.config import management as config
from ai_trading.logging import logger_once, log_finnhub_disabled

_SENT_DEPS_LOGGED: set[str] = set()


class FinnhubAPIException(Exception):
    """Thin Finnhub API error wrapper used in tests."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(str(status_code))


class _FinnhubFetcherStub:
    """Minimal stub with a ``fetch`` method; tests may monkeypatch this."""

    is_stub = True

    def fetch(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive
        raise NotImplementedError


def _build_fetcher() -> Any:
    enable = config.get_env("ENABLE_FINNHUB", "1").lower() not in ("0", "false")
    if not enable:
        if "finnhub" not in _SENT_DEPS_LOGGED:
            log_finnhub_disabled("GLOBAL")
            _SENT_DEPS_LOGGED.add("finnhub")
        return _FinnhubFetcherStub()
    try:
        import finnhub  # type: ignore
    except Exception:
        if "finnhub" not in _SENT_DEPS_LOGGED:
            logger_once.warning(
                "FINNHUB_OPTIONAL_DEP_MISSING",
                extra={"package": "finnhub-python"},
            )
            _SENT_DEPS_LOGGED.add("finnhub")
        return _FinnhubFetcherStub()
    api_key = config.get_env("FINNHUB_API_KEY")
    if not api_key:
        if "finnhub" not in _SENT_DEPS_LOGGED:
            log_finnhub_disabled("GLOBAL")
            _SENT_DEPS_LOGGED.add("finnhub")
        return _FinnhubFetcherStub()

    class FinnhubFetcher:
        """Simple wrapper around ``finnhub.Client``."""

        is_stub = False

        def __init__(self, client: Any) -> None:
            self._client = client

        def fetch(
            self,
            symbol: str,
            start: datetime,
            end: datetime,
            *,
            resolution: str = "1",
        ) -> Any:
            """Fetch candles from Finnhub and return a normalized DataFrame.

            Output columns: timestamp (UTC tz-aware), open, high, low, close, volume.
            Returns an empty DataFrame when Finnhub responds with no_data or an error.
            """
            pd = __import__("pandas")
            try:
                # finnhub-python exposes `stock_candles` (plural)
                resp = self._client.stock_candles(
                    symbol,
                    resolution,
                    int(start.timestamp()),
                    int(end.timestamp()),
                )
            except AttributeError as e:  # older/newer client mismatch
                # Fallback to the singular name if present; otherwise re-raise
                func = getattr(self._client, "stock_candle", None)
                if callable(func):
                    resp = func(symbol, resolution, int(start.timestamp()), int(end.timestamp()))
                else:
                    raise e

            # Expected shape: { s: 'ok'|'no_data', t: [...], o: [...], h: [...], l: [...], c: [...], v: [...] }
            if not isinstance(resp, dict) or resp.get("s") != "ok":
                return pd.DataFrame()
            try:
                ts = pd.to_datetime(resp.get("t", []), unit="s", utc=True)
                df = pd.DataFrame(
                    {
                        "timestamp": ts,
                        "open": resp.get("o", []),
                        "high": resp.get("h", []),
                        "low": resp.get("l", []),
                        "close": resp.get("c", []),
                        "volume": resp.get("v", []),
                    }
                )
                # Drop obviously invalid rows if lengths were inconsistent
                df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
                return df
            except Exception:
                return pd.DataFrame()

    return FinnhubFetcher(finnhub.Client(api_key))


fh_fetcher = _build_fetcher()

__all__ = ["fh_fetcher", "FinnhubAPIException", "_SENT_DEPS_LOGGED"]
