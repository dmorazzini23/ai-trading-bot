from __future__ import annotations

import types

import pytest

pytest.importorskip("pandas")

from ai_trading.core import bot_engine as be


def _make_fetcher(monkeypatch: pytest.MonkeyPatch) -> be.DataFetcher:
    """Create a ``DataFetcher`` instance with heavy setup disabled."""

    monkeypatch.setattr(be.DataFetcher, "__post_init__", lambda self: None)
    fetcher = be.DataFetcher()
    fetcher.settings = types.SimpleNamespace()
    fetcher._warn_once = lambda *a, **k: None  # type: ignore[attr-defined]
    return fetcher


def test_normalize_stock_bars_accepts_short_ohlcv_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    fetcher = _make_fetcher(monkeypatch)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
    raw = pd.DataFrame(
        {
            "o": [101.0, 102.0],
            "h": [103.0, 104.0],
            "l": [99.5, 100.5],
            "c": [102.5, 103.5],
            "v": [1_000, 1_500],
        },
        index=index,
    )

    normalized = fetcher._normalize_stock_bars("AAPL", raw, label="Daily")
    assert normalized is not None
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.index.tz is not None

    prepared = fetcher._prepare_daily_dataframe(normalized, "AAPL")
    assert prepared is not None
    assert "timestamp" in prepared.columns
    for column in ("open", "high", "low", "close", "volume"):
        assert column in prepared.columns
