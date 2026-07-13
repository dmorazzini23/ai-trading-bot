import os
import sys
import types
import asyncio
from typing import Any, cast

import pytest

pd = pytest.importorskip("pandas")
os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")

from ai_trading.data import fetch as data_fetcher
from ai_trading.data.timeutils import canonicalize_data_timeframe


class DummyClient:
    pass


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1Min", "1Min"),
        ("5Min", "5Min"),
        ("15Min", "15Min"),
        ("1Hour", "1Hour"),
        ("1Day", "1Day"),
    ],
)
def test_data_timeframe_canonicalizer_preserves_amount(
    raw: str,
    expected: str,
) -> None:
    assert canonicalize_data_timeframe(raw) == expected


def test_get_bars_reference_role_preserves_five_minute_request(monkeypatch) -> None:
    now = pd.Timestamp("2026-07-13T20:00:00Z")
    captured: dict[str, str] = {}

    def _reference(symbol, start, end, timeframe, **kwargs):
        del symbol, start, end, kwargs
        captured["timeframe"] = timeframe
        return pd.DataFrame()

    monkeypatch.setattr(data_fetcher, "_fetch_reference_bars", _reference)

    data_fetcher.get_bars(
        "AAPL",
        "5Min",
        now - pd.Timedelta(days=1),
        now,
        feed_role="reference",
    )

    assert captured["timeframe"] == "5Min"


def test_get_bars_never_none(monkeypatch):
    now = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [now],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100],
        }
    )
    monkeypatch.setattr(
        data_fetcher,
        "_alpaca_get_bars",
        lambda client, symbol, start, end, timeframe="1Day": df,
    )
    result = data_fetcher.get_bars(
        "AAPL", "1Day", now - pd.Timedelta(days=1), now, feed=cast(Any, DummyClient())
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_get_bars_requires_settings(monkeypatch):
    now = pd.Timestamp("2024-01-01", tz="UTC")
    monkeypatch.setattr(data_fetcher, "_current_settings", lambda: None)
    with pytest.raises(RuntimeError, match="Configuration is unavailable"):
        data_fetcher.get_bars(
            "AAPL", "1Day", now - pd.Timedelta(days=1), now
        )


@pytest.mark.asyncio
async def test_run_with_concurrency_returns_worker_failures() -> None:
    async def _ok() -> str:
        return "ok"

    async def _fail() -> str:
        raise RuntimeError("fetch failed")

    results, succeeded, failed = await data_fetcher.run_with_concurrency(
        2,
        [_ok(), _fail()],
    )

    assert results[0] == "ok"
    assert isinstance(results[1], RuntimeError)
    assert succeeded == 1
    assert failed == 1


@pytest.mark.asyncio
async def test_run_with_concurrency_propagates_cancellation() -> None:
    cleanup = asyncio.Event()

    async def _cancel() -> str:
        raise asyncio.CancelledError()

    async def _slow() -> str:
        try:
            await asyncio.sleep(30)
        finally:
            cleanup.set()
        return "slow"

    with pytest.raises(asyncio.CancelledError):
        await data_fetcher.run_with_concurrency(2, [_cancel(), _slow()])

    assert cleanup.is_set()
