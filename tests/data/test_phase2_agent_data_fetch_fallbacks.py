from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import bars as bars_mod
from ai_trading.data import fetch as fetch_mod
from ai_trading.data import fetch_yf
from ai_trading.data.fallback import concurrency


def _yf_multiindex_frame(symbols: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([datetime(2026, 4, 24, 14, 30, tzinfo=UTC)])
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([symbols, fields])
    values: list[float] = []
    for offset, _symbol in enumerate(symbols):
        values.extend([100.0 + offset, 101.0 + offset, 99.0 + offset, 100.5 + offset, 1000 + offset])
    return pd.DataFrame([values], index=index, columns=columns)


def test_fetch_yf_batched_retries_chunks_writes_cache_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch_yf, "_yf_chunk_size", lambda: 2)
    monkeypatch.setattr(fetch_yf, "_yf_retries", lambda: 2)
    monkeypatch.setattr(fetch_yf, "_cache_read_or_none", lambda _key: None)
    monkeypatch.setattr(fetch_yf, "_sleep_backoff", lambda attempt: backoffs.append(attempt))
    monkeypatch.setattr(
        fetch_yf,
        "get_env",
        lambda key, default=None, **_kwargs: True if key == "PYTEST_YF_ALLOW_NETWORK" else default,
    )
    writes: list[tuple[str, tuple[str, ...]]] = []
    backoffs: list[int] = []
    download_calls: list[tuple[str, ...]] = []

    def fake_download(tickers: list[str], **kwargs: Any) -> pd.DataFrame:
        download_calls.append(tuple(tickers))
        assert kwargs["interval"] == "1m"
        if len(download_calls) == 1:
            raise RuntimeError("temporary yahoo outage")
        if tickers == ["AAPL", "MSFT"]:
            return _yf_multiindex_frame(tickers)
        return pd.DataFrame(
            {
                "Open": [200.0],
                "High": [201.0],
                "Low": [199.0],
                "Close": [200.5],
                "Volume": [2000],
            },
            index=pd.DatetimeIndex([datetime(2026, 4, 24, 14, 31, tzinfo=UTC)]),
        )

    def fake_cache_write(key: str, frame: pd.DataFrame) -> None:
        writes.append((key, tuple(str(column) for column in frame.columns)))

    monkeypatch.setattr(fetch_yf, "_download_batch", fake_download)
    monkeypatch.setattr(fetch_yf, "_cache_write", fake_cache_write)

    result = fetch_yf.fetch_yf_batched([" aapl", "MSFT", "aapl", "", "goog"], interval="1min")

    assert list(result) == ["AAPL", "MSFT", "GOOG"]
    assert result["AAPL"] is not None
    assert result["MSFT"] is not None
    assert result["GOOG"] is not None
    assert list(result["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
    assert download_calls == [("AAPL", "MSFT"), ("AAPL", "MSFT"), ("GOOG",)]
    assert backoffs == [0]
    assert len(writes) == 2


def test_fetch_yf_batched_uses_cache_and_keeps_missing_multiindex_symbol_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch_yf, "_yf_chunk_size", lambda: 2)
    monkeypatch.setattr(
        fetch_yf,
        "get_env",
        lambda key, default=None, **_kwargs: True if key == "PYTEST_YF_ALLOW_NETWORK" else default,
    )
    monkeypatch.setattr(fetch_yf, "_cache_read_or_none", lambda _key: _yf_multiindex_frame(["AAPL"]))
    monkeypatch.setattr(
        fetch_yf,
        "_download_batch",
        lambda *_args, **_kwargs: pytest.fail("cache hit should avoid download"),
    )

    result = fetch_yf.fetch_yf_batched(["aapl", "msft"], interval="bogus")

    assert result["AAPL"] is not None
    assert result["MSFT"] is None


def test_fetch_yf_batched_does_not_cache_malformed_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch_yf, "_yf_chunk_size", lambda: 2)
    monkeypatch.setattr(fetch_yf, "_cache_read_or_none", lambda _key: None)
    monkeypatch.setattr(
        fetch_yf,
        "get_env",
        lambda key, default=None, **_kwargs: True if key == "PYTEST_YF_ALLOW_NETWORK" else default,
    )
    writes: list[str] = []
    malformed = pd.DataFrame(
        {"Open": [100.0], "Close": [100.5]},
        index=pd.DatetimeIndex([datetime(2026, 4, 24, 14, 30, tzinfo=UTC)]),
    )

    monkeypatch.setattr(fetch_yf, "_download_batch", lambda *_args, **_kwargs: malformed)
    monkeypatch.setattr(fetch_yf, "_cache_write", lambda key, _frame: writes.append(key))

    result = fetch_yf.fetch_yf_batched(["aapl"], interval="1d")

    assert result == {"AAPL": None}
    assert writes == []


def test_safe_backup_get_bars_returns_empty_frame_when_provider_hook_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fetch_mod,
        "_backup_get_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider down")),
    )
    monkeypatch.setattr(fetch_mod, "_ensure_pandas", lambda: pd)

    frame = fetch_mod._safe_backup_get_bars(
        "AAPL",
        datetime(2026, 4, 24, tzinfo=UTC),
        datetime(2026, 4, 25, tzinfo=UTC),
        "1d",
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty


def test_get_cached_or_primary_discards_invalid_and_unauthorized_cached_feeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mod._OVERRIDE_MAP.clear()
    fetch_mod._cycle_feed_override.clear()
    fetch_mod._override_set_ts.clear()
    fetch_mod._FEED_SWITCH_CACHE.clear()
    monkeypatch.setattr(fetch_mod, "_now_ts", lambda: 100.0)
    monkeypatch.setattr(fetch_mod, "_time_now", lambda default=None: 100.0)
    monkeypatch.setattr(fetch_mod, "_is_sip_unauthorized", lambda: True)
    monkeypatch.setattr(fetch_mod, "_sip_explicitly_disabled", lambda: False)

    fetch_mod._OVERRIDE_MAP[("AAPL", "iex")] = ("bogus", 95.0)
    assert fetch_mod._get_cached_or_primary("AAPL", "iex") == "iex"
    assert ("AAPL", "iex") not in fetch_mod._OVERRIDE_MAP

    fetch_mod._cycle_feed_override["AAPL"] = "sip"
    fetch_mod._override_set_ts["AAPL"] = 99.0
    fetch_mod._FEED_SWITCH_CACHE[("AAPL", "1Min")] = ("sip", 120.0)

    assert fetch_mod._get_cached_or_primary("AAPL", "iex") == "iex"
    assert "AAPL" not in fetch_mod._cycle_feed_override
    assert ("AAPL", "1Min") not in fetch_mod._FEED_SWITCH_CACHE




