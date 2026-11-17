from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch
from ai_trading.data.fetch import fallback_order
from ai_trading.data.fetch.normalize import normalize_ohlcv_df




def _frame(ts: datetime | None = None, close: float = 101.0) -> pd.DataFrame:
    timestamp = ts or datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp": [timestamp],
            "open": [100.0],
            "high": [max(102.0, close)],
            "low": [min(99.0, close)],
            "close": [close],
            "volume": [1000],
        }
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("alpaca_sip", "sip"),
        (" delayed-sip ", "delayed_sip"),
        ("dsip", "delayed_sip"),
        ("IEX", "iex"),
    ],
)
def test_normalize_feed_value_accepts_aliases(raw: str, expected: str) -> None:
    assert fetch._normalize_feed_value(raw) == expected


@pytest.mark.parametrize("raw", ["", "alpaca_otc", object()])
def test_normalize_feed_value_rejects_unknown_values(raw: object) -> None:
    with pytest.raises(ValueError):
        fetch._normalize_feed_value(raw)


def test_record_feed_switch_cache_expires_and_clears_symbol_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 10.0}
    monkeypatch.setattr(fetch, "_now_ts", lambda: clock["now"])
    monkeypatch.setattr(fetch, "_time_now", lambda default=None: clock["now"])
    monkeypatch.setattr(fetch, "_OVERRIDE_TTL_S", 5.0)
    monkeypatch.setattr(fetch, "_sip_allowed", lambda: True)
    monkeypatch.setattr(fetch, "_is_sip_unauthorized", lambda: False)
    monkeypatch.setattr(fetch, "_sip_explicitly_disabled", lambda: False)

    fetch._record_feed_switch("AAPL", "1Min", "iex", "sip")

    clock["now"] = 16.0
    assert fetch._get_cached_or_primary("AAPL", "iex") == "iex"
    assert ("AAPL", "iex") not in fetch._OVERRIDE_MAP


def test_fallback_cache_for_cycle_keeps_iex_but_blocks_sip_when_not_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "_sip_allowed", lambda: False)
    fetch._CYCLE_FALLBACK_FEED[("cycle-1", "AAPL", "1Min")] = "iex"
    fetch._CYCLE_FALLBACK_FEED[("cycle-1", "MSFT", "1Min")] = "sip"

    assert fetch._fallback_cache_for_cycle("cycle-1", "AAPL", "1Min") == "iex"
    assert fetch._fallback_cache_for_cycle("cycle-1", "MSFT", "1Min") is None


def test_backup_get_bars_demotes_promoted_finnhub_without_key_to_yahoo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yahoo_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(fetch, "_current_settings", lambda: SimpleNamespace(backup_data_provider="yahoo"))
    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: None if key == "FINNHUB_API_KEY" else default)
    monkeypatch.setattr(fetch, "_finnhub_get_bars", lambda *_args, **_kwargs: pytest.fail("finnhub should be demoted"))
    monkeypatch.setattr(fetch, "log_backup_provider_used", lambda *_args, **_kwargs: None)

    def fake_yahoo(symbol: str, _start: Any, _end: Any, interval: str) -> pd.DataFrame:
        yahoo_calls.append((symbol, interval))
        return _frame()

    monkeypatch.setattr(fetch, "_yahoo_get_bars", fake_yahoo)
    fallback_order.promote_high_resolution("aapl", provider="finnhub")

    result = fetch._backup_get_bars("aapl", datetime(2026, 4, 24, tzinfo=UTC), datetime(2026, 4, 25, tzinfo=UTC), "1m")

    assert yahoo_calls == [("aapl", "1m")]
    assert result.attrs["data_provider"] == "yahoo"
    assert fallback_order.resolve_promoted_provider("AAPL") is None


def test_backup_get_bars_finnhub_disabled_falls_back_to_yahoo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "_current_settings", lambda: SimpleNamespace(backup_data_provider="finnhub"))
    monkeypatch.setattr(
        fetch,
        "get_env",
        lambda key, default=None, **_kwargs: "0" if key == "ENABLE_FINNHUB" else default,
    )
    monkeypatch.setattr(fetch, "_finnhub_get_bars", lambda *_args, **_kwargs: pytest.fail("disabled finnhub should not run"))
    monkeypatch.setattr(fetch, "log_backup_provider_used", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "_yahoo_get_bars", lambda *_args, **_kwargs: _frame())

    result = fetch._backup_get_bars("AAPL", datetime(2026, 4, 24, tzinfo=UTC), datetime(2026, 4, 25, tzinfo=UTC), "1d")

    assert result.attrs["data_provider"] == "yahoo"
    assert result.attrs["data_feed"] == "yahoo"


def test_backup_get_bars_splits_long_yahoo_one_minute_ranges_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2026, 4, 1, tzinfo=UTC)
    end = datetime(2026, 4, 17, tzinfo=UTC)
    calls: list[tuple[datetime, datetime]] = []
    first_ts = datetime(2026, 4, 8, 14, 30, tzinfo=UTC)

    monkeypatch.setattr(fetch, "_current_settings", lambda: SimpleNamespace(backup_data_provider="yahoo"))
    monkeypatch.setattr(fetch, "get_env", lambda _key, default=None, **_kwargs: default)
    monkeypatch.setattr(fetch, "log_backup_provider_used", lambda *_args, **_kwargs: None)

    def fake_yahoo(_symbol: str, chunk_start: datetime, chunk_end: datetime, _interval: str) -> pd.DataFrame:
        calls.append((chunk_start, chunk_end))
        close = 100.0 + len(calls)
        timestamp = first_ts if len(calls) in {1, 2} else datetime(2026, 4, 16, 14, 30, tzinfo=UTC)
        return _frame(timestamp, close=close)

    monkeypatch.setattr(fetch, "_yahoo_get_bars", fake_yahoo)

    result = fetch._backup_get_bars("AAPL", start, end, "1m")

    assert len(calls) == 3
    assert calls[0] == (start, datetime(2026, 4, 8, tzinfo=UTC))
    assert calls[-1] == (datetime(2026, 4, 15, tzinfo=UTC), end)
    assert list(result["close"]) == [102.0, 103.0]
    assert result.attrs["yf_1m_range_split"] is True
    assert result.attrs["data_provider"] == "yahoo"








def test_normalize_ohlcv_df_remaps_provider_suffixes_and_preserves_requested_columns() -> None:
    raw = pd.DataFrame(
        {
            "Timestamp": [
                datetime(2026, 4, 24, 14, 31, tzinfo=UTC),
                datetime(2026, 4, 24, 14, 30, tzinfo=UTC),
                datetime(2026, 4, 24, 14, 31, tzinfo=UTC),
            ],
            "Open_IEX": ["100", "99", "101"],
            "High_IEX": ["102", "100", "103"],
            "Low_IEX": ["98", "97", "99"],
            "Close_IEX": ["101", "100", "102"],
            "Volume_IEX": ["1000", "900", "1100"],
            "Trade Count": [10, 9, 11],
        }
    )
    raw.attrs["provider"] = "alpaca"

    normalized = normalize_ohlcv_df(raw, include_columns=("timestamp", "trade_count"))

    assert list(normalized.columns) == ["timestamp", "open", "high", "low", "close", "volume", "trade_count"]
    assert list(normalized["close"]) == [100, 102]
    assert list(normalized["trade_count"]) == [9, 11]
    assert normalized.index.is_monotonic_increasing
    assert str(normalized.index.tz) == "UTC"
    assert normalized.attrs["provider"] == "alpaca"


def test_normalize_ohlcv_df_returns_empty_timestamp_frame_for_unusable_payload() -> None:
    normalized = normalize_ohlcv_df(
        pd.DataFrame({"timestamp": ["not-a-date"], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}),
        include_columns=("timestamp",),
    )

    assert normalized.empty
    assert list(normalized.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_normalize_ohlcv_df_rejects_range_index_without_timestamp_column() -> None:
    raw = pd.DataFrame(
        {
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.2],
            "volume": [100],
        }
    )

    normalized = normalize_ohlcv_df(raw, include_columns=("timestamp",))

    assert isinstance(raw.index, pd.RangeIndex)
    assert normalized.empty
    assert list(normalized.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_normalize_ohlcv_df_rejects_reset_range_index_timestamp_column() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": [0, 1],
            "open": [10.0, 10.1],
            "high": [10.5, 10.6],
            "low": [9.5, 9.6],
            "close": [10.2, 10.3],
            "volume": [100, 120],
        }
    )

    normalized = normalize_ohlcv_df(raw, include_columns=("timestamp",))

    assert isinstance(raw.index, pd.RangeIndex)
    assert normalized.empty
    assert list(normalized.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_yahoo_get_bars_rejects_range_index_before_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame(
        {
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.2],
            "volume": [100],
        }
    )

    monkeypatch.setattr(fetch, "fetch_yf_batched", lambda symbols, **_kwargs: {symbols[0]: raw})

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    result = fetch._yahoo_get_bars("AAPL", start, start + timedelta(days=1), "1d")

    assert result.empty
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_last_minute_bar_age_ignores_future_cached_timestamp(caplog) -> None:
    fetch._MINUTE_CACHE.pop("AAPL", None)
    caplog.set_level("WARNING", logger="ai_trading.data.fetch")
    try:
        fetch._MINUTE_CACHE["AAPL"] = (
            int(datetime.now(tz=UTC).timestamp()) + 60,
            int(datetime.now(tz=UTC).timestamp()),
        )

        assert fetch.last_minute_bar_age_seconds("AAPL") is None
        assert any(record.message == "future_bar_timestamp" for record in caplog.records)
    finally:
        fetch._MINUTE_CACHE.pop("AAPL", None)
