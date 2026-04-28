from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch
from ai_trading.data.fetch import backoff, empty_handling
from ai_trading.data.fetch import fallback_order
from ai_trading.data.fetch.normalize import normalize_ohlcv_df


@pytest.fixture(autouse=True)
def _reset_fetch_branch_state() -> None:
    fetch._OVERRIDE_MAP.clear()
    fetch._cycle_feed_override.clear()
    fetch._override_set_ts.clear()
    fetch._FEED_SWITCH_CACHE.clear()
    fetch._CYCLE_FALLBACK_FEED.clear()
    fetch._BACKUP_USAGE_LOGGED.clear()
    fetch._YF_WARNING_CACHE.clear()
    fetch._BOOTSTRAP_BACKUP_REASON = None
    backoff._PROVIDER_COOLDOWNS.clear()
    backoff._EMPTY_BAR_COUNTS.clear()
    backoff._SKIPPED_SYMBOLS.clear()
    empty_handling._RETRY_COUNTS.clear()
    fallback_order.reset()


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


def test_backoff_fetch_feed_uses_http_fallback_while_primary_on_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = ("AAPL", "1Min")
    start = datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    fallback_frame = _frame()

    backoff._PROVIDER_COOLDOWNS[key] = 200.0
    monkeypatch.setattr(backoff, "monotonic_time", lambda: 100.0)
    monkeypatch.setattr(backoff, "max_data_fallbacks", lambda: 1)
    monkeypatch.setattr(backoff, "_provider_decision_window", lambda: 30.0)
    monkeypatch.setattr(backoff, "_disable_primary_provider", lambda: None)
    monkeypatch.setattr(backoff, "_fetch_bars", lambda *_args, **_kwargs: pytest.fail("primary should be skipped"))
    monkeypatch.setattr(backoff, "_http_fallback", lambda *_args, **_kwargs: fallback_frame)

    result = backoff._fetch_feed("AAPL", start, end, "1Min", feed="iex")

    assert result is fallback_frame
    assert key not in backoff._EMPTY_BAR_COUNTS
    assert key not in backoff._SKIPPED_SYMBOLS
    assert backoff._PROVIDER_COOLDOWNS[key] == 200.0


def test_empty_handling_retries_then_clears_counts_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []
    monkeypatch.setattr(empty_handling, "is_market_open", lambda: True)
    monkeypatch.setattr(empty_handling.time, "sleep", lambda delay: sleeps.append(delay))

    def fetch_fn() -> pd.DataFrame:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise fetch.EmptyBarsError("empty")
        return _frame()

    result = empty_handling.fetch_with_retries("AAPL", "1Min", fetch_fn, [0.25])

    assert not result.empty
    assert sleeps == [0.25]
    assert empty_handling._RETRY_COUNTS == {}


def test_empty_handling_no_session_reraises_and_clears_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(empty_handling, "is_market_open", lambda: True)

    def fetch_fn() -> pd.DataFrame:
        raise fetch.EmptyBarsError("empty")

    with pytest.raises(fetch.EmptyBarsError):
        empty_handling.fetch_with_retries(
            "MSFT",
            "1Min",
            fetch_fn,
            [0.01],
            window_has_trading_session=lambda: False,
        )

    assert ("MSFT", "1Min") not in empty_handling._RETRY_COUNTS


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
