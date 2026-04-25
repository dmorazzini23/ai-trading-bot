from __future__ import annotations

from datetime import UTC, datetime, timedelta
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch


@pytest.fixture(autouse=True)
def _clear_fetch_phase2_state() -> Generator[None, None, None]:
    fetch._ALPACA_FAILURE_EVENTS.clear()
    fetch._ALPACA_CONSECUTIVE_FAILURES.clear()
    fetch._FALLBACK_WINDOWS.clear()
    fetch._FALLBACK_METADATA.clear()
    fetch._FALLBACK_UNTIL.clear()
    yield
    fetch._ALPACA_FAILURE_EVENTS.clear()
    fetch._ALPACA_CONSECUTIVE_FAILURES.clear()
    fetch._FALLBACK_WINDOWS.clear()
    fetch._FALLBACK_METADATA.clear()
    fetch._FALLBACK_UNTIL.clear()


def test_daily_memo_normalization_accepts_mapping_tuple_and_epoch() -> None:
    frame = pd.DataFrame({"close": [1.0]})
    naive = datetime(2026, 4, 25, 12, 0)
    epoch = datetime(2026, 4, 25, 12, tzinfo=UTC).timestamp()

    mapping = fetch._normalize_daily_memo({"dataframe": frame, "timestamp": naive})
    tuple_payload = fetch._normalize_daily_memo((frame, epoch))

    assert mapping == {"df": frame, "ts": naive.replace(tzinfo=UTC)}
    assert tuple_payload == {"df": frame, "ts": datetime(2026, 4, 25, 12, tzinfo=UTC)}
    assert fetch._normalize_daily_memo({"df": frame}) is None
    assert fetch._normalize_daily_memo("bad") is None
    assert fetch._coerce_memo_ts("not-a-date") is None


def test_rate_limit_cooldown_prefers_retry_after_and_clamps_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert fetch._rate_limit_cooldown(SimpleNamespace(headers={"Retry-After": "12.5"})) == 12.5
    assert fetch._rate_limit_cooldown(SimpleNamespace(retry_after="4")) == 4.0

    monkeypatch.setattr(fetch, "get_env", lambda *_args, **_kwargs: "-10")
    assert fetch._rate_limit_cooldown(SimpleNamespace(headers={"Retry-After": "bad"})) == 0.0


def test_consecutive_and_windowed_alpaca_failures_drive_yahoo_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "_yahoo_failure_threshold", lambda: 2)
    monkeypatch.setattr(fetch, "_yahoo_failure_window_seconds", lambda: 10.0)
    monkeypatch.setattr(fetch, "monotonic_time", lambda: 101.0)
    monkeypatch.setattr(fetch, "_ALPACA_CONSECUTIVE_FAILURE_THRESHOLD", 3)
    monkeypatch.setattr(
        fetch,
        "provider_monitor",
        SimpleNamespace(
            is_disabled=lambda _provider: False,
            fail_counts={"alpaca": 0},
            threshold=10,
        ),
    )

    assert fetch._yahoo_fallback_allowed("AAPL", "1Min") is False
    assert fetch._record_alpaca_failure_event("AAPL", "1Min", now=100.0) == 1
    assert fetch._record_alpaca_failure_event("AAPL", "1Min", now=101.0) == 2
    assert fetch._yahoo_fallback_allowed("AAPL", "1Min", force=True) is True
    assert fetch._yahoo_fallback_allowed("AAPL", "1Min") is True

    fetch._clear_alpaca_failure_events("AAPL", "1Min")
    assert fetch._alpaca_failure_count("AAPL", now=102.0) == 0
    assert fetch._consecutive_failure_count("AAPL", "1Min") == 0


def test_yahoo_fallback_allowed_honors_provider_monitor_disable_and_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "_ALPACA_CONSECUTIVE_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(
        fetch,
        "provider_monitor",
        SimpleNamespace(is_disabled=lambda provider: provider == "alpaca", fail_counts={}, threshold=3),
    )
    assert fetch._yahoo_fallback_allowed("MSFT", "1Min") is True

    monkeypatch.setattr(fetch, "_ALPACA_CONSECUTIVE_FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(
        fetch,
        "provider_monitor",
        SimpleNamespace(is_disabled=lambda _provider: False, fail_counts={"alpaca": 3}, threshold=3),
    )
    assert fetch._yahoo_fallback_allowed("MSFT", "1Min") is True


@pytest.mark.parametrize(
    ("metadata", "hint", "expected_reason"),
    [
        ({"status_code": 403, "feed": "sip", "message": "subscription required"}, None, "unauthorized_sip"),
        ({"status_code": 503}, None, "server_error"),
        ({"status_code": 400}, None, "bad_request"),
        ({"gap_ratio_pct": 8.0, "gap_ratio_limit_pct": 5.0}, None, "gap_ratio_exceeded"),
        ({}, "safe_mode entered", "provider_safe_mode"),
        ({}, "forced backup provider", "forced_backup_provider"),
        ({}, "configured_source_override", "configured_source_override"),
        ({"exception": TimeoutError("slow")}, None, "timeout"),
        ({"exception": ConnectionError("down")}, None, "upstream_unavailable"),
    ],
)
def test_classify_fallback_reason_covers_operational_categories(
    metadata: dict[str, Any],
    hint: str | None,
    expected_reason: str,
) -> None:
    reason, details = fetch._classify_fallback_reason(hint, metadata)

    assert reason == expected_reason
    assert isinstance(details, dict)


def test_build_backup_usage_extra_normalizes_symbols_and_merges_detail() -> None:
    payload = fetch._build_backup_usage_extra(
        None,
        None,
        None,
        "primary_probe deferred",
        {"fallback_provider": "yahoo", "symbol": "spy", "timeframe": "1Min"},
    )

    assert payload["provider"] == "yahoo"
    assert payload["symbol"] == "spy"
    assert payload["timeframe"] == "1Min"
    assert payload["reason"] == "primary_probe_deferred"
    assert payload["detail"] == "primary_probe deferred"


def test_fallback_frame_usable_validates_close_and_timestamp_window() -> None:
    start = datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    good = pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=1), start + timedelta(minutes=4)],
            "close": [100.0, 101.0],
        }
    )
    stale = pd.DataFrame({"timestamp": [start - timedelta(hours=1)], "close": [100.0]})
    all_nan_close = pd.DataFrame({"timestamp": [start], "close": [None]})

    assert fetch._fallback_frame_is_usable(good, start, end) is True
    assert fetch._fallback_frame_is_usable(stale, start, end) is False
    assert fetch._fallback_frame_is_usable(all_nan_close, start, end) is False
    assert fetch._fallback_frame_is_usable([], start, end) is True
    assert fetch._frame_has_rows(pd.DataFrame()) is False
    assert fetch._frame_has_rows([1]) is True


def test_gap_ratio_limit_env_precedence_and_iex_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    values: dict[str, str] = {"AI_TRADING_GAP_RATIO_LIMIT": "0.02"}
    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: values.get(key, default))
    assert fetch._resolve_gap_ratio_limit(default_ratio=0.05) == 0.02

    values.clear()
    values["GAP_RATIO_MAX_PCT"] = "7.5"
    assert fetch._resolve_gap_ratio_limit(default_ratio=0.05) == 0.075

    values.clear()
    values["DATA_MAX_GAP_RATIO_BPS"] = "25"
    assert fetch._resolve_gap_ratio_limit(default_ratio=0.05) == 0.0025

    values.clear()
    assert fetch._resolve_gap_ratio_limit(default_ratio=0.01, feed="alpaca_iex") == 0.30
    assert fetch._format_gap_ratio_reason(0.1234, 0.05) == "gap_ratio=12.34% > limit=5.00%"
