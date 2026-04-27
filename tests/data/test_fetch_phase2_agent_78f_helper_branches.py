from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch
from ai_trading.data.fetch import fallback_order


BASE_START = datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
BASE_END = BASE_START + timedelta(minutes=5)


class _Monitor:
    decision_window_seconds = 3
    min_recovery_seconds = 0

    def __init__(self) -> None:
        self.switchovers: list[tuple[str, str]] = []

    def record_switchover(self, from_provider: str, to_provider: str) -> None:
        self.switchovers.append((from_provider, to_provider))


@pytest.fixture(autouse=True)
def _reset_fetch_helper_state(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_order.reset()
    fetch._FALLBACK_WINDOWS.clear()
    fetch._FALLBACK_METADATA.clear()
    fetch._FALLBACK_UNTIL.clear()
    fetch._FALLBACK_SUPPRESS_UNTIL.clear()
    fetch._BACKUP_SKIP_UNTIL.clear()
    fetch._BACKUP_PRIMARY_PROBE_AT.clear()
    fetch._GLOBAL_BACKUP_SKIP_UNTIL.clear()
    fetch._BACKUP_USAGE_LOGGED.clear()
    fetch._CYCLE_FALLBACK_FEED.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    fetch._set_fetch_state({})
    fetch._DATA_HEALTH_STATE.update(
        {
            "consecutive_failures": 0,
            "first_failure_monotonic": 0.0,
            "status": "healthy",
            "last_error_at": None,
        },
    )
    monkeypatch.setattr(fetch, "is_safe_mode_active", lambda: False)
    yield
    fetch._set_fetch_state({})


def _frame(*, close: float | None = 101.0, timestamp: Any = BASE_START) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [timestamp],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [close],
            "volume": [1000],
        },
    )


def test_provider_degradation_resets_window_and_success_clears_gap_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[dict[str, Any]] = []
    clock = {"now": 100.0}
    monkeypatch.setattr(fetch, "_DATA_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(fetch, "_DATA_FAILURE_WINDOW", 10.0)
    monkeypatch.setattr(fetch, "monotonic_time", lambda: clock["now"])
    monkeypatch.setattr(fetch.runtime_state, "update_data_provider_state", lambda **kwargs: updates.append(kwargs))

    fetch._record_gap_ratio_state(0.42, metadata={"gap_ratio": 0.42})
    fetch._record_provider_failure_event("timeout", http_code=504)
    assert updates[-1]["status"] == "healthy"
    assert fetch._DATA_HEALTH_STATE["consecutive_failures"] == 1

    clock["now"] = 105.0
    fetch._record_provider_failure_event("timeout", http_code=504)
    assert updates[-1]["status"] == "degraded"
    assert updates[-1]["http_code"] == 504

    clock["now"] = 140.0
    fetch._record_provider_failure_event("late_error")
    assert updates[-1]["status"] == "degraded"
    assert fetch._DATA_HEALTH_STATE["consecutive_failures"] == 1

    fetch._record_provider_success_event()
    assert updates[-1]["status"] == "healthy"
    assert updates[-1]["reason"] == "recovered"
    assert fetch._current_gap_ratio() is None


def test_fallback_frame_usability_rejects_empty_invalid_and_out_of_window_frames() -> None:
    assert fetch._fallback_frame_is_usable(pd.DataFrame(), BASE_START, BASE_END) is False
    assert fetch._fallback_frame_is_usable(_frame(close=None), BASE_START, BASE_END) is False
    assert fetch._fallback_frame_is_usable(_frame(timestamp="not-a-date"), BASE_START, BASE_END) is False
    assert fetch._fallback_frame_is_usable(_frame(timestamp=BASE_END + timedelta(hours=1)), BASE_START, BASE_END) is False
    assert fetch._fallback_frame_is_usable(_frame(timestamp=BASE_START + timedelta(minutes=1)), BASE_START, BASE_END) is True


def test_mark_fallback_skips_empty_yahoo_and_same_provider_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _Monitor()
    updates: list[dict[str, Any]] = []
    logs: list[tuple[Any, ...]] = []
    monkeypatch.setattr(fetch, "provider_monitor", monitor)
    monkeypatch.setattr(fetch.runtime_state, "update_data_provider_state", lambda **kwargs: updates.append(kwargs))
    monkeypatch.setattr(fetch, "_resolve_backup_provider", lambda: ("yahoo", "yahoo"))
    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: "1" if key == "ALPACA_ALLOW_SIP" else default)
    monkeypatch.setattr(fetch, "_sip_configured", lambda: True)
    monkeypatch.setattr(fetch, "_fallback_ttl_seconds", lambda: 30)
    monkeypatch.setattr(fetch, "monotonic_time", lambda: 500.0)
    monkeypatch.setattr(fetch, "_get_cycle_id", lambda: "cycle-78f")
    monkeypatch.setattr(fetch, "log_backup_provider_used", lambda *args, **_kwargs: logs.append(args))

    empty_yahoo = pd.DataFrame()
    empty_yahoo.attrs["data_provider"] = "yahoo"
    fetch._mark_fallback(
        "AAPL",
        "1Min",
        BASE_START,
        BASE_END,
        from_provider="alpaca_iex",
        fallback_df=empty_yahoo,
        resolved_provider="yahoo",
    )
    assert fallback_order.FALLBACK_PROVIDERS == []
    assert monitor.switchovers == []
    assert updates == []

    same_provider = _frame()
    same_provider.attrs["data_provider"] = "yahoo"
    fetch._mark_fallback(
        "AAPL",
        "1Min",
        BASE_START,
        BASE_END,
        from_provider="yahoo",
        fallback_df=same_provider,
        resolved_provider="yahoo",
    )
    assert fallback_order.FALLBACK_PROVIDERS == []
    assert monitor.switchovers == []
    assert logs == []


def test_mark_fallback_prefers_frame_provider_metadata_and_suppresses_duplicate_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _Monitor()
    updates: list[dict[str, Any]] = []
    logs: list[tuple[Any, ...]] = []
    monkeypatch.setattr(fetch, "provider_monitor", monitor)
    monkeypatch.setattr(fetch.runtime_state, "update_data_provider_state", lambda **kwargs: updates.append(kwargs))
    monkeypatch.setattr(fetch, "_resolve_backup_provider", lambda: ("yahoo", "yahoo"))
    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: "1" if key == "ALPACA_ALLOW_SIP" else default)
    monkeypatch.setattr(fetch, "_sip_configured", lambda: True)
    monkeypatch.setattr(fetch, "_fallback_ttl_seconds", lambda: 30)
    monkeypatch.setattr(fetch, "monotonic_time", lambda: 600.0)
    monkeypatch.setattr(fetch, "_get_cycle_id", lambda: "cycle-78f")
    monkeypatch.setattr(fetch, "log_backup_provider_used", lambda *args, **_kwargs: logs.append(args))

    finnhub = _frame()
    finnhub.attrs["data_provider"] = "finnhub"
    finnhub.attrs["data_feed"] = "finnhub_low_latency"
    fetch._set_fetch_state({"last_fetch_attempt": {"http_status": 504}})

    fetch._mark_fallback(
        "MSFT",
        "1Min",
        BASE_START,
        BASE_END,
        from_provider="alpaca_iex",
        fallback_df=finnhub,
        reason="timeout",
    )
    fetch._mark_fallback(
        "MSFT",
        "1Min",
        BASE_START,
        BASE_END,
        from_provider="alpaca_iex",
        fallback_df=finnhub,
        reason="timeout",
    )

    metadata = fetch.get_fallback_metadata("MSFT", "1Min", BASE_START, BASE_END)
    assert metadata is not None
    assert metadata["fallback_provider"] == "finnhub"
    assert metadata["configured_fallback_provider"] == "yahoo"
    assert metadata["fallback_feed"] == "finnhub_low_latency"
    assert fallback_order.FALLBACK_PROVIDERS == ["finnhub", "finnhub"]
    assert monitor.switchovers == [("alpaca_iex", "finnhub"), ("alpaca_iex", "finnhub")]
    assert len(logs) == 1
    assert updates[-1]["active"] == "finnhub"
    assert updates[-1]["using_backup"] is True


def test_small_helpers_normalize_memos_exceptions_and_fetch_log_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    naive_ts = datetime(2026, 4, 24, 14, 30)
    memo = fetch._normalize_daily_memo({"dataframe": "df", "timestamp": naive_ts})
    assert memo == {"df": "df", "ts": naive_ts.replace(tzinfo=UTC)}
    assert fetch._normalize_daily_memo(("df", BASE_START.timestamp()))["ts"] == BASE_START
    assert fetch._normalize_daily_memo(("df", object())) is None
    assert fetch._normalize_daily_memo({"df": "df"}) is None

    class MappingMessageError(Exception):
        message = {"detail": "provider said no" * 30}

    assert fetch._safe_exception_message(None) is None
    assert fetch._safe_exception_message(MappingMessageError(), limit=20) == "provider said noprov"

    fetch._record_gap_ratio_state(0.123456)
    payload = fetch._build_fetch_log_extra(
        {"message": "x" * 250},
        symbol="spy",
        timeframe="1Min",
        attempt=0,
        status="429",
        exception=TimeoutError("too slow"),
        retry_after="4.5",
        provider="alpaca_iex",
    )

    assert payload["symbol"] == "spy"
    assert payload["provider"] == "alpaca_iex"
    assert payload["attempt"] == 1
    assert payload["http_status"] == 429
    assert payload["retry_after"] == 4.5
    assert payload["gap_ratio"] == pytest.approx(0.123456)
    assert payload["gap_ratio_pct"] == pytest.approx(12.3456)
    assert payload["exc_type"] == "TimeoutError"
    assert payload["message"] == "too slow"

    truncated = fetch._build_fetch_log_extra(
        {"message": "x" * 250},
        symbol="spy",
        timeframe=None,
        attempt=None,
    )
    assert len(truncated["message"]) == 200

    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: {"ALPACA_FALLBACK_TTL_SECONDS": "bad", "FALLBACK_TTL_SECONDS": "12"}.get(key, default))
    assert fetch._fallback_ttl_seconds() == 12
