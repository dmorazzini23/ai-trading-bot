from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.data import bars as bars_mod
from ai_trading.data import provider_monitor as pm


class _AlertRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_alert(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append({"args": args, "kwargs": kwargs})


class _MetricRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float | None]] = []

    def labels(self, **labels: Any) -> "_MetricRecorder":
        self._provider = str(labels.get("provider", ""))
        return self

    def inc(self, value: float | None = None) -> None:
        self.calls.append(("inc", getattr(self, "_provider", ""), value))

    def set(self, value: float) -> None:
        self.calls.append(("set", getattr(self, "_provider", ""), value))


def _fake_env(values: dict[str, Any]):
    def fake_get_env(name: str, default: Any = None, *, cast=None, **_kwargs: Any) -> Any:
        value = values.get(name, default)
        if cast is not None and value is not None:
            return cast(value)
        return value

    return fake_get_env


def _ohlcv_frame(ts: datetime | None = None) -> Any:
    pd = bars_mod.pd
    stamp = ts or datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp": [stamp],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1234],
        }
    )


def _empty_frame() -> Any:
    return bars_mod.empty_bars_dataframe()


def test_safe_mode_failsoft_degrades_without_disabling_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alerts = _AlertRecorder()
    disables: list[str] = []
    halt_writes: list[tuple[str, dict[str, Any] | None]] = []
    clears: list[str] = []
    monitor = SimpleNamespace(alert_manager=alerts, disable=lambda provider: disables.append(provider))

    monkeypatch.setattr(pm, "provider_monitor", monitor)
    monkeypatch.setattr(pm, "monotonic_time", lambda: 1000.0)
    monkeypatch.setattr(pm, "_resolve_safe_mode_recovery_passes", lambda: 2)
    monkeypatch.setattr(pm, "_safe_mode_failsoft_enabled", lambda: True)
    monkeypatch.setattr(pm, "_failsoft_gap_ratio_limit", lambda: 0.08)
    monkeypatch.setattr(pm, "_write_halt_flag", lambda reason, metadata=None: halt_writes.append((reason, dict(metadata or {}))))
    monkeypatch.setattr(pm, "_clear_halt_flag", lambda: clears.append("cleared"))
    monkeypatch.setattr(pm, "_quote_recovery_age_limit_ms", lambda: 500.0)
    monkeypatch.setattr(pm, "_current_intraday_feed", lambda: "iex")
    monkeypatch.setattr(pm, "_gap_ratio_threshold_for_feed", lambda _feed: 0.30)
    monkeypatch.setattr(pm, "get_env", _fake_env({}))

    pm._last_halt_reason = None
    pm._last_halt_ts = 0.0
    pm._SAFE_MODE_ACTIVE = False
    pm._SAFE_MODE_REASON = None
    pm._SAFE_MODE_DEGRADED_ONLY = False
    pm._gap_event_diagnostics = {
        "alpaca": {
            "events": 3,
            "max_gap_ratio": 0.2,
            "total_missing": 9,
            "last_window_start": "2026-04-24T14:30:00+00:00",
            "last_window_end": "2026-04-24T14:33:00+00:00",
            "samples": deque([{"missing_after": 3}], maxlen=5),
        }
    }

    pm._trigger_provider_safe_mode(
        "minute_gap",
        count=3,
        metadata={
            "provider": "alpaca_iex",
            "primary_provider": "alpaca_iex",
            "fallback_provider": "yfinance",
            "used_backup": True,
            "fallback_contiguous": "contiguous",
            "gap_ratio_pct": "5",
            "missing_after": "0",
        },
    )

    assert pm.is_safe_mode_active() is True
    assert pm.safe_mode_reason() == "minute_gap"
    assert pm.safe_mode_degraded_only() is True
    assert disables == []
    assert halt_writes == []
    alert_metadata = alerts.calls[0]["kwargs"]["metadata"]
    assert alert_metadata["feeds_attempted"] == ("alpaca-iex", "alpaca", "yahoo")
    assert alert_metadata["failsoft_reason"] == "fallback_contiguous"
    assert alert_metadata["gap_metrics"]["samples"] == [{"missing_after": 3}]

    recovery_monitor = pm.ProviderMonitor(cooldown=1, max_cooldown=60, alert_manager=alerts)
    recovery_monitor._last_switchover_provider = "alpaca_iex"
    recovery_monitor.record_health_pass(
        True,
        provider="alpaca_iex",
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=25.0,
    )
    assert pm.is_safe_mode_active() is True
    recovery_monitor.record_health_pass(
        True,
        provider="alpaca_iex",
        gap_ratio=0.01,
        quote_timestamp_present=True,
        quote_age_ms=25.0,
    )

    assert pm.is_safe_mode_active() is False
    assert pm.safe_mode_reason() is None
    assert pm.safe_mode_degraded_only() is False
    assert clears == ["cleared"]


def test_provider_failure_quality_disable_and_expired_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alerts = _AlertRecorder()
    disabled_metric = _MetricRecorder()
    disable_total = _MetricRecorder()
    disable_duration = _MetricRecorder()
    failure_duration = _MetricRecorder()

    monkeypatch.setattr(pm, "provider_disabled", disabled_metric)
    monkeypatch.setattr(pm, "provider_disable_total", disable_total)
    monkeypatch.setattr(pm, "provider_disable_duration_seconds", disable_duration)
    monkeypatch.setattr(pm, "provider_failure_duration_seconds", failure_duration)
    monkeypatch.setattr(pm, "get_env", _fake_env({"PYTEST_RUNNING": "1"}))

    monitor = pm.ProviderMonitor(
        threshold=5,
        cooldown=5,
        max_cooldown=90,
        alert_manager=alerts,
    )
    callbacks: list[timedelta] = []
    monitor.register_disable_callback("yahoo", callbacks.append)

    monitor.record_failure(
        "yahoo",
        "nan_close",
        error="all closes are NaN",
        exception=ValueError("bad bars"),
        retry_after=7.5,
    )

    assert monitor.fail_counts["yahoo"] == 1
    assert monitor.is_disabled("yahoo") is True
    assert callbacks and callbacks[0].total_seconds() == 60.0
    assert ("set", "yahoo", 1) in disabled_metric.calls

    monitor.disabled_until["yahoo"] = datetime.now(UTC) - timedelta(seconds=1)
    assert monitor.is_disabled("yahoo") is False
    assert "yahoo" not in monitor.disabled_until
    assert "yahoo" not in monitor.disable_counts
    assert alerts.calls[-1]["kwargs"]["metadata"]["disable_count"] == 1
    assert any(call[0] == "inc" and call[1] == "yahoo" for call in disable_duration.calls)


def test_update_data_health_switches_to_backup_then_bias_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "DATA_COOLDOWN_SECONDS": "0",
        "DATA_PROVIDER": "",
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_BIAS_ENABLED": "true",
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_MIN_PASSES": "1",
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_COOLDOWN_SCALE": "0",
        "PYTEST_RUNNING": "1",
    }
    monkeypatch.setattr(pm, "get_env", _fake_env(values))
    monkeypatch.setattr(pm, "_env_value", lambda name, default=None, **_kwargs: values.get(name, default))
    monkeypatch.setattr(pm, "_env_text", lambda name, default="": str(values.get(name, default) or ""))
    monkeypatch.setattr(pm, "_FIRST_DECISION", True)

    monitor = pm.ProviderMonitor(cooldown=0, max_cooldown=60, primary_dwell_seconds=0)
    monitor.decision_window_seconds = 0
    monitor.min_recovery_seconds = 0
    monitor.recovery_passes_required = 4

    assert (
        monitor.update_data_health(
            "alpaca_iex",
            "yahoo",
            healthy=False,
            reason="empty_bars",
            severity="degraded",
        )
        == "yahoo"
    )
    assert monitor.active_provider("alpaca_iex", "yahoo") == "yahoo"

    assert (
        monitor.update_data_health(
            "alpaca_iex",
            "yahoo",
            healthy=True,
            reason="quotes_recovered",
            severity="good",
        )
        == "alpaca_iex"
    )
    state = monitor._pair_states[("alpaca_iex", "yahoo")]
    assert state["active"] == "alpaca_iex"
    assert state["consecutive_passes"] == 0
    assert state["primary_recovery_bias"]["required_passes"] == 1


def test_http_get_bars_exception_and_payload_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RateLimit(RuntimeError):
        status_code = 429

    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bars_mod,
        "emit_once",
        lambda _logger, key, level, event, **payload: emitted.append(
            {"key": key, "level": level, "event": event, "payload": payload}
        ),
    )
    monkeypatch.setattr(
        bars_mod,
        "_raw_http_get_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RateLimit("slow down")),
    )

    failed = bars_mod.http_get_bars("msft", "1Min", object(), object(), feed="sip")

    assert isinstance(failed, bars_mod.BarsFetchFailed)
    assert failed.symbol == "MSFT"
    assert failed.feed == "sip"
    assert failed.status == 429
    assert failed.error == "slow down"
    assert emitted[0]["payload"]["status"] == 429
    assert bars_mod._extract_status_code({"status": 418}) == 418

    frame = _ohlcv_frame()
    frame.attrs["source"] = "fixture"
    normalized = bars_mod._normalize_bars_frame(frame)
    assert normalized.attrs["source"] == "fixture"
    assert list(normalized["close"]) == [100.5]
    assert bars_mod._coerce_http_bars(failed).empty
    assert bars_mod._ensure_df(SimpleNamespace(df=object())).empty
    assert bars_mod._parse_bars(_ohlcv_frame(), "MSFT", "UTC").iloc[0]["close"] == 100.5
    assert bars_mod._parse_bars(None, "MSFT", "UTC").empty


def test_entitlement_cache_boundaries_and_test_aware_sip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars_mod._ENTITLE_CACHE.clear()

    class Client:
        account_id = "77"
        entitlements = {"iex": True}

    client = Client()
    cache_key = bars_mod._entitle_cache_key(client)
    bars_mod._ENTITLE_CACHE[cache_key] = {"feeds": {"iex"}}
    monkeypatch.setattr(
        bars_mod,
        "get_env",
        _fake_env(
            {
                "PYTEST_RUNNING": "",
                "ALPACA_ALLOW_SIP": "1",
                "ALPACA_SIP_ENTITLED": "1",
                "ALPACA_HAS_SIP": "1",
            }
        ),
    )

    assert bars_mod._ensure_entitled_feed(client, "sip") == "iex"
    converted = bars_mod._ENTITLE_CACHE[cache_key]
    assert isinstance(converted, bars_mod._EntitlementCacheEntry)
    assert converted.feeds == {"iex"}
    assert converted.resolved == "iex"

    bars_mod._ENTITLE_CACHE.clear()
    monkeypatch.setattr(bars_mod, "_ensure_entitled_feed_orig", None)
    monkeypatch.setattr(bars_mod, "_get_entitled_feeds", lambda _client: {"sip"})
    monkeypatch.setattr(
        bars_mod,
        "get_env",
        _fake_env(
            {
                "PYTEST_CURRENT_TEST": "tests/data/test_provider_bars_phase2_agent_78b.py",
                "ALPACA_ALLOW_SIP": "1",
                "ALPACA_SIP_ENTITLED": "1",
                "ALPACA_HAS_SIP": "1",
                "ALPACA_SIP_UNAUTHORIZED": "",
            }
        ),
    )

    assert bars_mod._ensure_entitled_feed(client, "iex") == "sip"
    test_entry = bars_mod._ENTITLE_CACHE[cache_key]
    assert isinstance(test_entry, bars_mod._EntitlementCacheEntry)
    assert test_entry.resolved == "sip"


def test_stale_empty_bar_fallback_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 25, 18, 0, tzinfo=UTC)
    monkeypatch.setattr(bars_mod, "rth_session_utc", lambda _day: (_ for _ in ()).throw(RuntimeError("closed")))
    assert bars_mod._minute_fallback_window(now) == (now - timedelta(minutes=20), now)

    start = datetime(2026, 4, 24, 13, 30, tzinfo=UTC)
    end = start + timedelta(hours=8)
    monkeypatch.setattr(
        bars_mod,
        "get_settings",
        lambda: SimpleNamespace(alpaca_data_feed="iex", alpaca_adjustment="raw"),
    )
    monkeypatch.setattr(bars_mod, "_fetch_daily_bars", lambda *_args, **_kwargs: _empty_frame())
    monkeypatch.setattr(
        bars_mod,
        "_get_minute_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("stale minute cache")),
    )

    with pytest.raises(ValueError, match="empty_bars"):
        bars_mod.get_daily_bars("spy", object(), start, end)

    request = SimpleNamespace(
        symbol_or_symbols=["spy"],
        timeframe="2Hour",
        start=start,
        end=end,
        feed="iex",
    )
    calls = {"client": 0, "http": 0}

    def fake_client_fetch(_client: Any, _request: Any) -> Any:
        calls["client"] += 1
        return _empty_frame()

    def fake_http_get_bars(*_args: Any, **_kwargs: Any) -> Any:
        calls["http"] += 1
        return bars_mod.BarsFetchFailed("SPY", "iex", start, status=503, error="empty")

    monkeypatch.setattr(bars_mod, "_client_fetch_stock_bars", fake_client_fetch)
    monkeypatch.setattr(bars_mod, "http_get_bars", fake_http_get_bars)
    monkeypatch.setattr(bars_mod, "_ensure_entitled_feed", lambda _client, requested: requested or "iex")
    monkeypatch.setattr(bars_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(bars_mod, "_empty_should_emit", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(bars_mod, "_empty_record", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(bars_mod, "_empty_classify", lambda **_kwargs: 20)

    recovered = bars_mod.safe_get_stock_bars(object(), request, "spy", context="empty-stale")

    assert recovered.empty
    assert calls == {"client": 2, "http": 1}
    assert request.symbol_or_symbols == ["SPY"]
