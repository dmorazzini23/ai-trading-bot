from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch


BASE_START = datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
BASE_END = BASE_START + timedelta(minutes=5)
PAST_START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
PAST_END = PAST_START + timedelta(minutes=5)


class _Resp:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any | None = None,
        *,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.headers = {"Content-Type": "application/json", "x-request-id": f"req-{status_code}"}
        if headers:
            self.headers.update(headers)
        self.text = json.dumps(self._payload) if text is None else text

    def json(self) -> Any:
        return self._payload


class _Session:
    def __init__(self, *items: Any) -> None:
        self.items = list(items)
        self.calls: list[dict[str, Any]] = []
        self.last_request: Any | None = None

    def get(self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: Any = None) -> Any:
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {}), "timeout": timeout})
        if not self.items:
            raise AssertionError("unexpected HTTP call")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _Metric:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def labels(self, **labels: Any) -> "_Metric":
        self.calls.append(("labels", labels, ""))
        return self

    def inc(self, value: float = 1.0) -> None:
        self.calls.append(("inc", {"value": value}, ""))

    def set(self, value: float) -> None:
        self.calls.append(("set", {"value": value}, ""))


class _Metrics:
    def __init__(self) -> None:
        self.incr_calls: list[tuple[str, float, dict[str, str] | None]] = []
        self.unauthorized = 0
        self.timeout = 0
        self.empty_payload = 0
        self.empty_fallback = 0

    def incr(self, metric: str, *, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        self.incr_calls.append((metric, value, tags))


class _Alerts:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def create_alert(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _Monitor:
    min_recovery_seconds = 0
    max_cooldown = 60
    cooldown = 0
    threshold = 1
    decision_window_seconds = 3

    def __init__(self) -> None:
        self.disabled_until: dict[str, datetime] = {}
        self.fail_counts: dict[str, int] = {}
        self.disable_counts: dict[str, int] = {}
        self.outage_start: dict[str, datetime] = {}
        self.disabled_since: dict[str, datetime] = {}
        self.alert_manager = _Alerts()
        self.failures: list[tuple[str, str, dict[str, Any]]] = []
        self.successes: list[str] = []
        self.switchovers: list[tuple[str, str]] = []
        self.health_updates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.active_choice: str | None = None

    def register_disable_callback(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def safe_mode_cycle_marker(self) -> tuple[int, str]:
        return (1, "provider_safe_mode")

    def record_failure(self, provider: str, reason: str = "", *args: Any, **kwargs: Any) -> None:
        del args
        self.fail_counts[provider] = self.fail_counts.get(provider, 0) + 1
        self.failures.append((provider, reason, kwargs))

    def record_success(self, provider: str) -> None:
        self.successes.append(provider)
        self.disabled_until.pop(provider, None)

    def record_switchover(self, from_provider: str, to_provider: str) -> None:
        self.switchovers.append((from_provider, to_provider))

    def disable(self, provider: str, duration: float | None = None, **_kwargs: Any) -> None:
        seconds = 60.0 if duration is None else float(duration)
        self.disabled_until[provider] = datetime.now(tz=UTC) + timedelta(seconds=seconds)
        self.disable_counts[provider] = self.disable_counts.get(provider, 0) + 1

    def is_disabled(self, provider: str) -> bool:
        until = self.disabled_until.get(provider)
        return isinstance(until, datetime) and until > datetime.now(tz=UTC)

    def active_provider(self, primary: str, backup: str) -> str:
        return self.active_choice or primary

    def update_data_health(self, *args: Any, **kwargs: Any) -> None:
        self.health_updates.append((args, kwargs))


def _direct_session_get(session: _Session, url: str, *, params: dict[str, Any], headers: dict[str, str], timeout: Any) -> Any:
    return session.get(url, params=params, headers=headers, timeout=timeout)


def _bar_payload(ts: datetime = BASE_START, close: float = 101.0) -> dict[str, Any]:
    return {"t": ts.isoformat().replace("+00:00", "Z"), "o": 100.0, "h": 102.0, "l": 99.0, "c": close, "v": 1000}


def _frame(ts: datetime = BASE_START, *, close: float = 101.0, provider: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": [ts],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [close],
            "volume": [1000],
        },
    )
    if provider:
        df.attrs["data_provider"] = provider
        df.attrs["data_feed"] = provider
    return df


@pytest.fixture
def fetch_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    env: dict[str, Any] = {
        "ALPACA_API_KEY": "key",
        "ALPACA_SECRET_KEY": "secret",
        "ALPACA_ALLOW_SIP": "1",
        "ALPACA_HAS_SIP": "1",
        "ENABLE_HTTP_FALLBACK": "1",
        "BACKUP_DATA_PROVIDER": "yahoo",
        "AI_TRADING_GLOBAL_BACKUP_SKIP_ENABLED": "0",
        "FETCH_BARS_MAX_RETRIES": "2",
        "PYTEST_RUNNING": "1",
    }

    def fake_get_env(key: str, default: Any = None, **kwargs: Any) -> Any:
        value = env.get(key, default)
        cast = kwargs.get("cast")
        if cast is not None and value is not None:
            try:
                return cast(value)
            except (TypeError, ValueError):
                return default
        return value

    monitor = _Monitor()
    metrics = _Metrics()
    metric = _Metric()

    stores = (
        fetch._OVERRIDE_MAP,
        fetch._cycle_feed_override,
        fetch._override_set_ts,
        fetch._FEED_SWITCH_CACHE,
        fetch._CYCLE_FALLBACK_FEED,
        fetch._BACKUP_USAGE_LOGGED,
        fetch._YF_WARNING_CACHE,
        fetch._FALLBACK_WINDOWS,
        fetch._FALLBACK_METADATA,
        fetch._FALLBACK_UNTIL,
        fetch._FALLBACK_SUPPRESS_UNTIL,
        fetch._BACKUP_SKIP_UNTIL,
        fetch._BACKUP_PRIMARY_PROBE_AT,
        fetch._GLOBAL_BACKUP_SKIP_UNTIL,
        fetch._ALPACA_FAILURE_EVENTS,
        fetch._ALPACA_CONSECUTIVE_FAILURES,
        fetch._ALPACA_EMPTY_ERROR_COUNTS,
        fetch._IEX_EMPTY_COUNTS,
        fetch._FEED_FAILOVER_ATTEMPTS,
        fetch._FEED_OVERRIDE_BY_TF,
    )
    for store in stores:
        store.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    fetch._EMPTY_BAR_COUNTS.clear()
    fetch._FEED_SWITCH_LOGGED.clear()
    fetch._FEED_SWITCH_HISTORY.clear()
    fetch._BOOTSTRAP_BACKUP_REASON = None

    monkeypatch.setattr(fetch, "get_env", fake_get_env)
    monkeypatch.setattr(fetch, "_current_settings", lambda: SimpleNamespace(backup_data_provider=env.get("BACKUP_DATA_PROVIDER", "yahoo"), data_provider="alpaca"))
    monkeypatch.setattr(fetch, "get_backup_data_provider", lambda: str(env.get("BACKUP_DATA_PROVIDER", "yahoo")))
    monkeypatch.setattr(fetch, "provider_monitor", monitor)
    monkeypatch.setattr(fetch, "metrics", metrics)
    monkeypatch.setattr(fetch, "provider_fallback", metric)
    monkeypatch.setattr(fetch, "provider_disabled", metric)
    monkeypatch.setattr(fetch, "provider_disable_total", metric, raising=False)
    monkeypatch.setattr(fetch, "alpaca_auth_headers", lambda: {"APCA-API-KEY-ID": "key", "APCA-API-SECRET-KEY": "secret"})
    monkeypatch.setattr(fetch, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch, "_last_complete_minute", lambda _pd=None: BASE_END + timedelta(minutes=1))
    monkeypatch.setattr(fetch.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 2)
    monkeypatch.setattr(fetch, "provider_priority", lambda: ("alpaca_iex", "alpaca_sip", "yahoo"))
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: True)
    monkeypatch.setattr(fetch, "resolve_alpaca_feed", lambda feed=None: (str(feed or "iex").replace("alpaca_", "")))
    monkeypatch.setattr(fetch, "_get_alpaca_data_base_url", lambda: "https://data.alpaca.test")
    monkeypatch.setattr(fetch, "get_alpaca_data_base_url", lambda: "https://data.alpaca.test")
    monkeypatch.setattr(fetch, "_sip_allowed", lambda: True)
    monkeypatch.setattr(fetch, "_sip_configured", lambda: True)
    monkeypatch.setattr(fetch, "_is_sip_unauthorized", lambda: False)
    monkeypatch.setattr(fetch, "_sip_explicitly_disabled", lambda: False)
    monkeypatch.setattr(fetch, "_enable_http_fallback", lambda: True)
    monkeypatch.setattr(fetch, "_http_fallback_permitted", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(fetch, "_fetch_bars_max_retries", lambda: 2)
    monkeypatch.setattr(fetch, "_fetch_bars_backoff_base", lambda: 1.0)
    monkeypatch.setattr(fetch, "_fetch_bars_backoff_cap", lambda: 1.0)
    monkeypatch.setattr(fetch, "_safe_empty_should_emit", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(fetch, "_safe_empty_record", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(fetch, "_safe_empty_classify", lambda **_kwargs: logging.INFO)
    monkeypatch.setattr(fetch, "_symbol_exists", lambda _symbol: True)
    monkeypatch.setattr(fetch, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(fetch.runtime_state, "update_data_provider_state", lambda **_kwargs: None)
    monkeypatch.setattr(fetch, "log_backup_provider_used", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "log_fetch_attempt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "log_empty_retries_exhausted", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "warn_finnhub_disabled_no_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "log_finnhub_disabled", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "inc_provider_fallback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "record_unauthorized_sip_event", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(fetch, "_repair_rth_minute_gaps", lambda df, **_kwargs: (df, {"expected": len(df), "missing_after": 0, "gap_ratio": 0.0}, False))
    monkeypatch.setattr(fetch, "_verify_minute_continuity", lambda df, *_args, **_kwargs: df)
    monkeypatch.setattr(fetch, "_resolve_gap_ratio_limit", lambda *args, **kwargs: 0.30)
    monkeypatch.setattr(fetch, "_ALPACA_CONSECUTIVE_FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(fetch, "_alpaca_disabled_until", None, raising=False)
    monkeypatch.setattr(fetch, "_ALPACA_DISABLED_ALERTED", False, raising=False)
    monkeypatch.setattr(fetch, "_alpaca_empty_streak", 0, raising=False)
    monkeypatch.setattr(fetch, "_alpaca_disable_count", 0, raising=False)
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED_UNTIL", None, raising=False)
    return env


def test_fetch_bars_invalid_feed_uses_backup_and_records_fallback(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_Resp(400, text="invalid feed requested", headers={"Content-Type": "text/plain"}))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: _frame(provider="yahoo"))

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert not result.empty
    assert result.attrs["data_provider"] == "yahoo"
    assert fetch_env["BACKUP_DATA_PROVIDER"] == "yahoo"
    assert fetch._FALLBACK_WINDOWS


def test_fetch_bars_rate_limit_with_retry_after_returns_backup_and_sets_cooldown(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = fetch.provider_monitor
    session = _Session(_Resp(429, {"message": "slow down"}, headers={"Retry-After": "7"}))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_backup_get_bars", lambda *_args, **_kwargs: _frame(provider="yahoo"))

    result = fetch._fetch_bars("MSFT", BASE_START, BASE_END, "1Min", feed="iex")

    assert not result.empty
    assert monitor.failures[0][1] == "rate_limited"
    assert ("MSFT", "1Min") in fetch._BACKUP_SKIP_UNTIL
    assert any(call[0] == "data.fetch.rate_limited" for call in fetch.metrics.incr_calls)


def test_fetch_bars_timeout_retries_and_recovers_on_primary(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(fetch.Timeout("slow"), _Resp(200, {"bars": [_bar_payload(close=111.0)]}))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_session_get", _direct_session_get)
    monkeypatch.setattr(fetch, "_sip_fallback_allowed", lambda *_args, **_kwargs: False)

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert list(result["close"]) == [111.0]
    assert len(session.calls) == 2
    assert any(call[0] == "data.fetch.timeout" for call in fetch.metrics.incr_calls)


def test_fetch_bars_connection_error_exhaustion_logs_and_raises(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(fetch.ConnectionError("down"))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_session_get", _direct_session_get)
    monkeypatch.setattr(fetch, "_fetch_bars_max_retries", lambda: 1)
    monkeypatch.setattr(fetch, "_select_fallback_target", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(fetch, "_sip_fallback_allowed", lambda *_args, **_kwargs: False)

    with pytest.raises(fetch.ConnectionError):
        fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert fetch.provider_monitor.failures[0][1] == "connection_error"


def test_fetch_bars_request_exception_500_retries_and_recovers(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    error = fetch.RequestException("server exploded")
    error.response = SimpleNamespace(status_code=503)  # type: ignore[attr-defined]
    session = _Session(error, _Resp(200, {"bars": [_bar_payload(close=112.0)]}))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_session_get", _direct_session_get)
    monkeypatch.setattr(fetch, "_sip_fallback_allowed", lambda *_args, **_kwargs: False)

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert list(result["close"]) == [112.0]
    assert fetch.provider_monitor.failures[0][1] == "http_5xx"
    assert any(call[0] == "data.fetch.error" for call in fetch.metrics.incr_calls)


def test_fetch_bars_explicit_sip_unauthorized_returns_empty_and_marks_lockout(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_Resp(403, {"message": "sip subscription not permitted"}, text="SIP subscription not permitted", headers={"Retry-After": "11"}))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_backup_get_bars", lambda *_args, **_kwargs: pd.DataFrame())

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="sip")

    assert result is None or result.empty
    assert fetch._SIP_UNAUTHORIZED is True
    assert fetch.provider_monitor.failures[0][0] == "alpaca_sip"


def test_fetch_bars_no_content_returns_normalized_empty_frame(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        _Resp(204, None, text="", headers={"Content-Type": "text/plain"}),
        _Resp(204, None, text="", headers={"Content-Type": "text/plain"}),
        _Resp(204, None, text="", headers={"Content-Type": "text/plain"}),
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_fetch_bars_max_retries", lambda: 1)

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert result is None or result.empty
    if result is not None:
        assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_bars_empty_iex_switches_to_sip_success(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_Resp(200, {"bars": [], "error": "empty"}), _Resp(200, {"bars": [_bar_payload(close=120.0)]}))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert list(result["close"]) == [120.0]
    assert [call["params"]["feed"] for call in session.calls] == ["iex", "sip"]
    assert fetch._get_cached_or_primary("AAPL", "iex") == "sip"


def test_fetch_bars_empty_iex_and_empty_sip_uses_http_backup(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        _Resp(200, {"bars": [], "error": "empty"}),
        _Resp(200, {"bars": [], "error": "empty"}),
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: _frame(provider="yahoo"))
    monkeypatch.setattr(fetch, "_fetch_bars_max_retries", lambda: 1)

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert not result.empty
    assert result.attrs["data_provider"] == "yahoo"
    assert fetch.metrics.empty_payload >= 1


def test_fetch_bars_first_empty_schedules_retry_then_recovers(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_Resp(200, {"bars": []}), _Resp(200, {"bars": [_bar_payload(close=130.0)]}))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)
    monkeypatch.setattr(fetch, "_sip_allowed", lambda: False)
    monkeypatch.setattr(fetch, "_sip_configured", lambda: False)
    monkeypatch.setattr(fetch, "_sip_fallback_allowed", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(fetch, "_should_use_backup_on_empty", lambda: False)
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 0)
    monkeypatch.setattr(fetch, "_http_fallback_permitted", lambda *_args, **_kwargs: False)

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert list(result["close"]) == [130.0]
    assert len(session.calls) == 2


def test_fetch_bars_provider_disabled_gate_routes_to_backup(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = fetch.provider_monitor
    monitor.disable("alpaca", duration=30)
    monkeypatch.setattr(fetch, "_alpaca_disabled_until", datetime.now(tz=UTC) + timedelta(seconds=30), raising=False)
    monkeypatch.setattr(fetch, "_HTTP_SESSION", _Session())
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: _frame(provider="yahoo"))

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert not result.empty
    assert monitor.alert_manager.calls
    assert ("AAPL", "1Min") in fetch._BACKUP_SKIP_UNTIL


def test_fetch_bars_skip_window_bypasses_primary_until_probe(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    future = datetime.now(tz=UTC) + timedelta(minutes=3)
    fetch._BACKUP_SKIP_UNTIL[("AAPL", "1Min")] = future
    fetch._SKIPPED_SYMBOLS.add(("AAPL", "1Min"))
    monkeypatch.setattr(fetch, "_HTTP_SESSION", _Session())
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: _frame(provider="yahoo"))

    result = fetch._fetch_bars("AAPL", BASE_START, BASE_END, "1Min", feed="iex")

    assert not result.empty
    assert not fetch._get_fetch_state().get("last_fetch_attempt")


def test_get_minute_df_reference_role_delegates_to_reference_fetch(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_reference(symbol: str, _start: Any, _end: Any, timeframe: str, **kwargs: Any) -> pd.DataFrame:
        calls.append((timeframe, kwargs["feed"]))
        return _frame(provider="alpaca_reference")

    monkeypatch.setattr(fetch, "_fetch_reference_bars", fake_reference)

    result = fetch.get_minute_df("AAPL", BASE_START, BASE_END, feed="delayed_sip")

    assert not result.empty
    assert calls == [("1Min", "delayed_sip")]


def test_get_minute_df_forced_yahoo_provider_uses_backup_and_records_metrics(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_env_source_override", lambda _tf: ("yahoo", "DATA_SOURCE"))
    monkeypatch.setattr(fetch, "_yahoo_get_bars", lambda *_args, **_kwargs: _frame(ts=PAST_START, provider="yahoo"))
    monkeypatch.setattr(fetch, "_fetch_bars", lambda *_args, **_kwargs: pytest.fail("primary should be skipped"))

    result = fetch.get_minute_df("AAPL", PAST_START, PAST_END)

    assert not result.empty
    assert result.attrs["data_provider"] == "yahoo"
    assert ("AAPL", "1Min") in fetch._BACKUP_SKIP_UNTIL
    assert any(call[0] == "data.fetch.fallback_success" for call in fetch.metrics.incr_calls)


def test_get_minute_df_provider_monitor_backup_choice_skips_primary(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = fetch.provider_monitor
    monitor.active_choice = "yahoo"
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: _frame(ts=PAST_START, provider="yahoo"))
    monkeypatch.setattr(fetch, "_fetch_bars", lambda *_args, **_kwargs: pytest.fail("monitor selected backup"))

    result = fetch.get_minute_df("AAPL", PAST_START, PAST_END, feed="iex")

    assert not result.empty
    assert monitor.health_updates
    assert ("AAPL", "1Min") in fetch._SKIPPED_SYMBOLS


def test_get_minute_df_backup_skip_probe_forces_primary_with_bypass(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    fetch._BACKUP_SKIP_UNTIL[("AAPL", "1Min")] = datetime.now(tz=UTC) + timedelta(minutes=5)
    fetch._SKIPPED_SYMBOLS.add(("AAPL", "1Min"))
    monkeypatch.setattr(fetch, "_backup_primary_probe_due", lambda *_args: True)
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: _frame(ts=PAST_START, provider="yahoo"))

    def fake_fetch_bars(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append({"args": args, "kwargs": kwargs})
        return _frame(ts=PAST_START, provider="alpaca")

    monkeypatch.setattr(fetch, "_fetch_bars", fake_fetch_bars)

    result = fetch.get_minute_df("AAPL", PAST_START, PAST_END, feed="iex")

    assert not result.empty
    assert calls and calls[0]["kwargs"]["bypass_backup_skip"] is True


def test_fetch_bars_recovery_probe_bypasses_fallback_ttl_and_exact_window(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    tf_key = ("AAPL", "1Min")
    fetch._BACKUP_SKIP_UNTIL[tf_key] = datetime.now(tz=UTC) + timedelta(minutes=5)
    fetch._SKIPPED_SYMBOLS.add(tf_key)
    fetch._FALLBACK_UNTIL[tf_key] = int((datetime.now(tz=UTC) + timedelta(minutes=5)).timestamp())
    fetch._FALLBACK_WINDOWS.add(fetch._fallback_key("AAPL", "1Min", BASE_START, BASE_END))
    monkeypatch.setattr(fetch, "_session_get", _direct_session_get)
    monkeypatch.setattr(
        fetch,
        "_HTTP_SESSION",
        _Session(_Resp(200, {"bars": [_bar_payload(close=109.0)]})),
    )
    monkeypatch.setattr(
        fetch,
        "_safe_backup_get_bars",
        lambda *_args, **_kwargs: pytest.fail("recovery probe should not route to backup"),
    )
    monkeypatch.setattr(
        fetch,
        "_backup_get_bars",
        lambda *_args, **_kwargs: pytest.fail("recovery probe should not route to backup"),
    )

    result = fetch._fetch_bars(
        "AAPL",
        BASE_START,
        BASE_END,
        "1Min",
        feed="iex",
        bypass_backup_skip=True,
    )

    assert not result.empty
    assert list(result["close"]) == [109.0]


def test_get_minute_df_primary_success_does_not_schedule_backup_skip(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_fetch_bars", lambda *_args, **_kwargs: _frame(ts=PAST_START, provider="alpaca"))

    result = fetch.get_minute_df("AAPL", PAST_START, PAST_END, feed="iex")

    assert not result.empty
    assert ("AAPL", "1Min") not in fetch._BACKUP_SKIP_UNTIL
    assert ("AAPL", "1Min") not in fetch._SKIPPED_SYMBOLS


def test_get_minute_df_empty_primary_switches_to_alt_feed_success(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_fetch_bars(_symbol: str, _start: Any, _end: Any, _tf: str, *, feed: str = "iex", **_kwargs: Any) -> pd.DataFrame:
        calls.append(feed)
        if feed == "iex":
            raise fetch.EmptyBarsError("empty")
        return _frame(provider="alpaca")

    monkeypatch.setattr(fetch, "_fetch_bars", fake_fetch_bars)

    result = fetch.get_minute_df("AAPL", BASE_START, BASE_END, feed="iex")

    assert not result.empty
    assert calls == ["iex", "sip"]
    assert fetch._get_cached_or_primary("AAPL", "iex") == "sip"


def test_get_minute_df_empty_threshold_uses_yahoo_backup_and_marks_skip(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    fetch._EMPTY_BAR_COUNTS[("AAPL", "1Min")] = 3
    monkeypatch.setattr(fetch, "_fetch_bars", lambda *_args, **_kwargs: (_ for _ in ()).throw(fetch.EmptyBarsError("empty")))
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: _frame(ts=PAST_START, provider="yahoo"))
    monkeypatch.setattr(fetch, "_sip_configured", lambda: False)
    monkeypatch.setattr(fetch, "_sip_allowed", lambda: False)

    result = fetch.get_minute_df("AAPL", PAST_START, PAST_END, feed="iex")

    assert not result.empty
    assert result.attrs["data_provider"] == "yahoo"
    assert ("AAPL", "1Min") in fetch._SKIPPED_SYMBOLS


def test_mark_fallback_records_metadata_cooldowns_runtime_state_and_cycle_cache(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    updates: list[dict[str, Any]] = []
    monkeypatch.setattr(fetch.runtime_state, "update_data_provider_state", lambda **kwargs: updates.append(kwargs))
    frame = _frame(provider="yahoo")
    frame.attrs["fallback_reason"] = "gap_ratio_exceeded"
    fetch._set_fetch_state({"coverage_meta": {"gap_ratio": 0.5, "gap_ratio_pct": 50.0, "gap_over_limit": True}})

    fetch._mark_fallback(
        "AAPL",
        "1Min",
        BASE_START,
        BASE_END,
        from_provider="alpaca_iex",
        fallback_df=frame,
        resolved_provider="yahoo",
        resolved_feed="yahoo",
        reason="gap_ratio_exceeded",
    )

    metadata = fetch.get_fallback_metadata("AAPL", "1Min", BASE_START, BASE_END)
    assert metadata is not None
    assert metadata["fallback_provider"] == "yahoo"
    assert metadata["fallback_reason"] == "gap_ratio_exceeded"
    assert fetch._fallback_cache_for_cycle(fetch._get_cycle_id(), "AAPL", "1Min") == "yahoo"
    assert fetch._FALLBACK_UNTIL[("AAPL", "1Min")] > 0
    assert updates[-1]["using_backup"] is True


def test_set_backup_skip_handles_existing_global_and_bad_until(fetch_env: dict[str, Any]) -> None:
    future = datetime.now(tz=UTC) + timedelta(minutes=10)
    fetch._GLOBAL_BACKUP_SKIP_UNTIL["1Min"] = future

    fetch._set_backup_skip("AAPL", "1Min", until=datetime.now(tz=UTC) + timedelta(minutes=20))
    assert fetch._BACKUP_SKIP_UNTIL[("AAPL", "1Min")] == future
    assert ("AAPL", "1Min") in fetch._BACKUP_PRIMARY_PROBE_AT

    fetch._set_backup_skip("MSFT", "1Min", until=object())
    assert ("MSFT", "1Min") in fetch._SKIPPED_SYMBOLS
    assert ("MSFT", "1Min") not in fetch._BACKUP_SKIP_UNTIL


def test_sip_fallback_allowed_precheck_marks_unauthorized(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_Resp(403, {"message": "forbidden"}))
    monkeypatch.setattr(fetch, "_SIP_PRECHECK_DONE", False, raising=False)
    monkeypatch.setattr(fetch, "_detect_pytest_env", lambda: False)
    monkeypatch.setattr(fetch, "_sip_allowed", lambda: True)
    monkeypatch.setattr(fetch, "_sip_configured", lambda: True)
    monkeypatch.setattr(fetch, "_intraday_feed_prefers_sip", lambda: True)

    assert fetch._sip_fallback_allowed(session, {}, "1Min") is False
    assert fetch._SIP_UNAUTHORIZED is True
    assert fetch.provider_monitor.failures[0][0] == "alpaca"


def test_ensure_ohlcv_schema_recovers_index_aliases_and_reports_payload_metadata(fetch_env: dict[str, Any]) -> None:
    indexed = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5]},
        index=pd.DatetimeIndex([BASE_START], name="Timestamp"),
    )
    normalized = fetch.ensure_ohlcv_schema(indexed, source="alpaca_iex", frequency="1Min")
    assert list(normalized.columns[:6]) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert normalized.index.tz is not None

    bad = pd.DataFrame({"timestamp": [BASE_START], "close": [1.0]})
    bad.attrs["raw_payload_columns"] = ["timestamp", "close"]
    bad.attrs["raw_payload_provider"] = "alpaca"
    with pytest.raises(fetch.MissingOHLCVColumnsError) as excinfo:
        fetch.ensure_ohlcv_schema(bad, source="alpaca_iex", frequency="1Min")

    assert excinfo.value.raw_payload_columns == ("timestamp", "close")
    assert excinfo.value.raw_payload_provider == "alpaca"


def test_reload_host_limit_releases_reserved_and_records_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"AI_TRADING_HTTP_HOST_LIMIT": "4"}
    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: values.get(key, default))
    fetch._HOST_LIMITS.clear()
    fetch._HOST_COUNTS.clear()
    fetch._HOST_LIMIT_ENV = None
    sem = fetch.threading.Semaphore(0)
    fetch._HOST_LIMITS["data.alpaca.test"] = sem
    fetch._HOST_COUNTS["data.alpaca.test"] = {"limit": 2, "pending": 1, "reserved": 1, "current": 0, "peak": 0}

    assert fetch.reload_host_limit_if_env_changed() == ("4", 4)
    assert fetch._HOST_COUNTS["data.alpaca.test"]["reserved"] == 0
    assert fetch._HOST_COUNTS["data.alpaca.test"]["pending"] == 0

    values["AI_TRADING_HTTP_HOST_LIMIT"] = "1"
    assert fetch.reload_host_limit_if_env_changed() == ("1", 1)
    meta = fetch._HOST_COUNTS["data.alpaca.test"]
    assert meta["limit"] == 1
    assert meta["reserved"] >= 0
    assert meta["pending"] >= 0


def test_has_alpaca_keys_uses_downgrade_cache_live_and_config_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch._ALPACA_CREDS_CACHE = None
    monkeypatch.setattr(fetch, "_pytest_active", lambda: False)
    monkeypatch.setattr(fetch, "monotonic_time", lambda: 100.0)
    monkeypatch.setattr(fetch, "is_data_feed_downgraded", lambda: True)
    assert fetch._has_alpaca_keys() is False

    monkeypatch.setattr(fetch, "is_data_feed_downgraded", lambda: False)
    fetch._ALPACA_CREDS_CACHE = (True, 90.0)
    assert fetch._has_alpaca_keys() is True

    fetch._ALPACA_CREDS_CACHE = None
    monkeypatch.setattr(fetch, "alpaca_credential_status", lambda: (True, True))
    assert fetch._has_alpaca_keys() is True


def test_yahoo_get_bars_chunks_long_minute_ranges_and_returns_empty_for_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(days=15)
    calls: list[tuple[datetime, datetime, str]] = []

    def fake_batched(symbols: list[str], *, start: datetime, end: datetime, interval: str, **_kwargs: Any) -> dict[str, pd.DataFrame]:
        calls.append((start, end, interval))
        ts = pd.DatetimeIndex([start + timedelta(minutes=1)], tz="UTC", name="timestamp")
        return {symbols[0]: pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10]}, index=ts)}

    monkeypatch.setattr(fetch, "_last_complete_minute", lambda _pd=None: end)
    monkeypatch.setattr(fetch, "fetch_yf_batched", fake_batched)

    result = fetch._yahoo_get_bars("BRK.B", start, end, "1m")
    assert len(calls) == 3
    assert not result.empty
    assert "timestamp" in result.columns

    monkeypatch.setattr(fetch, "fetch_yf_batched", lambda symbols, **_kwargs: {symbols[0]: pd.DataFrame()})
    empty = fetch._yahoo_get_bars("AAPL", start, start + timedelta(minutes=1), "1d")
    assert empty.empty
    with pytest.raises(ValueError):
        fetch._yahoo_get_bars("AAPL", None, end, "1d")
    with pytest.raises(ValueError):
        fetch._yahoo_get_bars("AAPL", start, None, "1d")


def test_flatten_and_normalize_ohlcv_recovers_aliases_and_rejects_all_nan_close() -> None:
    idx = pd.DatetimeIndex([BASE_START, BASE_START, BASE_START + timedelta(minutes=1)], tz="UTC", name="timestamp")
    raw = pd.DataFrame(
        {
            "regularMarketOpen": ["100", "101", "102"],
            "regularMarketDayHigh": ["103", "104", "105"],
            "regularMarketLow": ["99", "100", "101"],
            "regularMarketPrice": ["101", "102", "103"],
            "regularMarketVolume": ["1000", "1100", "1200"],
            "close_duplicate": [1, 2, 3],
        },
        index=idx,
    )

    normalized = fetch._flatten_and_normalize_ohlcv(raw, "AAPL", "1Min")
    assert list(normalized["close"]) == [102, 103]
    assert normalized["timestamp"].dt.tz is not None

    fallback_close = pd.DataFrame(
        {
            "timestamp": [BASE_START],
            "open": [87.0],
            "high": [89.0],
            "low": [86.0],
            "close": [None],
            "avg_price": [88.0],
            "volume": [5],
        },
    )
    recovered = fetch._flatten_and_normalize_ohlcv(fallback_close, "MSFT", "1Min")
    assert list(recovered["close"]) == [88.0]

    missing = pd.DataFrame({"timestamp": [BASE_START], "close": [None], "volume": [1]})
    with pytest.raises(fetch.DataFetchError):
        fetch._flatten_and_normalize_ohlcv(missing, "BAD", "1Min")


def test_repair_rth_minute_gaps_uses_backup_and_records_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=3)
    primary = _frame(start, provider="alpaca")
    backup = pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=1), start + timedelta(minutes=2)],
            "open": [101.0, 102.0],
            "high": [102.0, 103.0],
            "low": [100.0, 101.0],
            "close": [101.5, 102.5],
            "volume": [100, 100],
        },
    )
    backup.attrs["data_provider"] = "yahoo"
    monkeypatch.setattr(fetch, "_resolve_gap_ratio_limit", lambda **_kwargs: 0.0)
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: backup)
    monkeypatch.setattr(fetch, "record_gap_statistics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "record_minute_gap_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetch, "log_throttled_event", lambda *_args, **_kwargs: None)

    repaired, meta, used_backup = fetch._repair_rth_minute_gaps(
        primary,
        symbol="AAPL",
        start=start,
        end=end,
        tz=fetch.ZoneInfo("America/New_York"),
    )

    assert isinstance(used_backup, bool)
    assert meta["expected"] >= 1
    if used_backup:
        assert meta["used_backup"] is True
        assert meta["missing_after"] == 0
    assert not repaired.empty


def test_should_skip_symbol_sets_coverage_metadata_for_catastrophic_gap() -> None:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=3)
    frame = _frame(start - timedelta(days=1))
    frame.attrs["symbol"] = "AAPL"

    assert fetch.should_skip_symbol(
        frame,
        window=(start, end),
        tz="America/New_York",
        max_gap_ratio=0.0,
    ) is True
    assert frame.attrs["_coverage_meta"]["skip_flagged"] is True


def test_get_daily_df_forced_yahoo_and_fresh_memo(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    daily = _frame(PAST_START, provider="yahoo")
    monkeypatch.setattr(fetch, "_env_source_override", lambda tf: ("yahoo", "DATA_SOURCE") if tf == "1Day" else None)
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: daily)
    monkeypatch.setattr(fetch, "should_import_alpaca_sdk", lambda: True)
    result = fetch.get_daily_df("AAPL", PAST_START, PAST_END)
    assert not result.empty
    assert result.attrs["data_provider"] == "yahoo"

    memo_frame = _frame(PAST_START, provider="memo")
    monkeypatch.setattr(fetch, "_env_source_override", lambda _tf: None)
    monkeypatch.setattr(fetch, "_is_fresh", lambda _ts: True)
    memo = {"df": memo_frame, "ts": datetime.now(tz=UTC)}
    memo_result = fetch.get_daily_df("AAPL", BASE_START, BASE_END, memo=memo)
    assert memo_result is not memo_frame
    assert list(memo_result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert memo_result.attrs["data_provider"] == "memo"


def test_get_daily_df_rejects_future_memo_frame(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    future_memo = _frame(datetime.now(tz=UTC) + timedelta(minutes=1), provider="memo")
    backup = _frame(BASE_START, provider="yahoo")

    monkeypatch.setattr(fetch, "_env_source_override", lambda _tf: None)
    monkeypatch.setattr(fetch, "should_import_alpaca_sdk", lambda: False)
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: backup)

    result = fetch.get_daily_df(
        "AAPL",
        BASE_START,
        BASE_END,
        memo={"df": future_memo, "ts": datetime.now(tz=UTC)},
    )

    assert result is not future_memo
    assert result.attrs["data_provider"] == "yahoo"


def test_get_daily_df_drops_future_provider_rows(
    fetch_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    future_ts = datetime.now(tz=UTC) + timedelta(days=1)
    provider_frame = pd.DataFrame(
        {
            "timestamp": [PAST_START, future_ts],
            "open": [100.0, 200.0],
            "high": [102.0, 202.0],
            "low": [99.0, 199.0],
            "close": [101.0, 201.0],
            "volume": [1000, 2000],
        }
    )
    provider_frame.attrs["data_provider"] = "finnhub"

    monkeypatch.setattr(fetch, "_env_source_override", lambda tf: ("finnhub",) if tf == "1Day" else None)
    monkeypatch.setattr(fetch, "_finnhub_get_bars", lambda *_args, **_kwargs: provider_frame)
    caplog.set_level(logging.WARNING, logger="ai_trading.data.fetch")

    result = fetch.get_daily_df("AAPL", PAST_START, PAST_END)

    assert list(result["timestamp"]) == [pd.Timestamp(PAST_START)]
    assert list(result["close"]) == [101.0]
    assert any(record.message == "future_bar_timestamp" for record in caplog.records)


def test_get_daily_df_rejects_malformed_memo_frame(fetch_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    malformed_memo = pd.DataFrame(
        {
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1000],
        },
    )
    backup = _frame(BASE_START, provider="yahoo")

    monkeypatch.setattr(fetch, "_env_source_override", lambda _tf: None)
    monkeypatch.setattr(fetch, "should_import_alpaca_sdk", lambda: False)
    monkeypatch.setattr(fetch, "_safe_backup_get_bars", lambda *_args, **_kwargs: backup)
    monkeypatch.setattr(fetch, "_is_fresh", lambda _ts: True)

    result = fetch.get_daily_df(
        "AAPL",
        BASE_START,
        BASE_END,
        memo={"df": malformed_memo, "ts": datetime.now(tz=UTC)},
    )

    assert result is not malformed_memo
    assert result.attrs["data_provider"] == "yahoo"


def test_post_process_detects_all_nan_close_and_empty_slot_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_fallback_slots_remaining", lambda: 0)
    monkeypatch.setattr(fetch, "_http_fallback_permitted", lambda *_args, **_kwargs: False)
    fetch._set_fetch_state({"window_has_session": True, "initial_feed": "iex"})
    with pytest.raises(fetch.EmptyBarsError):
        fetch._post_process(pd.DataFrame(), symbol="AAPL", timeframe="1Min")

    bad = pd.DataFrame(
        {"timestamp": [BASE_START], "open": [1.0], "high": [1.0], "low": [1.0], "close": [None], "volume": [1]},
    )
    monkeypatch.setattr(fetch, "_flatten_and_normalize_ohlcv", lambda *_args, **_kwargs: bad)
    with pytest.raises(fetch.DataFetchError):
        fetch._post_process(bad, symbol="AAPL", timeframe="1Min")
