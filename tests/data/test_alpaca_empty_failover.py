from __future__ import annotations

from datetime import UTC, datetime, timedelta
import types

import pytest

from ai_trading.data import fetch as data_fetch
from ai_trading.data.fetch import iex_fallback


@pytest.mark.parametrize(
    ("feed", "sip_configured", "sip_unauthorized", "expected"),
    [
        ("iex", False, False, False),
        ("iex", True, False, True),
        ("iex", True, True, False),
        ("sip", False, False, True),
        ("yahoo", False, False, True),
        (None, False, False, True),
    ],
)
def test_should_disable_alpaca_on_empty(feed, sip_configured, sip_unauthorized, expected, monkeypatch):
    monkeypatch.setattr(data_fetch, "_sip_configured", lambda: sip_configured, raising=False)
    monkeypatch.setattr(data_fetch, "_is_sip_unauthorized", lambda: sip_unauthorized, raising=False)
    assert data_fetch._should_disable_alpaca_on_empty(feed) is expected


def test_mark_fallback_skips_switchover_for_iex_without_sip(monkeypatch):
    data_fetch._FALLBACK_WINDOWS.clear()
    monkeypatch.setattr(data_fetch, "_sip_configured", lambda: False, raising=False)
    monkeypatch.setattr(data_fetch, "log_backup_provider_used", lambda *a, **k: {})
    monkeypatch.setattr(data_fetch, "fallback_order", types.SimpleNamespace(register_fallback=lambda *a, **k: None))
    recorded: list[tuple] = []
    monkeypatch.setattr(
        data_fetch,
        "provider_monitor",
        types.SimpleNamespace(record_switchover=lambda *a, **k: recorded.append((a, k))),
        raising=False,
    )

    data_fetch._mark_fallback(
        "AAPL",
        "1Min",
        datetime.now(UTC),
        datetime.now(UTC),
        from_provider="alpaca_iex",
        fallback_feed="yahoo",
        resolved_provider="yahoo",
        resolved_feed="yahoo",
        reason="close_column_all_nan",
    )

    assert recorded == []


def test_mark_fallback_records_switchover_when_sip_available(monkeypatch):
    data_fetch._FALLBACK_WINDOWS.clear()
    monkeypatch.setattr(data_fetch, "_sip_configured", lambda: True, raising=False)
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setattr(data_fetch, "log_backup_provider_used", lambda *a, **k: {})
    monkeypatch.setattr(data_fetch, "fallback_order", types.SimpleNamespace(register_fallback=lambda *a, **k: None))
    recorded: list[tuple] = []
    monkeypatch.setattr(
        data_fetch,
        "provider_monitor",
        types.SimpleNamespace(record_switchover=lambda *a, **k: recorded.append((a, k))),
        raising=False,
    )

    data_fetch._mark_fallback(
        "AAPL",
        "1Min",
        datetime.now(UTC),
        datetime.now(UTC),
        from_provider="alpaca_iex",
        fallback_feed="yahoo",
        resolved_provider="yahoo",
        resolved_feed="yahoo",
        reason="close_column_all_nan",
    )

    assert recorded


def test_yahoo_fallback_requires_consecutive_failures(monkeypatch):
    data_fetch._ALPACA_CONSECUTIVE_FAILURES.clear()
    data_fetch._ALPACA_FAILURE_EVENTS.clear()
    original_monitor = data_fetch.provider_monitor
    stub_monitor = types.SimpleNamespace(
        fail_counts={},
        threshold=5,
        is_disabled=lambda name: False,
    )
    monkeypatch.setattr(data_fetch, "provider_monitor", stub_monitor, raising=False)
    monkeypatch.setattr(data_fetch, "_ALPACA_CONSECUTIVE_FAILURE_THRESHOLD", 3, raising=False)

    assert data_fetch._yahoo_fallback_allowed("AAPL", "1Min") is False
    data_fetch._record_alpaca_failure_event("AAPL", timeframe="1Min")
    assert data_fetch._yahoo_fallback_allowed("AAPL", "1Min") is False
    data_fetch._record_alpaca_failure_event("AAPL", timeframe="1Min")
    assert data_fetch._yahoo_fallback_allowed("AAPL", "1Min") is False
    data_fetch._record_alpaca_failure_event("AAPL", timeframe="1Min")
    assert data_fetch._yahoo_fallback_allowed("AAPL", "1Min") is True
    data_fetch._clear_alpaca_failure_events("AAPL", timeframe="1Min")
    if original_monitor is not None:
        data_fetch.provider_monitor = original_monitor


def test_iex_empty_records_consecutive_failures(monkeypatch):
    data_fetch._ALPACA_CONSECUTIVE_FAILURES.clear()
    data_fetch._ALPACA_FAILURE_EVENTS.clear()
    data_fetch._IEX_EMPTY_COUNTS.clear()
    monkeypatch.setattr(iex_fallback, "get_alpaca_data_base_url", lambda: "https://example.com", raising=False)
    monkeypatch.setattr(iex_fallback, "get_alpaca_http_headers", lambda: {}, raising=False)
    monkeypatch.setattr(iex_fallback, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(iex_fallback, "_SIP_UNAUTHORIZED", False, raising=False)

    class StubResp:
        def __init__(self, bars: list[dict[str, object]] | None = None) -> None:
            self._bars = bars or []

        def json(self) -> dict[str, list[dict[str, object]]]:
            return {"bars": list(self._bars)}

    class StubSession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object] | None]] = []

        def get(self, url: str, params: dict[str, object] | None = None, headers: dict[str, str] | None = None) -> StubResp:
            self.calls.append((url, params))
            return StubResp([])

    session = StubSession()
    start = datetime.now(UTC) - timedelta(minutes=2)
    end = datetime.now(UTC)
    iex_fallback.fetch_bars("AAPL", start, end, "1Min", session=session)
    consecutive = data_fetch._consecutive_failure_count("AAPL", "1Min")
    assert consecutive >= 2
    data_fetch._clear_alpaca_failure_events("AAPL", timeframe="1Min")
