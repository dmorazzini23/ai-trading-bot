from __future__ import annotations

from datetime import UTC, datetime
import types

import pytest

from ai_trading.data import fetch as data_fetch


@pytest.mark.parametrize(
    ("feed", "sip_configured", "expected"),
    [
        ("iex", False, False),
        ("iex", True, True),
        ("sip", False, True),
        ("yahoo", False, True),
        (None, False, True),
    ],
)
def test_should_disable_alpaca_on_empty(feed, sip_configured, expected, monkeypatch):
    monkeypatch.setattr(data_fetch, "_sip_configured", lambda: sip_configured, raising=False)
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
