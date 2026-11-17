from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("TRADING__DEGRADED_MAX_CANDIDATES", raising=False)
    yield
    monkeypatch.delenv("TRADING__DEGRADED_MAX_CANDIDATES", raising=False)


def test_primary_feed_derisk_state_triggers_after_contiguous_fallback(monkeypatch):
    runtime = SimpleNamespace(state={})
    clock = {"value": 100.0}
    monkeypatch.setattr(bot_engine.time, "time", lambda: clock["value"])
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_EXIT_ONLY_ON_DEGRADED", "0")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_MODE", "scale")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_AFTER_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_SCALE_MULT", "0.4")
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "using_backup": True,
            "reason": "upstream_unavailable",
            "timeframes": {"1Min": True},
            "status": "degraded",
        },
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_quote_status",
        lambda: {"synthetic": False, "quote_age_ms": 100.0},
    )

    first = bot_engine._resolve_primary_feed_derisk_state(runtime)
    assert first["triggered"] is False
    assert first["duration_s"] == pytest.approx(0.0)

    clock["value"] = 132.0
    second = bot_engine._resolve_primary_feed_derisk_state(runtime)
    assert second["triggered"] is True
    assert second["scale"] == pytest.approx(0.4)
    assert second["duration_s"] == pytest.approx(32.0)


def test_primary_feed_derisk_state_blocks_after_prolonged_degrade(monkeypatch):
    runtime = SimpleNamespace(state={})
    clock = {"value": 100.0}
    monkeypatch.setattr(bot_engine.time, "time", lambda: clock["value"])
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_EXIT_ONLY_ON_DEGRADED", "0")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_MODE", "scale")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_AFTER_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_BLOCK_AFTER_SEC", "45")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_SCALE_MULT", "0.4")
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "using_backup": True,
            "reason": "upstream_unavailable",
            "timeframes": {"1Min": True},
            "status": "degraded",
        },
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_quote_status",
        lambda: {"synthetic": False, "quote_age_ms": 100.0},
    )

    bot_engine._resolve_primary_feed_derisk_state(runtime)
    clock["value"] = 132.0
    scaled = bot_engine._resolve_primary_feed_derisk_state(runtime)
    assert scaled["triggered"] is True
    assert scaled["block"] is False
    assert scaled["scale"] == pytest.approx(0.4)

    clock["value"] = 151.0
    blocked = bot_engine._resolve_primary_feed_derisk_state(runtime)
    assert blocked["triggered"] is True
    assert blocked["prolonged_block"] is True
    assert blocked["block"] is True
    assert blocked["scale"] == pytest.approx(1.0)
    assert "prolonged_degraded_feed" in str(blocked.get("reason") or "")


def test_primary_feed_derisk_state_exit_only_blocks_new_exposure(monkeypatch):
    runtime = SimpleNamespace(state={})
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_EXIT_ONLY_ON_DEGRADED", "1")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_MODE", "scale")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_AFTER_SEC", "300")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_SCALE_MULT", "0.4")
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "using_backup": True,
            "reason": "alpaca_sip_unauthorized",
            "timeframes": {"1Min": True},
            "status": "degraded",
        },
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_quote_status",
        lambda: {"synthetic": False, "quote_age_ms": 100.0},
    )

    state = bot_engine._resolve_primary_feed_derisk_state(runtime)

    assert state["triggered"] is False
    assert state["exit_only"] is True
    assert state["block"] is True
    assert state["scale"] == pytest.approx(1.0)


def test_primary_feed_derisk_state_resets_when_fallback_clears(monkeypatch):
    runtime = SimpleNamespace(state={})
    clock = {"value": 200.0}
    monkeypatch.setattr(bot_engine.time, "time", lambda: clock["value"])
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_EXIT_ONLY_ON_DEGRADED", "0")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_AFTER_SEC", "10")
    provider_state = {"value": {"timeframes": {"1Min": True}, "using_backup": True}}
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: provider_state["value"],
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_quote_status",
        lambda: {"synthetic": False, "quote_age_ms": 10.0},
    )

    bot_engine._resolve_primary_feed_derisk_state(runtime)
    clock["value"] = 220.0
    provider_state["value"] = {"timeframes": {"1Min": False}, "using_backup": False}
    healthy = bot_engine._resolve_primary_feed_derisk_state(runtime)

    assert healthy["triggered"] is False
    assert healthy["duration_s"] == pytest.approx(0.0)
    assert runtime.state.get(bot_engine._PRIMARY_FEED_DERISK_SINCE_TS_KEY) is None


def test_primary_feed_derisk_suppresses_when_backup_is_healthy(monkeypatch):
    runtime = SimpleNamespace(state={})
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_AFTER_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PRIMARY_FEED_DERISK_EXIT_ONLY_ON_DEGRADED", "0")
    monkeypatch.setenv(
        "AI_TRADING_PRIMARY_FEED_DERISK_SUPPRESS_ON_HEALTHY_BACKUP",
        "1",
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_data_provider_state",
        lambda: {
            "using_backup": True,
            "reason": "data_available_via_backup",
            "timeframes": {"1Min": True},
            "status": "degraded",
            "data_status": "ready",
        },
    )
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "observe_quote_status",
        lambda: {"synthetic": False, "quote_age_ms": 100.0},
    )

    state = bot_engine._resolve_primary_feed_derisk_state(runtime)

    assert state["fallback_active"] is True
    assert state["backup_healthy"] is True
    assert state["fallback_derisk_suppressed"] is True
    assert state["triggered"] is False
    assert state["block"] is False
    assert state["duration_s"] == pytest.approx(0.0)
