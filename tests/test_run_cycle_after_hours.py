from __future__ import annotations

import sys
import types
from datetime import datetime
from zoneinfo import ZoneInfo

from ai_trading import main
from ai_trading.alpaca_api import AlpacaAuthenticationError, is_alpaca_service_available


def test_run_cycle_skips_when_market_closed(monkeypatch):
    """run_cycle should short-circuit on closed sessions and broker-sync once."""

    monkeypatch.setattr(main, "_is_market_open_base", lambda: False)
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "0")
    sync_calls = {"count": 0}

    class _StubExecutionEngine:
        def synchronize_broker_state(self):
            sync_calls["count"] += 1
            return types.SimpleNamespace(open_orders=[], positions=[])

    runtime_obj = types.SimpleNamespace(execution_engine=_StubExecutionEngine())
    stub_bot_engine = types.ModuleType("ai_trading.core.bot_engine")

    class _StubBotState:
        pass

    stub_bot_engine.BotState = _StubBotState
    stub_bot_engine.get_ctx = lambda: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", stub_bot_engine)

    stub_runtime = types.ModuleType("ai_trading.core.runtime")
    stub_runtime.build_runtime = lambda _cfg: runtime_obj
    stub_runtime.enhance_runtime_with_context = lambda runtime, _ctx: runtime
    monkeypatch.setitem(sys.modules, "ai_trading.core.runtime", stub_runtime)
    monkeypatch.setattr(
        main,
        "_resolve_cached_context",
        lambda _cfg, state_cls, runtime_builder: (state_cls(), runtime_builder(_cfg), "test-hash"),
    )

    main.run_cycle()

    assert sync_calls["count"] == 1


def test_run_cycle_calls_market_close_helper_when_closed(monkeypatch):
    """run_cycle should call market-close helper before returning on closed sessions."""

    monkeypatch.setattr(main, "_is_market_open_base", lambda: False)
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "0")
    monkeypatch.setenv("EXECUTION_MODE", "sim")
    calls = {"count": 0}
    monkeypatch.setattr(
        main,
        "_maybe_trigger_market_close_training",
        lambda *_args, **_kwargs: calls.__setitem__("count", calls["count"] + 1),
    )

    main.run_cycle()

    assert calls["count"] == 1


def test_market_close_training_triggers_once_per_day(monkeypatch):
    """Market-close training helper should run only once per New York date."""

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "0")
    monkeypatch.setattr(
        main,
        "_LAST_MARKET_CLOSE_TRAINING_DATE",
        None,
        raising=False,
    )
    calls = {"count": 0}
    monkeypatch.setattr(
        main,
        "_invoke_market_close_training",
        lambda: calls.__setitem__("count", calls["count"] + 1),
    )
    now_est = datetime(2026, 1, 6, 16, 5, tzinfo=ZoneInfo("America/New_York"))

    main._maybe_trigger_market_close_training(now_est)
    main._maybe_trigger_market_close_training(now_est)

    assert calls["count"] == 1


def test_market_close_training_triggers_overnight_catchup_once_per_session(monkeypatch):
    """Overnight catch-up should map to the previous business session key."""

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED", "1")
    monkeypatch.setattr(
        main,
        "_LAST_MARKET_CLOSE_TRAINING_DATE",
        None,
        raising=False,
    )
    calls = {"count": 0}
    monkeypatch.setattr(
        main,
        "_invoke_market_close_training",
        lambda: calls.__setitem__("count", calls["count"] + 1),
    )
    now_est = datetime(2026, 1, 7, 0, 10, tzinfo=ZoneInfo("America/New_York"))

    main._maybe_trigger_market_close_training(now_est)
    main._maybe_trigger_market_close_training(now_est)

    assert calls["count"] == 1
    assert main._LAST_MARKET_CLOSE_TRAINING_DATE == "2026-01-06"


def test_market_close_training_skips_overnight_when_catchup_disabled(monkeypatch):
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED", "0")
    monkeypatch.setattr(
        main,
        "_LAST_MARKET_CLOSE_TRAINING_DATE",
        None,
        raising=False,
    )
    calls = {"count": 0}
    monkeypatch.setattr(
        main,
        "_invoke_market_close_training",
        lambda: calls.__setitem__("count", calls["count"] + 1),
    )
    now_est = datetime(2026, 1, 7, 0, 10, tzinfo=ZoneInfo("America/New_York"))

    main._maybe_trigger_market_close_training(now_est)

    assert calls["count"] == 0


def test_run_cycle_aborts_on_alpaca_auth_failure(monkeypatch, caplog):
    sys.modules.pop("ai_trading.core.bot_engine", None)
    monkeypatch.setenv("ALLOW_AFTER_HOURS", "1")
    monkeypatch.setattr(main, "_is_market_open_base", lambda: True)

    import ai_trading.alpaca_api as alpaca_api

    monkeypatch.setattr(alpaca_api, "_ALPACA_SERVICE_AVAILABLE", True)

    def raise_auth(*_a, **_k):
        monkeypatch.setattr(alpaca_api, "_ALPACA_SERVICE_AVAILABLE", False)
        raise AlpacaAuthenticationError("Unauthorized")

    monkeypatch.setattr(alpaca_api, "alpaca_get", raise_auth)

    with caplog.at_level("CRITICAL"):
        main.run_cycle()

    assert "ALPACA_AUTH_PREFLIGHT_FAILED" in caplog.text
    assert "ai_trading.core.bot_engine" not in sys.modules
    assert not is_alpaca_service_available()
