from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ai_trading.config.runtime import TradingConfig
from ai_trading.core import bot_engine
from ai_trading.policy.compiler import SafetyTier


def test_active_effective_policy_caches_until_config_changes(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_POLICY_STRICT_CONFIG_GOVERNANCE", "0")
    cfg = TradingConfig.from_env(allow_missing_drawdown=True)
    state = bot_engine.BotState()

    calls = {"count": 0}
    real_compile = bot_engine.compile_effective_policy

    def _compile_counted(cfg_obj, env=None):
        calls["count"] += 1
        return real_compile(cfg_obj, env)

    monkeypatch.setattr(bot_engine, "compile_effective_policy", _compile_counted)
    monkeypatch.setattr(bot_engine, "startup_policy_diff", lambda *args, **kwargs: [])

    first = bot_engine._active_effective_policy(state, cfg)
    second = bot_engine._active_effective_policy(state, cfg)

    assert calls["count"] == 1
    assert first.policy_hash == second.policy_hash

    updated_cap = 0.07 if abs(float(cfg.capital_cap) - 0.07) > 1e-9 else 0.08
    cfg.update(capital_cap=updated_cap)
    bot_engine._active_effective_policy(state, cfg)

    assert calls["count"] == 2


def test_operational_safety_hysteresis_applies_confirm_and_dwell(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_SAFETY_TIER_MIN_DWELL_SEC", "120")
    monkeypatch.setenv("AI_TRADING_SAFETY_TIER_SAFE_EXIT_CONFIRM_CYCLES", "2")
    monkeypatch.setenv("AI_TRADING_SAFETY_TIER_NORMAL_CONFIRM_CYCLES", "1")

    now = datetime.now(UTC)
    state = bot_engine.BotState()
    state.operational_tier_last_change_ts = now

    held_tier, held_reasons = bot_engine._apply_operational_safety_hysteresis(
        state=state,
        previous_tier=SafetyTier.SAFE,
        candidate_tier=SafetyTier.NORMAL,
        candidate_reasons=("NORMAL_TIER",),
        now=now + timedelta(seconds=30),
    )
    assert held_tier is SafetyTier.SAFE
    assert "SAFETY_TIER_HYSTERESIS_HOLD" in held_reasons

    promoted_tier, _ = bot_engine._apply_operational_safety_hysteresis(
        state=state,
        previous_tier=SafetyTier.SAFE,
        candidate_tier=SafetyTier.NORMAL,
        candidate_reasons=("NORMAL_TIER",),
        now=now + timedelta(seconds=130),
    )
    assert promoted_tier is SafetyTier.NORMAL


def test_startup_healthcheck_logs_backup_usage_alert(monkeypatch, caplog) -> None:
    monkeypatch.setenv("AI_TRADING_STARTUP_PREFETCH_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_STARTUP_BACKUP_ALERT_THRESHOLD", "1")
    monkeypatch.setattr(bot_engine, "data_source_health_check", lambda ctx, symbols: None)
    bot_engine.REGIME_SYMBOLS = ["AAPL"]

    totals = iter((0, 2))
    monkeypatch.setattr(
        bot_engine,
        "backup_provider_used_total",
        lambda provider=None: next(totals),
    )

    ctx = SimpleNamespace(data_fetcher=SimpleNamespace(_daily_cache={}))
    bot_engine._HEALTH_CHECK_FAILURES = 0

    with caplog.at_level(logging.WARNING):
        bot_engine._initialize_bot_context_post_setup(ctx)

    assert any(
        record.getMessage() == "STARTUP_BACKUP_PROVIDER_USAGE_HIGH"
        for record in caplog.records
    )


def test_startup_healthcheck_does_not_alert_on_single_fallback_by_default(monkeypatch, caplog) -> None:
    monkeypatch.setenv("AI_TRADING_STARTUP_PREFETCH_ENABLED", "0")
    monkeypatch.delenv("AI_TRADING_STARTUP_BACKUP_ALERT_THRESHOLD", raising=False)
    monkeypatch.setattr(bot_engine, "data_source_health_check", lambda ctx, symbols: None)
    bot_engine.REGIME_SYMBOLS = ["AAPL"]

    totals = iter((0, 1))
    monkeypatch.setattr(
        bot_engine,
        "backup_provider_used_total",
        lambda provider=None: next(totals),
    )

    ctx = SimpleNamespace(data_fetcher=SimpleNamespace(_daily_cache={}))
    bot_engine._HEALTH_CHECK_FAILURES = 0

    with caplog.at_level(logging.WARNING):
        bot_engine._initialize_bot_context_post_setup(ctx)

    assert not any(
        record.getMessage() == "STARTUP_BACKUP_PROVIDER_USAGE_HIGH"
        for record in caplog.records
    )


def test_clear_transient_halt_state_unlatches_derisk_block() -> None:
    state = bot_engine.BotState()
    state.halt_trading = True
    state.halt_reason = "DERISK_SLO_BREACH_BLOCK"

    changed = bot_engine._clear_transient_halt_state(state)

    assert changed is True
    assert state.halt_trading is False
    assert state.halt_reason is None


def test_clear_transient_halt_state_keeps_non_transient_halt() -> None:
    state = bot_engine.BotState()
    state.halt_trading = True
    state.halt_reason = "DAILY_RISK_BUDGET_HARD_STOP"

    changed = bot_engine._clear_transient_halt_state(state)

    assert changed is False
    assert state.halt_trading is True
    assert state.halt_reason == "DAILY_RISK_BUDGET_HARD_STOP"


def test_record_netting_model_liveness_emits_ml_and_rl_when_enabled(monkeypatch) -> None:
    calls: dict[str, int] = {"ml": 0, "rl": 0}

    def _note_ml_signal(*, now=None) -> None:
        _ = now
        calls["ml"] += 1

    def _note_rl_signal(*, now=None) -> None:
        _ = now
        calls["rl"] += 1

    monkeypatch.setattr(bot_engine, "note_ml_signal", _note_ml_signal)
    monkeypatch.setattr(bot_engine, "note_rl_signals_emitted", _note_rl_signal)
    monkeypatch.setattr(bot_engine, "RL_AGENT", object())
    monkeypatch.setenv("USE_RL_AGENT", "1")

    bot_engine._record_netting_model_liveness(proposals_total=3)

    assert calls["ml"] == 1
    assert calls["rl"] == 1


def test_record_netting_model_liveness_skips_when_no_proposals(monkeypatch) -> None:
    calls: dict[str, int] = {"ml": 0, "rl": 0}

    def _note_ml_signal(*, now=None) -> None:
        _ = now
        calls["ml"] += 1

    def _note_rl_signal(*, now=None) -> None:
        _ = now
        calls["rl"] += 1

    monkeypatch.setattr(bot_engine, "note_ml_signal", _note_ml_signal)
    monkeypatch.setattr(bot_engine, "note_rl_signals_emitted", _note_rl_signal)
    monkeypatch.setattr(bot_engine, "RL_AGENT", object())
    monkeypatch.setenv("USE_RL_AGENT", "1")

    bot_engine._record_netting_model_liveness(proposals_total=0)

    assert calls["ml"] == 0
    assert calls["rl"] == 0
