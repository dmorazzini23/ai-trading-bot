from typing import Any

import pytest

import ai_trading.core.bot_engine as bot_engine
from ai_trading.config.runtime import TradingConfig
from ai_trading.config.management import reload_trading_config


def _clear_cache(func: Any) -> None:
    cache_clear = getattr(func, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


@pytest.fixture(autouse=True)
def clear_gating_caches():
    _clear_cache(bot_engine._strict_data_gating_enabled)
    _clear_cache(bot_engine._gap_ratio_gate_limit)
    _clear_cache(bot_engine._fallback_gap_ratio_limit)
    _clear_cache(bot_engine._fallback_quote_max_age_seconds)
    _clear_cache(bot_engine._liquidity_fallback_cap)
    _clear_cache(bot_engine._degraded_gap_limit_ratio)
    yield
    _clear_cache(bot_engine._strict_data_gating_enabled)
    _clear_cache(bot_engine._gap_ratio_gate_limit)
    _clear_cache(bot_engine._fallback_gap_ratio_limit)
    _clear_cache(bot_engine._fallback_quote_max_age_seconds)
    _clear_cache(bot_engine._liquidity_fallback_cap)
    _clear_cache(bot_engine._degraded_gap_limit_ratio)


def test_fallback_gap_ratio_prefers_trading_config(monkeypatch):
    # Ensure legacy environment values are present but should be ignored.
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "20")
    monkeypatch.setenv("AI_TRADING_FALLBACK_GAP_LIMIT_BPS", "30")
    cfg = TradingConfig(
        gap_ratio_limit=0.01,
        gap_limit_bps=120,
        fallback_gap_limit_bps=900,
    )
    monkeypatch.setattr(bot_engine, "get_trading_config", lambda: cfg)
    _clear_cache(bot_engine._fallback_gap_ratio_limit)

    result = bot_engine._fallback_gap_ratio_limit()

    assert result == pytest.approx(0.09)


def test_gap_ratio_limit_relaxes_when_failsoft(monkeypatch):
    monkeypatch.setenv("TRADING__SAFE_MODE_FAILSOFT", "true")
    monkeypatch.setenv("TRADING__DEGRADED_GAP_LIMIT_BPS", "2500")
    reload_trading_config()
    monkeypatch.setattr(bot_engine.provider_monitor, "safe_mode_degraded_only", lambda: True)
    monkeypatch.setattr(bot_engine.provider_monitor, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(bot_engine.runtime_state, "observe_data_provider_state", lambda: {"using_backup": True})
    _clear_cache(bot_engine._gap_ratio_gate_limit)

    result = bot_engine._gap_ratio_gate_limit()

    assert result == pytest.approx(0.25)


def test_strict_gating_env_toggle_runtime(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "0")
    assert not bot_engine._strict_data_gating_enabled()
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    assert bot_engine._strict_data_gating_enabled()


def test_fallback_quote_age_env_toggle_runtime(monkeypatch):
    monkeypatch.setenv("AI_TRADING_FALLBACK_QUOTE_MAX_AGE_SEC", "4.5")
    assert bot_engine._fallback_quote_max_age_seconds() == pytest.approx(4.5)
    monkeypatch.setenv("AI_TRADING_FALLBACK_QUOTE_MAX_AGE_SEC", "9.0")
    assert bot_engine._fallback_quote_max_age_seconds() == pytest.approx(9.0)


def test_gap_limit_env_toggle_runtime(monkeypatch):
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "250")
    assert bot_engine._gap_ratio_gate_limit() == pytest.approx(0.025)
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "125")
    assert bot_engine._gap_ratio_gate_limit() == pytest.approx(0.0125)


def test_liquidity_cap_env_toggle_runtime(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LIQ_FALLBACK_CAP", "0.35")
    assert bot_engine._liquidity_fallback_cap() == pytest.approx(0.35)
    monkeypatch.setenv("AI_TRADING_LIQ_FALLBACK_CAP", "0.10")
    assert bot_engine._liquidity_fallback_cap() == pytest.approx(0.10)

