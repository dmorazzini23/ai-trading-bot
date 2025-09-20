import importlib
import sys

import pytest

from ai_trading.config.management import TradingConfig
from ai_trading.config.runtime import (
    ConfigSpec,
    _cast_value,
    _parse_numeric_sequence,
    _strip_inline_comment,
)


def _reload_config(monkeypatch, **env):
    module_name = "ai_trading.config"
    sys.modules.pop(module_name, None)
    for key in [
        "MAX_DRAWDOWN_THRESHOLD",
        "AI_TRADING_MAX_DRAWDOWN_THRESHOLD",
        "TRADING_MODE",
        "KELLY_FRACTION",
        "CONF_THRESHOLD",
        "MAX_POSITION_SIZE",
    ]:
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return importlib.import_module(module_name)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    sys.modules.pop("ai_trading.config", None)


def test_comment_stripped_from_float_env(monkeypatch):
    monkeypatch.delenv("MAX_DRAWDOWN_THRESHOLD", raising=False)
    cfg = TradingConfig.from_env({"MAX_DRAWDOWN_THRESHOLD": "0.08  # comment"})
    assert cfg.max_drawdown_threshold == pytest.approx(0.08)


def test_inline_comments_ignored_for_all_numeric_fields(monkeypatch):
    numeric_env = {
        "CAPITAL_CAP": "0.33  # comment",
        "DOLLAR_RISK_LIMIT": "0.07  # comment",
        "DAILY_LOSS_LIMIT": "0.05  # comment",
        "MAX_POSITION_SIZE": "9000  # comment",
        "MAX_POSITION_EQUITY_FALLBACK": "210000  # comment",
        "AI_TRADING_POSITION_SIZE_MIN_USD": "125  # comment",
        "AI_TRADING_SECTOR_EXPOSURE_CAP": "0.4  # comment",
        "MAX_DRAWDOWN_THRESHOLD": "0.11  # comment",
        "TRAILING_FACTOR": "0.25  # comment",
        "TAKE_PROFIT_FACTOR": "1.9  # comment",
        "KELLY_FRACTION": "0.65  # comment",
        "KELLY_FRACTION_MAX": "0.3  # comment",
        "MIN_CONFIDENCE": "0.7  # comment",
        "CONF_THRESHOLD": "0.77  # comment",
        "MIN_SAMPLE_SIZE": "12  # comment",
        "CONFIDENCE_LEVEL": "0.91  # comment",
        "SIGNAL_CONFIRMATION_BARS": "4  # comment",
        "DELTA_THRESHOLD": "0.03  # comment",
        "AI_TRADING_VOLUME_THRESHOLD": "1200  # comment",
        "MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS": "600  # comment",
        "MAX_DATA_FALLBACKS": "3  # comment",
        "DATA_PROVIDER_BACKOFF_FACTOR": "1.25  # comment",
        "DATA_PROVIDER_MAX_COOLDOWN": "120  # comment",
        "AI_TRADING_HTTP_TIMEOUT": "8.5  # comment",
        "AI_TRADING_HOST_LIMIT": "4  # comment",
        "MAX_EMPTY_RETRIES": "5  # comment",
        "SYMBOL_PROCESS_BUDGET": "120  # comment",
        "CYCLE_BUDGET_FRACTION": "0.6  # comment",
        "CYCLE_COMPUTE_BUDGET": "0.5  # comment",
        "AI_TRADING_SIGNAL_HOLD_EPS": "0.02  # comment",
        "MAX_SYMBOLS_PER_CYCLE": "250  # comment",
        "HEALTH_TICK_SECONDS": "240  # comment",
        "HARD_STOP_COOLDOWN_MIN": "25  # comment",
        "SLIPPAGE_LIMIT_TOLERANCE_BPS": "15.5  # comment",
        "SENTIMENT_BACKOFF_BASE": "7.5  # comment",
        "SENTIMENT_MAX_RETRIES": "8  # comment",
    }
    for key in numeric_env:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("TRADING_MODE", raising=False)

    cfg = TradingConfig.from_env(numeric_env)

    float_expectations = {
        "capital_cap": 0.33,
        "dollar_risk_limit": 0.07,
        "daily_loss_limit": 0.05,
        "max_position_size": 9000.0,
        "max_position_equity_fallback": 210000.0,
        "position_size_min_usd": 125.0,
        "sector_exposure_cap": 0.4,
        "max_drawdown_threshold": 0.11,
        "trailing_factor": 0.25,
        "take_profit_factor": 1.9,
        "kelly_fraction": 0.65,
        "kelly_fraction_max": 0.3,
        "min_confidence": 0.7,
        "conf_threshold": 0.77,
        "confidence_level": 0.91,
        "delta_threshold": 0.03,
        "allow_after_hours_volume_threshold": 1200.0,
        "data_provider_backoff_factor": 1.25,
        "http_timeout_seconds": 8.5,
        "symbol_process_budget_seconds": 120.0,
        "cycle_budget_fraction": 0.6,
        "cycle_compute_budget_factor": 0.5,
        "signal_hold_epsilon": 0.02,
        "health_tick_seconds": 240.0,
        "slippage_limit_tolerance_bps": 15.5,
        "sentiment_backoff_base": 7.5,
    }
    int_expectations = {
        "min_sample_size": 12,
        "signal_confirmation_bars": 4,
        "minute_data_freshness_tolerance_seconds": 600,
        "max_data_fallbacks": 3,
        "data_provider_max_cooldown": 120,
        "host_concurrency_limit": 4,
        "max_empty_retries": 5,
        "max_symbols_per_cycle": 250,
        "hard_stop_cooldown_min": 25,
        "sentiment_retry_max": 8,
    }

    for attr, expected in float_expectations.items():
        assert getattr(cfg, attr) == pytest.approx(expected), attr
    for attr, expected in int_expectations.items():
        assert getattr(cfg, attr) == expected, attr


def test_strip_inline_comment_helper_behaviour():
    assert _strip_inline_comment("42  # answer") == "42"
    assert _strip_inline_comment("true  # enabled") == "true"
    assert _strip_inline_comment("1.0, 2.0  # range") == "1.0, 2.0"
    assert _strip_inline_comment("value#notacomment") == "value#notacomment"
    assert _strip_inline_comment("   # full comment") == ""

    tuple_value = _parse_numeric_sequence("0.1, 0.9  # thresholds")
    assert tuple_value == pytest.approx((0.1, 0.9))

    tuple_spec = ConfigSpec(
        field="demo_tuple",
        env=("DEMO_TUPLE",),
        cast="tuple[float]",
        default=(),
        description="demo",
    )
    assert _cast_value(tuple_spec, "0.2, 0.4  # inline") == pytest.approx((0.2, 0.4))


def test_bool_cast_handles_inline_comments(monkeypatch):
    monkeypatch.delenv("ALLOW_AFTER_HOURS", raising=False)
    cfg = TradingConfig.from_env({
        "ALLOW_AFTER_HOURS": "true  # enable after hours",
    })

    assert cfg.allow_after_hours is True

