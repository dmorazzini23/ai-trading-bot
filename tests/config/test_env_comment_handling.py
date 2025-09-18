import importlib
import sys

import pytest

from ai_trading.config.management import TradingConfig


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
    cfg = _reload_config(monkeypatch, MAX_DRAWDOWN_THRESHOLD="0.08  # comment")
    assert cfg.get_max_drawdown_threshold() == pytest.approx(0.08)


def test_inline_comments_ignored_for_all_numeric_fields(monkeypatch):
    numeric_env = {
        "CAPITAL_CAP": "0.33  # comment",
        "DOLLAR_RISK_LIMIT": "0.07  # comment",
        "DAILY_LOSS_LIMIT": "0.05  # comment",
        "MAX_POSITION_SIZE": "9000  # comment",
        "MAX_POSITION_EQUITY_FALLBACK": "210000  # comment",
        "POSITION_SIZE_MIN_USD": "125  # comment",
        "SECTOR_EXPOSURE_CAP": "0.4  # comment",
        "MAX_DRAWDOWN_THRESHOLD": "0.11  # comment",
        "TRAILING_FACTOR": "0.25  # comment",
        "TAKE_PROFIT_FACTOR": "1.9  # comment",
        "MAX_POSITION_SIZE_PCT": "0.15  # comment",
        "MAX_VAR_95": "0.05  # comment",
        "MIN_PROFIT_FACTOR": "1.5  # comment",
        "MIN_SHARPE_RATIO": "1.2  # comment",
        "MIN_WIN_RATE": "0.55  # comment",
        "KELLY_FRACTION": "0.65  # comment",
        "KELLY_FRACTION_MAX": "0.3  # comment",
        "MIN_SAMPLE_SIZE": "12  # comment",
        "CONFIDENCE_LEVEL": "0.91  # comment",
        "CONF_THRESHOLD": "0.77  # comment",
        "LOOKBACK_PERIODS": "45  # comment",
        "SCORE_CONFIDENCE_MIN": "0.5  # comment",
        "SIGNAL_CONFIRMATION_BARS": "4  # comment",
        "DELTA_THRESHOLD": "0.03  # comment",
        "MIN_CONFIDENCE": "0.7  # comment",
        "MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS": "600  # comment",
        "BUY_THRESHOLD": "0.61  # comment",
        "SIGNAL_PERIOD": "25  # comment",
        "FAST_PERIOD": "12  # comment",
        "SLOW_PERIOD": "30  # comment",
        "LIMIT_ORDER_SLIPPAGE": "0.02  # comment",
        "MAX_SLIPPAGE_BPS": "35  # comment",
        "PARTICIPATION_RATE": "0.6  # comment",
        "POV_SLICE_PCT": "0.2  # comment",
        "ORDER_TIMEOUT_SECONDS": "120  # comment",
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
        "max_position_size_pct": 0.15,
        "max_var_95": 0.05,
        "min_profit_factor": 1.5,
        "min_sharpe_ratio": 1.2,
        "min_win_rate": 0.55,
        "kelly_fraction": 0.65,
        "kelly_fraction_max": 0.3,
        "confidence_level": 0.91,
        "conf_threshold": 0.77,
        "score_confidence_min": 0.5,
        "delta_threshold": 0.03,
        "min_confidence": 0.7,
        "buy_threshold": 0.61,
        "limit_order_slippage": 0.02,
        "participation_rate": 0.6,
        "pov_slice_pct": 0.2,
    }
    int_expectations = {
        "min_sample_size": 12,
        "lookback_periods": 45,
        "signal_confirmation_bars": 4,
        "minute_data_freshness_tolerance_seconds": 600,
        "signal_period": 25,
        "fast_period": 12,
        "slow_period": 30,
        "max_slippage_bps": 35,
        "order_timeout_seconds": 120,
    }

    for attr, expected in float_expectations.items():
        assert getattr(cfg, attr) == pytest.approx(expected), attr
    for attr, expected in int_expectations.items():
        assert getattr(cfg, attr) == expected, attr

