"""Regression tests for newly introduced runtime configuration defaults."""

from __future__ import annotations

from ai_trading.config.runtime import TradingConfig


def test_new_config_defaults(monkeypatch):
    """New configuration fields expose sane default values and types."""

    for env_name in (
        "ORDERS_PENDING_NEW_WARN_S",
        "ORDERS_PENDING_NEW_ERROR_S",
        "DATA_MAX_GAP_RATIO_INTRADAY",
        "DATA_DAILY_FETCH_MIN_INTERVAL_S",
        "EXECUTION_MIN_QTY",
        "EXECUTION_MIN_NOTIONAL",
        "EXECUTION_MAX_OPEN_ORDERS",
        "LOGGING_DEDUPE_TTL_S",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.2")

    cfg = TradingConfig.from_env()

    assert isinstance(cfg.orders_pending_new_warn_s, int)
    assert cfg.orders_pending_new_warn_s == 60

    assert isinstance(cfg.orders_pending_new_error_s, int)
    assert cfg.orders_pending_new_error_s == 180

    assert isinstance(cfg.data_max_gap_ratio_intraday, float)
    assert cfg.data_max_gap_ratio_intraday == 0.005

    assert isinstance(cfg.data_daily_fetch_min_interval_s, int)
    assert cfg.data_daily_fetch_min_interval_s == 60

    assert isinstance(cfg.execution_min_qty, int)
    assert cfg.execution_min_qty == 1

    assert isinstance(cfg.execution_min_notional, float)
    assert cfg.execution_min_notional == 1.0

    assert isinstance(cfg.execution_max_open_orders, int)
    assert cfg.execution_max_open_orders == 100

    assert isinstance(cfg.logging_dedupe_ttl_s, int)
    assert cfg.logging_dedupe_ttl_s == 120
