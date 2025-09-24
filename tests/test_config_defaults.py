"""Regression tests for new configuration defaults."""

from __future__ import annotations

import pytest

from ai_trading.config.management import TradingConfig


@pytest.mark.parametrize(
    ("field", "expected", "type_"),
    (
        ("orders_pending_new_warn_s", 60, int),
        ("orders_pending_new_error_s", 180, int),
        ("data_max_gap_ratio_intraday", 0.005, float),
        ("data_daily_fetch_min_interval_s", 60, int),
        ("execution_min_qty", 1, int),
        ("execution_min_notional", 1.0, float),
        ("execution_max_open_orders", 100, int),
        ("logging_dedupe_ttl_s", 120, int),
    ),
)
def test_config_defaults(field: str, expected: float | int, type_: type[object]) -> None:
    cfg = TradingConfig()
    value = getattr(cfg, field)
    assert isinstance(value, type_)
    if isinstance(expected, float):
        assert value == pytest.approx(expected)
    else:
        assert value == expected
