"""Ensure TradingConfig exposes expected canonical fields."""

from ai_trading.config.management import TradingConfig


def test_trading_config_canonical_fields_exist():
    """TradingConfig should expose canonical field names."""
    cfg = TradingConfig()
    for name in (
        "sector_exposure_cap",
        "max_drawdown_threshold",
        "trailing_factor",
        "take_profit_factor",
    ):
        assert hasattr(cfg, name), f"Missing field: {name}"

