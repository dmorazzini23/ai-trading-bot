"""Pin the presence/behavior of TradingConfig aliases and tunables.

This guards against future regressions where legacy names disappear or
script-only knobs get dropped from the model surface.
"""
from __future__ import annotations

from ai_trading.config.management import TradingConfig


def test_trading_config_aliases_and_tunables_exist():
    cfg = TradingConfig()

    # Back-compat aliases map to canonical fields
    assert cfg.max_correlation_exposure == cfg.sector_exposure_cap
    assert cfg.max_drawdown == cfg.max_drawdown_threshold
    assert cfg.stop_loss_multiplier == cfg.trailing_factor
    assert cfg.take_profit_multiplier == cfg.take_profit_factor

    # Script/optimizer tunables exist (may be None by default)
    for name in (
        "max_position_size_pct",
        "max_var_95",
        "min_profit_factor",
        "min_sharpe_ratio",
        "min_win_rate",
    ):
        assert hasattr(cfg, name), f"Missing tunable: {name}"

