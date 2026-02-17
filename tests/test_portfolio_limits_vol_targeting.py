from __future__ import annotations

from ai_trading.risk.portfolio_limits import apply_portfolio_limits


def test_portfolio_limits_applies_vol_scaling() -> None:
    targets = {"AAPL": 10000.0, "MSFT": 8000.0}
    symbol_returns = {
        "AAPL": [0.04, -0.03, 0.05, -0.04, 0.03],
        "MSFT": [0.03, -0.02, 0.04, -0.03, 0.02],
    }
    result = apply_portfolio_limits(
        targets=targets,
        symbol_returns=symbol_returns,
        target_annual_vol=0.10,
        vol_min_scale=0.25,
        vol_max_scale=1.25,
    )
    assert result.scale <= 1.0
    assert "VOL_TARGET_SCALE" in result.reasons


def test_portfolio_limits_uses_weighted_portfolio_returns() -> None:
    result = apply_portfolio_limits(
        targets={"AAPL": 10000.0, "MSFT": -10000.0},
        symbol_returns={
            "AAPL": [0.03, -0.03, 0.02, -0.02],
            "MSFT": [0.03, -0.03, 0.02, -0.02],
        },
        target_annual_vol=0.10,
        vol_min_scale=0.25,
        vol_max_scale=1.25,
    )
    assert result.scale == 1.0
    assert "VOL_TARGET_SCALE" not in result.reasons
