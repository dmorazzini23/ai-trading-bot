from __future__ import annotations

import pytest

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


def test_portfolio_limits_can_disable_vol_targeting() -> None:
    result = apply_portfolio_limits(
        targets={"AAPL": 10000.0, "MSFT": 9000.0},
        symbol_returns={
            "AAPL": [0.06, -0.05, 0.05, -0.04],
            "MSFT": [0.05, -0.04, 0.04, -0.03],
        },
        vol_targeting_enabled=False,
        target_annual_vol=0.01,
    )
    assert result.scale == 1.0
    assert "VOL_TARGET_SCALE" not in result.reasons


def test_portfolio_limits_corr_cap_respects_cluster_cap() -> None:
    result = apply_portfolio_limits(
        targets={"AAPL": 10000.0, "MSFT": 10000.0, "GOOG": 10000.0},
        symbol_returns={
            "AAPL": [0.02, -0.01, 0.02, -0.01, 0.02],
            "MSFT": [0.02, -0.01, 0.02, -0.01, 0.02],
            "GOOG": [0.01, 0.00, -0.01, 0.00, 0.01],
        },
        vol_targeting_enabled=False,
        concentration_cap_enabled=False,
        corr_cap_enabled=True,
        corr_threshold=0.7,
        corr_group_gross_cap=0.9,
        max_cluster_weight=0.2,
    )
    assert result.scaled_targets["AAPL"] == pytest.approx(3000.0)
    assert result.scaled_targets["MSFT"] == pytest.approx(3000.0)
    assert "CORR_CLUSTER_CAP" in result.reasons
