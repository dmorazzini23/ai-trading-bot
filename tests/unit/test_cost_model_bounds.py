from __future__ import annotations

from ai_trading.execution.cost_model import CostModel, CostModelParameters


def test_estimate_cost_bps_clamps_to_configured_max() -> None:
    model = CostModel(
        params=CostModelParameters(
            version="test",
            min_bps=2.0,
            max_bps=25.0,
        )
    )

    estimate = model.estimate_cost_bps(
        spread_bps=5000.0,
        volatility_pct=3.0,
        participation_rate=1.0,
        tca_cost_bps=400.0,
    )
    assert estimate == 25.0


def test_estimate_cost_bps_clamps_to_configured_min() -> None:
    model = CostModel(
        params=CostModelParameters(
            version="test",
            min_bps=2.0,
            max_bps=25.0,
        )
    )

    estimate = model.estimate_cost_bps(
        spread_bps=-5.0,
        volatility_pct=-0.5,
        participation_rate=-0.1,
        tca_cost_bps=-100.0,
    )
    assert estimate == 2.0
