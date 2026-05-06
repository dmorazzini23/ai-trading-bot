from __future__ import annotations

import pytest

from ai_trading.portfolio.allocation import SleevePerfState, update_allocation_weights


def test_allocation_normalization_preserves_max_weight_after_clamping() -> None:
    weights = update_allocation_weights(
        base_weights={"day": 0.90, "swing": 0.05, "longshort": 0.05},
        perf_states={
            "day": SleevePerfState(
                rolling_expectancy=0.10,
                drawdown=0.0,
                stability_score=1.0,
                trade_count=20,
                confidence=1.0,
            )
        },
        min_weight=0.05,
        max_weight=0.40,
        daily_max_delta=0.05,
        expectancy_floor=0.0,
        drawdown_trigger=0.10,
        min_trades_for_adjust=10,
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 0.400001
    assert weights["day"] == pytest.approx(0.40)
