from __future__ import annotations

from ai_trading.portfolio.allocation import SleevePerfState, update_allocation_weights


def test_allocation_weight_updates_respect_bounds() -> None:
    base = {"day": 0.4, "swing": 0.35, "longshort": 0.25}
    perf = {
        "day": SleevePerfState(rolling_expectancy=0.01, drawdown=0.01, stability_score=0.9, trade_count=50, confidence=0.9),
        "swing": SleevePerfState(rolling_expectancy=-0.01, drawdown=0.08, stability_score=0.4, trade_count=50, confidence=0.8),
        "longshort": SleevePerfState(rolling_expectancy=0.0, drawdown=0.02, stability_score=0.5, trade_count=10, confidence=0.5),
    }
    updated = update_allocation_weights(
        base_weights=base,
        perf_states=perf,
        min_weight=0.10,
        max_weight=0.60,
        daily_max_delta=0.05,
        expectancy_floor=-0.001,
        drawdown_trigger=0.06,
        min_trades_for_adjust=15,
    )
    assert abs(sum(updated.values()) - 1.0) < 1e-9
    assert updated["day"] > base["day"] - 1e-6
    assert updated["swing"] < base["swing"] + 1e-6
