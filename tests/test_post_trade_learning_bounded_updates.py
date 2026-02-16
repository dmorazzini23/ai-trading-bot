from __future__ import annotations

from pathlib import Path

from ai_trading.analytics.post_trade_learning import (
    LearningBounds,
    compute_learning_updates,
    load_learning_overrides,
    write_learning_overrides,
)


def test_post_trade_learning_bounded_updates(tmp_path: Path) -> None:
    updates = compute_learning_updates(
        symbol_metrics={
            "AAPL": {"is_bps": 40.0, "flip_rate": 0.4},
            "MSFT": {"is_bps": 10.0, "flip_rate": 0.1},
        },
        bounds=LearningBounds(max_daily_delta_bps=3.0, max_daily_delta_frac=0.05),
        is_bps_trigger=18.0,
        flip_rate_trigger=0.25,
    )
    symbol_delta = updates["overrides"]["per_symbol_cost_buffer_bps"]["AAPL"]
    assert symbol_delta <= 3.0
    assert updates["overrides"]["global_deadband_frac_delta"] <= 0.05

    path = tmp_path / "learned_overrides.json"
    write_learning_overrides(str(path), updates)
    loaded = load_learning_overrides(str(path), max_age_days=30)
    assert "per_symbol_cost_buffer_bps" in loaded
