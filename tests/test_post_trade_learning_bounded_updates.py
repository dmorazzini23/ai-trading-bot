from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.analytics.post_trade_learning import (
    LearningBounds,
    compute_learning_updates,
    load_learning_overrides,
    write_learning_overrides,
)
from ai_trading.core import bot_engine


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


def test_post_trade_learning_runtime_schedule_writes_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tca_path = tmp_path / "tca_records.jsonl"
    rows = [
        {"symbol": "AAPL", "side": "buy", "status": "filled", "is_bps": 32.0},
        {"symbol": "AAPL", "side": "sell", "status": "filled", "is_bps": 28.0},
    ]
    tca_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_path = tmp_path / "learned_overrides.json"

    monkeypatch.setenv("AI_TRADING_POST_TRADE_LEARNING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_LEARNING_RUN_SCHEDULE", "daily_first_run")
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_LEARNING_OUTPUT_PATH", str(output_path))
    monkeypatch.setenv("AI_TRADING_MIN_TCA_SAMPLES_FOR_ADAPT", "1")
    monkeypatch.setenv("AI_TRADING_ALLOC_EXPECTANCY_WINDOW_TRADES", "10")
    monkeypatch.setenv("AI_TRADING_LEARNING_IS_BPS_TRIGGER", "18")
    monkeypatch.setenv("AI_TRADING_LEARNING_FLIP_RATE_TRIGGER", "0.25")

    state = bot_engine.BotState()
    run_ts = datetime(2026, 1, 5, 14, 35, tzinfo=UTC)
    bot_engine._run_post_trade_learning_update(
        state,
        now=run_ts,
        market_open_now=True,
    )

    assert state.last_learning_run_date == run_ts.date()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["overrides"]["per_symbol_cost_buffer_bps"]["AAPL"] > 0
