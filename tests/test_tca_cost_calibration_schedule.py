from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Mapping

from ai_trading.core import bot_engine
from ai_trading.policy.compiler import decompose_tca_components


def test_tca_cost_calibration_schedule_due_market_close(monkeypatch) -> None:
    state = bot_engine.BotState()
    now = datetime(2026, 2, 19, 22, 0, tzinfo=UTC)

    monkeypatch.setenv("AI_TRADING_TCA_COST_CALIBRATION_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_TCA_COST_CALIBRATION_SCHEDULE", "market_close")

    assert bot_engine._tca_cost_calibration_schedule_due(
        state, now=now, market_open_now=False
    )
    state.last_tca_cost_calibration_date = now.date()
    assert not bot_engine._tca_cost_calibration_schedule_due(
        state, now=now, market_open_now=False
    )


def test_run_tca_cost_calibration_updates_state_and_calls_calibrator(monkeypatch) -> None:
    state = bot_engine.BotState()
    now = datetime(2026, 2, 19, 22, 0, tzinfo=UTC)
    captured: dict[str, object] = {}

    monkeypatch.setenv("AI_TRADING_TCA_COST_CALIBRATION_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_TCA_COST_CALIBRATION_SCHEDULE", "market_close")
    monkeypatch.setenv("AI_TRADING_TCA_COST_CALIBRATION_LOOKBACK_DAYS", "30")
    monkeypatch.setattr(bot_engine, "_resolved_tca_path", lambda: "runtime/tca_records.jsonl")

    def _fake_calibrate_cost_model_from_tca(**kwargs):
        captured.update(kwargs)
        return {
            "records": 15,
            "model_path": kwargs.get("model_path"),
            "after": {"version": "calibrated-test"},
        }

    monkeypatch.setattr(bot_engine, "calibrate_cost_model_from_tca", _fake_calibrate_cost_model_from_tca)

    bot_engine._run_tca_cost_calibration(state, now=now, market_open_now=False)

    assert captured["lookback_days"] == 30
    assert captured["tca_path"] == "runtime/tca_records.jsonl"
    assert state.last_tca_cost_calibration_date == now.date()


def test_tca_feedback_penalty_map_filters_and_caps() -> None:
    penalties = bot_engine._tca_feedback_penalty_map(
        {
            "AAPL": {"median_bps": 20.0, "samples": 25},
            "MSFT": {"median_bps": 10.0, "samples": 25},
            "TSLA": {"median_bps": 30.0, "samples": 5},
            "09": {"median_bps": 16.0, "samples": 40},
        },
        min_samples=10,
        target_total_bps=12.0,
        max_penalty_bps=6.0,
    )

    assert penalties["AAPL"] == 6.0
    assert penalties["09"] == 4.0
    assert "MSFT" not in penalties
    assert "TSLA" not in penalties


def test_refresh_tca_feedback_components_loads_payload(monkeypatch, tmp_path) -> None:
    feedback_path = tmp_path / "tca_feedback.json"
    feedback_path.write_text(
        json.dumps({"edge_floor_adjust_bps": 1.25, "symbol_cost_penalty_bps": {"AAPL": 2.0}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_TCA_FEEDBACK_PATH", str(feedback_path))

    state = bot_engine.BotState()
    bot_engine._refresh_tca_feedback_components(state, force=True)

    payload = getattr(state, "_tca_feedback_components", {})
    assert isinstance(payload, dict)
    assert payload.get("edge_floor_adjust_bps") == 1.25
    assert payload.get("symbol_cost_penalty_bps", {}).get("AAPL") == 2.0
    assert float(getattr(state, "_tca_feedback_loaded_mono", 0.0) or 0.0) > 0.0


def test_decompose_tca_components_includes_symbol_and_hour_profiles() -> None:
    rows: list[Mapping[str, Any]] = [
        {
            "symbol": "AAPL",
            "is_bps": 10.0,
            "spread_paid_bps": 3.0,
            "fill_latency_ms": 100,
            "ts": "2026-03-13T15:00:00+00:00",
        },
        {
            "symbol": "AAPL",
            "is_bps": 20.0,
            "spread_paid_bps": 4.0,
            "fill_latency_ms": 120,
            "ts": "2026-03-13T15:05:00+00:00",
        },
        {
            "symbol": "MSFT",
            "is_bps": 30.0,
            "spread_paid_bps": 5.0,
            "fill_latency_ms": 200,
            "ts": "2026-03-13T16:00:00+00:00",
        },
    ]

    out = decompose_tca_components(rows)

    assert out["sample_count"] == 3.0
    assert out["by_symbol_total_bps"]["AAPL"]["samples"] == 2.0
    assert out["by_symbol_total_bps"]["AAPL"]["median_bps"] == 15.0
    assert out["by_hour_total_bps"]["15"]["samples"] == 2.0
    assert out["by_hour_total_bps"]["16"]["samples"] == 1.0
