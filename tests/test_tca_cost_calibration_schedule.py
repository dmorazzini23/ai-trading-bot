from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.core import bot_engine


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
