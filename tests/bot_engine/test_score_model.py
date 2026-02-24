from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from ai_trading.core import bot_engine


def test_score_from_bars_multi_horizon_bias(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SCORE_FAST_BARS", "1")
    monkeypatch.setenv("AI_TRADING_SCORE_SLOW_BARS", "5")
    monkeypatch.setenv("AI_TRADING_SCORE_VOL_BARS", "5")
    monkeypatch.setenv("AI_TRADING_SCORE_FAST_WEIGHT", "0.20")
    monkeypatch.setenv("AI_TRADING_SCORE_Z_CLIP", "3.0")
    monkeypatch.setenv("AI_TRADING_SCORE_MIN_ABS", "0.0")

    # Last bar uptick, but the multi-bar direction remains negative.
    df = pd.DataFrame({"close": [100.0, 98.0, 97.0, 96.0, 95.0, 95.3]})

    score, confidence = bot_engine._score_from_bars(df)

    assert score < 0.0
    assert confidence > 0.0


def test_score_from_bars_noise_gate_zeroes_small_signal(monkeypatch):
    monkeypatch.setenv("AI_TRADING_SCORE_FAST_BARS", "2")
    monkeypatch.setenv("AI_TRADING_SCORE_SLOW_BARS", "6")
    monkeypatch.setenv("AI_TRADING_SCORE_VOL_BARS", "6")
    monkeypatch.setenv("AI_TRADING_SCORE_FAST_WEIGHT", "0.5")
    monkeypatch.setenv("AI_TRADING_SCORE_Z_CLIP", "3.0")
    monkeypatch.setenv("AI_TRADING_SCORE_MIN_ABS", "0.25")

    df = pd.DataFrame(
        {
            "close": [
                100.00,
                100.01,
                100.00,
                100.01,
                100.00,
                100.01,
                100.00,
            ]
        }
    )

    score, confidence = bot_engine._score_from_bars(df)

    assert score == 0.0
    assert confidence == 0.0


def test_resolve_submit_none_reason_prefers_engine_reason():
    runtime = SimpleNamespace(
        execution_engine=SimpleNamespace(
            _last_submit_outcome={"status": "skipped", "reason": "order_pacing_cap"}
        )
    )

    reason = bot_engine._resolve_submit_none_reason(runtime)

    assert reason == "ORDER_PACING_CAP"
