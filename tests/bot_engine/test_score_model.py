from __future__ import annotations

from datetime import UTC, datetime, timedelta
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


def test_auth_forbidden_cooldown_records_and_expires(monkeypatch):
    state = bot_engine.BotState()
    now = datetime(2026, 3, 5, 18, 0, tzinfo=UTC)
    monkeypatch.setenv("AI_TRADING_AUTH_FORBIDDEN_COOLDOWN_SECONDS", "120")

    bot_engine._record_auth_forbidden_cooldown(
        state,
        symbol="aapl",
        side="buy",
        reason="AUTH_BROKER_HALT_FORBIDDEN",
        now=now,
    )

    remaining = bot_engine._auth_forbidden_cooldown_remaining_seconds(
        state,
        symbol="AAPL",
        side="buy",
        now=now + timedelta(seconds=30),
    )
    assert 89.0 <= remaining <= 91.0

    expired = bot_engine._auth_forbidden_cooldown_remaining_seconds(
        state,
        symbol="AAPL",
        side="buy",
        now=now + timedelta(seconds=121),
    )
    assert expired == 0.0


def test_auth_forbidden_cooldown_ignores_other_reasons(monkeypatch):
    state = bot_engine.BotState()
    now = datetime(2026, 3, 5, 18, 0, tzinfo=UTC)
    monkeypatch.setenv("AI_TRADING_AUTH_FORBIDDEN_COOLDOWN_SECONDS", "300")

    bot_engine._record_auth_forbidden_cooldown(
        state,
        symbol="AAPL",
        side="buy",
        reason="ORDER_SUBMIT_SKIPPED",
        now=now,
    )

    remaining = bot_engine._auth_forbidden_cooldown_remaining_seconds(
        state,
        symbol="AAPL",
        side="buy",
        now=now,
    )
    assert remaining == 0.0
