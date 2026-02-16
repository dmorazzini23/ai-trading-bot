from __future__ import annotations

import pytest

from ai_trading.replay.replay_engine import ReplayConfig, ReplayEngine


def _pipeline(event: dict[str, object]) -> dict[str, object]:
    close = float(event["close"])
    return {
        "symbol": event["symbol"],
        "ts": event["ts"],
        "sleeve": "day",
        "regime_profile": "balanced",
        "order": {
            "side": "buy",
            "qty": 10,
            "price": close,
            "order_type": "limit",
            "client_order_id": f"{event['symbol']}-{event['ts']}",
        },
    }


def test_replay_engine_is_deterministic() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42)
    bars = [
        {"symbol": "AAPL", "ts": "2025-01-01T10:00:00Z", "close": 100.0, "mid": 100.0},
        {"symbol": "AAPL", "ts": "2025-01-01T10:05:00Z", "close": 101.0, "mid": 101.0},
    ]
    engine = ReplayEngine(config, pipeline=_pipeline)
    first = engine.run(bars)
    second = engine.run(bars)
    assert first == second


def test_replay_engine_blocks_real_broker_submit() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42)
    engine = ReplayEngine(config, pipeline=_pipeline, broker_submit=lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="must not execute real broker submits"):
        engine.run([])
