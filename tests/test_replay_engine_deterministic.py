from __future__ import annotations

import pytest

from ai_trading.replay.replay_engine import ReplayConfig, ReplayEngine


def _pipeline(event: dict[str, object]) -> dict[str, object]:
    close_raw = event["close"]
    close = float(close_raw) if isinstance(close_raw, (int, float)) else 0.0
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


def test_replay_engine_next_bar_fill_avoids_same_bar_leakage() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42, fill_slippage_bps=0.0)
    bars = [
        {"symbol": "AAPL", "ts": "2025-01-01T10:00:00Z", "close": 100.0, "mid": 100.0},
        {"symbol": "AAPL", "ts": "2025-01-01T10:05:00Z", "close": 101.0, "mid": 101.25},
    ]

    result = ReplayEngine(config, pipeline=_pipeline).run(bars)

    filled_entries = [row for row in result["ledger_entries"] if row.get("status", "filled") == "filled"]
    assert len(filled_entries) == 1
    assert any(row.get("status") == "canceled" for row in result["ledger_entries"])
    assert filled_entries[0]["price"] == pytest.approx(101.25)
    tca = result["tca_records"][0]
    assert tca["decision_price"] == pytest.approx(100.0)
    assert tca["fill_price"] == pytest.approx(101.25)
    assert tca["fill_latency_ms"] == 300_000
    assert tca["benchmark"]["decision_ts"] == "2025-01-01T10:00:00+00:00"
    assert tca["benchmark"]["first_fill_ts"] == "2025-01-01T10:05:00+00:00"


def test_replay_engine_enforces_oms_gate_blocks() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42, fill_model="close")

    def blocked_pipeline(event: dict[str, object]) -> dict[str, object]:
        decision = _pipeline(event)
        decision["gates"] = ["OK_TRADE", "RISK_PORTFOLIO_HARD_BLOCK"]
        return decision

    result = ReplayEngine(config, pipeline=blocked_pipeline).run(
        [{"symbol": "AAPL", "ts": "2025-01-01T10:00:00Z", "close": 100.0}]
    )

    assert len(result["decision_records"]) == 1
    assert result["ledger_entries"] == []
    assert result["tca_records"] == []


def test_replay_engine_blocks_real_broker_submit() -> None:
    config = ReplayConfig(symbols=("AAPL",), seed=42)
    engine = ReplayEngine(config, pipeline=_pipeline, broker_submit=lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="must not execute real broker submits"):
        engine.run([])
