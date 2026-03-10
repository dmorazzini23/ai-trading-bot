from __future__ import annotations

from ai_trading.execution.simulated_broker import SimulatedBroker
from ai_trading.replay.event_loop import ReplayEventLoop


def test_replay_parity_invariant_blocks_duplicate_intents() -> None:
    bars = [
        {"symbol": "AAPL", "ts": "2026-02-18T15:10:00Z", "close": 192.0},
        {"symbol": "AAPL", "ts": "2026-02-18T15:11:00Z", "close": 192.2},
    ]

    def strategy(bar):
        return {
            "symbol": bar["symbol"],
            "side": "buy",
            "qty": 1,
            "price": bar["close"],
            "intent_key": "dup-key",
        }

    result = ReplayEventLoop(strategy=strategy, seed=7).run(bars)
    codes = {item["code"] for item in result["violations"]}
    assert "duplicate_intent" in codes


def test_replay_parity_invariant_blocks_position_cap_breaches() -> None:
    bars = [
        {"symbol": "MSFT", "ts": "2026-02-18T15:20:00Z", "close": 410.0},
    ]

    def strategy(bar):
        return {
            "symbol": bar["symbol"],
            "side": "buy",
            "qty": 10,
            "price": bar["close"],
            "intent_key": "cap-test-1",
        }

    result = ReplayEventLoop(
        strategy=strategy,
        seed=99,
        max_symbol_notional=1000.0,
        max_gross_notional=1000.0,
    ).run(bars)
    codes = {item["code"] for item in result["violations"]}
    assert "position_cap_exceeded" in codes
    assert result["orders"] == []


def test_replay_gross_notional_uses_symbol_specific_prices() -> None:
    bars = [
        {"symbol": "PENNY", "ts": "2026-02-18T15:30:00Z", "close": 1.0},
        {"symbol": "BRKA", "ts": "2026-02-18T15:31:00Z", "close": 500.0},
    ]

    def strategy(bar):
        if bar["symbol"] == "PENNY":
            return {
                "symbol": "PENNY",
                "side": "buy",
                "qty": 50,
                "price": bar["close"],
                "intent_key": "cheap-1",
            }
        return {
            "symbol": "BRKA",
            "side": "buy",
            "qty": 0.1,
            "price": bar["close"],
            "intent_key": "expensive-1",
        }

    broker = SimulatedBroker(
        seed=17,
        fill_probability=1.0,
        partial_fill_probability=0.0,
        min_fill_delay_ms=0,
        max_fill_delay_ms=0,
    )
    result = ReplayEventLoop(
        strategy=strategy,
        broker=broker,
        seed=17,
        max_symbol_notional=1_000.0,
        max_gross_notional=120.0,
    ).run(bars)

    codes = {item["code"] for item in result["violations"]}
    assert "position_cap_exceeded" not in codes
    assert len(result["orders"]) == 2
