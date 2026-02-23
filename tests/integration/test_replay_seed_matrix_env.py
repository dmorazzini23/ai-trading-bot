from __future__ import annotations

import os

from ai_trading.replay.event_loop import ReplayEventLoop


def test_replay_is_deterministic_for_current_seed_env() -> None:
    seed = int(str(os.environ.get("SEED", "42")).strip() or "42")
    bars = [
        {"symbol": "AAPL", "ts": "2026-02-18T15:00:00Z", "close": 190.0},
        {"symbol": "AAPL", "ts": "2026-02-18T15:01:00Z", "close": 190.5},
        {"symbol": "AAPL", "ts": "2026-02-18T15:02:00Z", "close": 191.0},
    ]

    def strategy(bar):
        return {
            "symbol": bar["symbol"],
            "side": "buy",
            "qty": 1,
            "price": bar["close"],
            "intent_key": f"{bar['symbol']}|{bar['ts']}|{seed}",
        }

    first = ReplayEventLoop(strategy=strategy, seed=seed).run(bars)
    second = ReplayEventLoop(strategy=strategy, seed=seed).run(bars)
    assert first["orders"] == second["orders"]
    assert first["events"] == second["events"]
