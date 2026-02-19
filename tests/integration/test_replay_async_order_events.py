from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.replay.event_loop import ReplayEventLoop


def _parse_utc(text: str) -> datetime:
    candidate = text
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def test_replay_event_loop_emits_async_fill_events_deterministically() -> None:
    bars = [
        {"symbol": "AAPL", "ts": "2026-02-18T15:00:00Z", "close": 190.0},
        {"symbol": "AAPL", "ts": "2026-02-18T15:01:00Z", "close": 190.5},
        {"symbol": "AAPL", "ts": "2026-02-18T15:02:00Z", "close": 191.0},
    ]

    def strategy(bar):
        return {
            "symbol": bar["symbol"],
            "side": "buy",
            "qty": 2,
            "price": bar["close"],
            "intent_key": f"{bar['symbol']}|{bar['ts']}",
        }

    first = ReplayEventLoop(strategy=strategy, seed=123).run(bars)
    second = ReplayEventLoop(strategy=strategy, seed=123).run(bars)
    assert first["events"] == second["events"]

    fills = [event for event in first["events"] if event.get("event_type") == "fill"]
    assert fills
    first_bar = _parse_utc(bars[0]["ts"])
    second_bar = _parse_utc(bars[1]["ts"])
    assert any(first_bar < _parse_utc(str(event["ts"])) < second_bar for event in fills)

