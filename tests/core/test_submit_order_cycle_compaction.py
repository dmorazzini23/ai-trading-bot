from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core import bot_engine


class _DummyExecEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, int, float | None]] = []
        self.skips: list[dict[str, object]] = []
        self._last_submit_outcome: dict[str, object] = {}

    def execute_order(self, symbol: str, side: object, qty: int, *, price: float | None = None, **_: object):
        self.calls.append((symbol, side, qty, price))
        return SimpleNamespace(id="ok")

    def _skip_submit(self, **kwargs: object) -> None:
        self.skips.append(dict(kwargs))
        self._last_submit_outcome = {
            "status": "skipped",
            "reason": kwargs.get("reason"),
            "symbol": kwargs.get("symbol"),
            "side": kwargs.get("side"),
        }


def test_submit_order_cycle_intent_compaction(monkeypatch):
    engine = _DummyExecEngine()
    ctx = SimpleNamespace(market_data=None)

    monkeypatch.setattr(bot_engine, "_exec_engine", engine)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "get_latest_price", lambda _symbol: 100.0)
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _ctx: SimpleNamespace(rth_only=False, allow_extended=True))
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setenv("AI_TRADING_CYCLE_INTENT_COMPACTION_ENABLED", "1")

    monkeypatch.setattr(bot_engine.state, "running", True, raising=False)
    bot_engine.state.cycle_submit_compaction = set()

    first = bot_engine.submit_order(ctx, "AAPL", 5, "buy", price=100.0)
    duplicate = bot_engine.submit_order(ctx, "AAPL", 5, "buy", price=100.0)
    opposite_side = bot_engine.submit_order(ctx, "AAPL", 5, "sell", price=100.0)

    assert first is not None
    assert duplicate is None
    assert opposite_side is not None
    assert len(engine.calls) == 2
    assert len(engine.skips) == 1
    assert engine.skips[0]["reason"] == "cycle_duplicate_intent"
    assert engine.skips[0]["symbol"] == "AAPL"
