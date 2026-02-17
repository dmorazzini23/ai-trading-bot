from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.core import bot_engine
from ai_trading.strategies.base import StrategySignal


class DummyStrategy:
    name = "dummy"

    def generate_signals(self, ctx):
        return []


class FailAllocator:
    def allocate(self, signals):  # pragma: no cover - should not be called
        raise AssertionError("allocate should not be called")


def test_run_strategy_no_signals(monkeypatch):
    ctx = SimpleNamespace(
        strategies=[DummyStrategy()],
        allocator=FailAllocator(),
        api=SimpleNamespace(list_positions=lambda: []),
        data_fetcher=SimpleNamespace(
            get_daily_df=lambda ctx, sym: pd.DataFrame(),
            get_minute_df=lambda ctx, sym: pd.DataFrame(),
        ),
    )

    monkeypatch.setattr(bot_engine, "RL_AGENT", None)
    import ai_trading.signals as sig

    monkeypatch.setattr(sig, "generate_position_hold_signals", lambda ctx, pos: [])
    monkeypatch.setattr(sig, "enhance_signals_with_position_logic", lambda s, ctx, h: s)

    bot_engine.run_multi_strategy(ctx)


def test_run_strategy_rl_feature_extraction_uses_feature_config(monkeypatch):
    class StrategyWithSignal:
        name = "dummy"

        def generate_signals(self, _ctx):
            return [StrategySignal(symbol="AAPL", side="buy", strength=1.0, confidence=1.0)]

    class DummyAllocator:
        def allocate(self, _signals):
            return []

    class DummyRLAgent:
        def __init__(self):
            self.calls: list[tuple[object, list[str] | None]] = []

        def predict(self, state, symbols=None):
            self.calls.append((state, symbols))
            return []

    bars = pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0, 101.5],
            "high": [101.0, 101.5, 102.0, 102.5],
            "low": [99.0, 99.5, 100.0, 100.5],
            "close": [100.2, 100.7, 101.2, 101.7],
            "volume": [1000, 1100, 1200, 1300],
        }
    )
    rl_agent = DummyRLAgent()
    ctx = SimpleNamespace(
        strategies=[StrategyWithSignal()],
        allocator=DummyAllocator(),
        api=SimpleNamespace(list_positions=lambda: []),
        data_fetcher=SimpleNamespace(
            get_daily_df=lambda _ctx, _sym: bars,
            get_minute_df=lambda _ctx, _sym: pd.DataFrame(),
        ),
    )

    monkeypatch.setattr(bot_engine, "RL_AGENT", rl_agent)
    import ai_trading.signals as sig

    monkeypatch.setattr(sig, "generate_position_hold_signals", lambda _ctx, _pos: [])
    monkeypatch.setattr(sig, "enhance_signals_with_position_logic", lambda s, _ctx, _h: s)

    bot_engine.run_multi_strategy(ctx)

    assert len(rl_agent.calls) == 1
    _, symbols = rl_agent.calls[0]
    assert symbols == ["AAPL"]
