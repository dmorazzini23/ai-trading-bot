from types import SimpleNamespace

import pandas as pd

from ai_trading.core import bot_engine


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
        api=SimpleNamespace(list_open_positions=lambda: []),
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
