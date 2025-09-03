from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core.bot_engine import SignalManager, BotState, _evaluate_trade_signal


def test_component_signals_unique(monkeypatch):
    sm = SignalManager()
    ctx = SimpleNamespace(signal_manager=sm)
    state = BotState()
    df = pd.DataFrame({"close": [1, 2, 3]})

    def fake_evaluate(self, ctx, state, df, ticker, model):
        self.last_components = [(1, 0.5, "vsa"), (1, 0.3, "vsa")]
        return 1, 0.8, "vsa+vsa"

    monkeypatch.setattr(SignalManager, "evaluate", fake_evaluate, raising=False)
    _evaluate_trade_signal(ctx, state, df, "TEST", None)
    labels = [lab for _, _, lab in ctx.signal_manager.last_components]
    assert len(labels) == len(set(labels))
