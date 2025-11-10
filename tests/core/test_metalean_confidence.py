from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core import bot_engine


class DummySignalManager(bot_engine.SignalManager):
    def evaluate(self, ctx, state, df, ticker, model):  # noqa: D401
        self.meta_confidence_capped = True
        self.last_components = [(1, 0.8, "momentum")]
        return (1, 0.9, "momentum")


def test_metalean_confidence_clamped(caplog):
    ctx = SimpleNamespace()
    ctx.signal_manager = DummySignalManager()
    state = SimpleNamespace()
    feat_df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

    score, confidence, _ = bot_engine._evaluate_trade_signal(
        ctx,
        state,
        feat_df,
        "AAPL",
        model=None,
    )

    assert score in (-1.0, 0.0, 1.0)
    assert confidence == pytest.approx(bot_engine._metafallback_confidence_cap())
    signal_logs = [record for record in caplog.records if record.message.startswith("SIGNAL_RESULT")]
    assert signal_logs, "SIGNAL_RESULT log not emitted"
    assert getattr(signal_logs[-1], "confidence_capped_due_to_history", False)
