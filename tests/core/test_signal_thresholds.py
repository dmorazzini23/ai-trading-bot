from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import BotState


def test_meta_confidence_cap_lowers_buy_threshold(monkeypatch):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None))
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "_mark_primary_provider_fallback", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "_clear_primary_provider_fallback", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_check_trade_frequency_limits", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_enter_short", lambda *a, **k: pytest.fail("unexpected short path"))
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.6)

    def fake_eval(*_args, **_kwargs):
        ctx.signal_manager.meta_confidence_capped = True
        return 1.0, 0.35, "meta"

    monkeypatch.setattr(bot_engine, "_evaluate_trade_signal", fake_eval)

    captured: dict[str, object] = {}

    def fake_enter_long(*args):
        captured["called"] = True
        captured["confidence"] = args[6]
        return True

    monkeypatch.setattr(bot_engine, "_enter_long", fake_enter_long)

    result = bot_engine.trade_logic(ctx, state, "TEST", balance=100000.0, model=None, regime_ok=True)

    assert result is True
    assert captured.get("called") is True
    assert captured.get("confidence") == pytest.approx(0.35)
