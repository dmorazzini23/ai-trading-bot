import importlib

from ai_trading.position_sizing import get_max_position_size


def test_bot_engine_and_position_sizing_agree(monkeypatch):
    monkeypatch.delenv("AI_TRADING_MAX_POSITION_SIZE", raising=False)
    be = importlib.reload(importlib.import_module("ai_trading.core.bot_engine"))
    cfg = be.S
    tcfg = be.state.mode_obj.config
    assert be.MAX_POSITION_SIZE == get_max_position_size(cfg, tcfg)

