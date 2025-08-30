import importlib

from ai_trading.position_sizing import get_max_position_size
from ai_trading.core.runtime import build_runtime


def test_bot_engine_and_position_sizing_agree(monkeypatch):
    monkeypatch.delenv("AI_TRADING_MAX_POSITION_SIZE", raising=False)
    be = importlib.reload(importlib.import_module("ai_trading.core.bot_engine"))
    runtime = build_runtime(be.state.mode_obj.config)
    cfg = runtime.cfg
    assert be.MAX_POSITION_SIZE == runtime.params["MAX_POSITION_SIZE"]
    assert be.MAX_POSITION_SIZE == get_max_position_size(cfg, cfg)

