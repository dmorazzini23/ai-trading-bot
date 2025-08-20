from ai_trading.config.management import TradingConfig
from ai_trading.core.bot_engine import get_allocator
from ai_trading.core.runtime import BotRuntime


def test_runtime_has_allocator_when_built():
    cfg = TradingConfig()
    rt = BotRuntime(cfg=cfg)
    assert getattr(rt, "allocator", None) is None
    rt.allocator = get_allocator()
    assert rt.allocator is not None
