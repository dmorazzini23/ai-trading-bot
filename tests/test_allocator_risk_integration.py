import os
import types

from ai_trading.strategies.base import StrategySignal
from ai_trading.core.enums import OrderSide
from ai_trading.strategy_allocator import StrategyAllocator
from ai_trading.risk.engine import RiskEngine, TradeSignal
from ai_trading.core.bot_engine import to_trade_signal

os.environ.setdefault("PYTEST_RUNNING", "1")


def test_position_size_accepts_allocator_signal():
    sig = StrategySignal("AAPL", OrderSide.BUY, strength=1.0, confidence=0.9)
    allocator = StrategyAllocator()
    allocated = allocator.allocate({"strat": [sig]})
    assert allocated, "allocator returned no signals"
    trade_sig = to_trade_signal(allocated[0])
    assert isinstance(trade_sig, TradeSignal)
    eng = RiskEngine()
    eng.asset_limits["equity"] = 1.0
    eng.strategy_limits["strat"] = 1.0
    eng.config = types.SimpleNamespace(position_size_min_usd=1, atr_multiplier=1.0)
    qty = eng.position_size(trade_sig, cash=1000.0, price=10.0)
    assert qty > 0
