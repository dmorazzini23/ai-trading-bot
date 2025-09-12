from types import SimpleNamespace

from ai_trading.execution.engine import ExecutionEngine, OrderSide
from ai_trading.core.bot_engine import compute_current_positions


def test_simulated_fills_update_position_ledger():
    engine = ExecutionEngine()
    engine.available_qty = 100

    class DummyApi:
        pass

    ctx = SimpleNamespace(api=DummyApi(), execution_engine=engine)
    engine.ctx = ctx

    engine.execute_order("AAPL", OrderSide.BUY, 10)
    assert engine.position_ledger == {"AAPL": 10}

    engine.execute_order("AAPL", OrderSide.SELL, 4)
    assert engine.position_ledger == {"AAPL": 6}

    positions = compute_current_positions(ctx)
    assert positions == {"AAPL": 6}

