import os
import pytest

from ai_trading.execution import ExecutionEngine
from ai_trading.core.enums import OrderSide
from ai_trading.core.constants import EXECUTION_PARAMETERS


def test_slippage_threshold_enforced(monkeypatch):
    os.environ["TESTING"] = "true"
    monkeypatch.setitem(EXECUTION_PARAMETERS, "MAX_SLIPPAGE_BPS", 10)
    engine = ExecutionEngine()
    # Force deterministic high slippage
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 99, raising=False)
    with pytest.raises(AssertionError):
        engine.execute_order("AAPL", OrderSide.BUY, 10, price=100.0)
