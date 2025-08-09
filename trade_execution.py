# Explicit shim so tests can reliably `from trade_execution import ExecutionEngine`
from ai_trading.trade_execution import ExecutionEngine  # noqa: F401
__all__ = ["ExecutionEngine"]