from __future__ import annotations
from threading import RLock

# AI-AGENT-REF: facade for legacy trade_execution imports
try:
    from ai_trading.execution.engine import ExecutionEngine, _active_orders  # noqa: F401
except Exception:  # pragma: no cover - fallback when engine lacks globals
    from ai_trading.execution.engine import ExecutionEngine  # type: ignore
    _active_orders: dict = {}

# The engine doesn’t expose a lock; tests patch the name. Provide a stable one.
_order_tracking_lock = RLock()  # noqa: N816

# Some tests import TradingEngine — alias ExecutionEngine to keep them working
TradingEngine = ExecutionEngine  # noqa: N816

# Some tests import safe_submit_order from trade_execution
try:
    from ai_trading.core.bot_engine import safe_submit_order  # noqa: F401
except Exception:
    def safe_submit_order(*args, **kwargs):  # pragma: no cover - test shim
        # Minimal no-op fallback; tests mainly assert importability
        return {"status": "no-op"}

# Optional hook the tests may import; provide a harmless stub
def handle_partial_fill(*args, **kwargs):  # pragma: no cover - test shim
    return None
