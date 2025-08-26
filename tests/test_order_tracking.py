import time
from unittest.mock import Mock, patch

from ai_trading.execution.engine import ExecutionEngine
from ai_trading.monitoring.order_health_monitor import _active_orders, _order_tracking_lock


def test_track_and_cleanup():
    """Orders are tracked and stale ones cleaned up."""
    with _order_tracking_lock:
        _active_orders.clear()
    engine = ExecutionEngine()
    order = Mock()
    order.id = "o1"
    order.symbol = "AAPL"
    order.side = "buy"
    order.quantity = 10
    order.status = "new"
    engine.track_order(order)
    assert any(o.order_id == "o1" for o in engine.get_pending_orders())
    engine._update_order_status("o1", "filled")
    assert engine.get_pending_orders() == []
    stale = Mock()
    stale.id = "o2"
    stale.symbol = "MSFT"
    stale.side = "sell"
    stale.quantity = 5
    stale.status = "new"
    engine.track_order(stale)
    with _order_tracking_lock:
        _active_orders["o2"].submitted_time = time.time() - 1000
    with patch.object(engine, "_cancel_stale_order", return_value=True):
        removed = engine.cleanup_stale_orders(max_age_seconds=600)
    assert removed == 1
