import time

from freezegun import freeze_time

from ai_trading.core.enums import OrderSide, OrderType
from ai_trading.execution.engine import Order, OrderManager


@freeze_time("2024-01-01 09:30:00", tz_offset=0)
def test_order_monitor_thread_survives_frozen_time():
    manager = OrderManager()
    manager.start_monitoring()
    try:
        order = Order("SPY", OrderSide.BUY, 10, order_type=OrderType.MARKET)
        manager.submit_order(order)
        time.sleep(0.2)
        assert manager._monitor_thread is not None
        assert manager._monitor_thread.is_alive()
    finally:
        manager.stop_monitoring()
