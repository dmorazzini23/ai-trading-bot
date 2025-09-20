from __future__ import annotations

import importlib


def test_order_health_monitor_importable_and_singleton_config():
    module = importlib.import_module("ai_trading.monitoring.order_health_monitor")
    monitor = module.OrderHealthMonitor()

    assert module.OrderHealthMonitor is not None
    assert monitor._config is module.CONFIG
