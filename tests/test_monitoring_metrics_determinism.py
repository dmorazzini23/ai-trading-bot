from __future__ import annotations

import ai_trading.monitoring.metrics as metrics_mod
from ai_trading.monitoring.metrics import MetricsCollector


def test_metrics_collector_label_key_is_order_invariant() -> None:
    collector = MetricsCollector()

    collector.inc_counter("orders_total", labels={"symbol": "AAPL", "side": "buy"})
    collector.inc_counter("orders_total", labels={"side": "buy", "symbol": "AAPL"})

    assert len(collector.counters) == 1
    key, value = next(iter(collector.counters.items()))
    assert key.startswith("orders_total_")
    assert value == 2


def test_metrics_collector_does_not_use_builtin_hash(
    monkeypatch,
) -> None:
    def _boom(*_args, **_kwargs) -> int:
        raise AssertionError("module-level hash() should not be used for metric keys")

    monkeypatch.setattr(metrics_mod, "hash", _boom, raising=False)
    collector = MetricsCollector()
    collector.inc_counter("orders_total", labels={"symbol": "MSFT"})
    collector.observe_latency("submit_latency_ms", 12.5, labels={"symbol": "MSFT"})
    collector.gauge_set("last_price", 123.45, labels={"symbol": "MSFT"})
