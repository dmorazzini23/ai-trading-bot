import importlib

from ai_trading import metrics


def test_metrics_reimport_does_not_duplicate():
    metrics.reset_registry()
    mod = importlib.import_module("ai_trading.execution.engine")
    first = getattr(metrics.REGISTRY, "_names_to_collectors", {}).get("orders_submitted_total")
    assert first is not None
    mod = importlib.reload(mod)
    second = getattr(metrics.REGISTRY, "_names_to_collectors", {}).get("orders_submitted_total")
    assert second is first
