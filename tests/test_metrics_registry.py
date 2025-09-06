import importlib

from ai_trading import metrics
from ai_trading.metrics import registry


def test_metrics_reimport_does_not_duplicate():
    metrics.reset_registry()
    mod = importlib.import_module("ai_trading.execution.engine")
    first = getattr(metrics.REGISTRY, "_names_to_collectors", {}).get("orders_submitted_total")
    assert first is not None
    mod = importlib.reload(mod)
    second = getattr(metrics.REGISTRY, "_names_to_collectors", {}).get("orders_submitted_total")
    assert second is first


def test_registry_instance_persists_across_reloads():
    registry.reset_registry()
    first = registry.get_registry()
    mod = importlib.reload(registry)
    second = mod.get_registry()
    assert first is second
