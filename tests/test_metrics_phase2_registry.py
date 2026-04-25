from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import ai_trading.metrics as metrics


class _BareRegistry:
    _names_to_collectors: dict[str, object]


class _ReadOnlyRegistry:
    @property
    def _names_to_collectors(self) -> dict[str, object]:
        raise AttributeError("read only")


class _Metric:
    def __init__(
        self,
        name: str,
        documentation: str,
        *_args: Any,
        registry: Any = None,
        **_kwargs: Any,
    ) -> None:
        self.name = name
        self.documentation = documentation
        self.registry = registry
        collectors = getattr(registry, "_names_to_collectors", None)
        if isinstance(collectors, dict):
            collectors[name] = self

    def inc(self, *_args: Any) -> None:
        return None

    def set(self, *_args: Any) -> None:
        return None

    def observe(self, *_args: Any) -> None:
        return None


def test_ensure_names_map_and_reset_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "_RESET_HOOKS", [])
    fallback = metrics._ensure_names_map(None)
    bare = _BareRegistry()
    readonly = metrics._ensure_names_map(_ReadOnlyRegistry())
    calls: list[Any] = []

    assert hasattr(fallback, "_names_to_collectors")
    assert metrics._ensure_names_map(bare) is bare
    assert bare._names_to_collectors == {}
    assert hasattr(readonly, "_names_to_collectors")

    metrics.register_reset_hook(lambda registry: calls.append(registry))
    metrics.register_reset_hook(lambda registry: calls.append(registry))
    registry = metrics.reset_registry(SimpleNamespace(_names_to_collectors={}))

    assert metrics.get_registry() is registry
    assert calls[-1] is registry


def test_metric_creation_reuses_existing_and_handles_registry_without_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SimpleNamespace(_names_to_collectors={})

    first = metrics._get_metric(_Metric, "orders_total", "Orders", registry=registry)
    second = metrics._get_metric(_Metric, "orders_total", "Orders", registry=registry)

    assert first is second

    no_register = SimpleNamespace()
    metric = metrics._get_metric(_Metric, "fills_total", "Fills", registry=no_register)
    assert no_register._names_to_collectors["fills_total"] is metric


def test_public_metric_helpers_and_compute_basic_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = SimpleNamespace(_names_to_collectors={})
    monkeypatch.setattr(metrics, "Counter", _Metric)
    monkeypatch.setattr(metrics, "Gauge", _Metric)
    monkeypatch.setattr(metrics, "Histogram", _Metric)
    monkeypatch.setattr(metrics, "REGISTRY", registry)

    assert metrics.get_counter("counter_total", "Counter").name == "counter_total"
    assert metrics.get_gauge("gauge_value", "Gauge").name == "gauge_value"
    assert metrics.get_histogram("latency_seconds", "Latency").name == "latency_seconds"
    assert metrics.compute_basic_metrics([]) == {"sharpe": 0.0, "max_drawdown": 0.0}


def test_lazy_metric_math_wrappers_cache_imported_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics._calculate_atr = None
    metrics._safe_divide = None

    import pandas as pd

    frame = pd.DataFrame(
        {
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.0, 11.0, 12.0],
        }
    )

    assert len(metrics.calculate_atr(frame, period=2)) == 3
    assert metrics.safe_divide(6, 3) == 2
    assert metrics._calculate_atr is not None
    assert metrics._safe_divide is not None
