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




