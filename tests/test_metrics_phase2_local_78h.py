from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trading import metrics
from ai_trading.risk import metrics as risk_metrics


def test_metrics_facade_fallback_import_and_registry_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = Path(metrics.__file__)
    original_import = builtins.__import__

    def block_prometheus(name: str, *args, **kwargs):
        if name == "prometheus_client":
            raise ImportError("prometheus unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_prometheus)
    spec = importlib.util.spec_from_file_location("metrics_fallback_probe", module_path)
    assert spec is not None and spec.loader is not None
    fallback_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback_metrics)

    registry = fallback_metrics.reset_registry()
    counter = fallback_metrics.get_counter("orders_total", "orders")

    assert fallback_metrics.PROMETHEUS_AVAILABLE is False
    assert counter.labels("ignored") is counter
    counter.inc()
    counter.set(1)
    counter.observe(2)
    assert fallback_metrics.get_counter("orders_total", "orders") is counter
    assert fallback_metrics.get_registry() is registry

    class LockedRegistry:
        def __setattr__(self, _name: str, _value: object) -> None:
            raise AttributeError("locked")

    assert hasattr(fallback_metrics._ensure_names_map(None), "_names_to_collectors")
    assert hasattr(fallback_metrics._ensure_names_map(LockedRegistry()), "_names_to_collectors")

    class BareRegistry:
        pass

    calls: list[object] = []

    def hook(registry_obj: object) -> None:
        calls.append(registry_obj)

    fallback_metrics.register_reset_hook(hook)
    fallback_metrics.register_reset_hook(hook)
    fallback_metrics.reset_registry(BareRegistry())

    assert len(calls) == 1


def test_metrics_get_metric_stores_for_registry_without_register() -> None:
    class Metric:
        def __init__(self, name: str, _doc: str, *args, registry, **kwargs) -> None:
            self.name = name
            self.args = args
            self.registry = registry
            self.kwargs = kwargs

    registry = SimpleNamespace(_names_to_collectors={})

    first = metrics._get_metric(Metric, "latency_seconds", "latency", "label", registry=registry)
    second = metrics._get_metric(Metric, "latency_seconds", "latency", registry=registry)

    assert first is second
    assert registry._names_to_collectors["latency_seconds"] is first


def test_risk_metrics_error_branches_and_degenerate_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    calc = risk_metrics.RiskMetricsCalculator()
    analyzer = risk_metrics.DrawdownAnalyzer()

    assert calc.calculate_expected_shortfall([0.01] * 30, confidence_level=1.0) == 0.01
    assert calc.calculate_max_drawdown(["bad", object()]) == 0.0
    assert calc.calculate_max_drawdown([-1.5, 0.1]) == 0.0
    assert calc.calculate_calmar_ratio([float("nan"), "bad"]) == 0.0
    assert calc.calculate_risk_of_ruin([0.01, 0.02], ruin_threshold=5.0) < 1.0

    def raise_stats_error(*_args, **_kwargs):
        raise risk_metrics.statistics.StatisticsError("bad stats")

    monkeypatch.setattr(risk_metrics.statistics, "mean", raise_stats_error)

    assert calc.calculate_expected_shortfall([-0.1] * 30) == 0.0
    assert calc.calculate_sharpe_ratio([0.01, -0.02]) == 0.0
    assert calc.calculate_sortino_ratio([-0.01, -0.02]) == 0.0
    assert calc.calculate_calmar_ratio([0.01, -0.02]) == 0.0
    assert calc.calculate_risk_of_ruin([0.01, -0.02]) == 0.0
    assert calc.calculate_scorecard([object(), float("nan")]) == {
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
        "tail_loss_95": 0.0,
        "risk_of_ruin": 0.0,
    }
    assert analyzer.calculate_drawdowns([100.0, "bad"]) == {}
    assert analyzer.is_in_drawdown("bad", 100.0) == (False, 0.0)
    assert analyzer.calculate_recovery_time([100.0, "bad"], 0, 0) is None
