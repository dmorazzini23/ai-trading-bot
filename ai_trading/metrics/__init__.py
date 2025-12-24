from __future__ import annotations

from collections.abc import Callable


class _FallbackRegistry:
    """Minimal registry used when the provided registry cannot be mutated."""

    def __init__(self) -> None:
        self._names_to_collectors: dict[str, object] = {}

    def register(self, *_, **__) -> None:
        pass

    def unregister(self, *_, **__) -> None:
        pass

PROMETHEUS_AVAILABLE = False
REGISTRY = None
CollectorRegistry = None
Gauge = None
Counter = None
Histogram = None
Summary = None
start_http_server = None
try:
    from prometheus_client import REGISTRY as _REGISTRY
    from prometheus_client import CollectorRegistry as _CollectorRegistry
    from prometheus_client import Counter as _Counter
    from prometheus_client import Gauge as _Gauge
    from prometheus_client import Histogram as _Histogram
    from prometheus_client import Summary as _Summary
    from prometheus_client import start_http_server as _start_http_server
    PROMETHEUS_AVAILABLE = True
    REGISTRY = _REGISTRY
    CollectorRegistry = _CollectorRegistry
    Gauge = _Gauge
    Counter = _Counter
    Histogram = _Histogram
    Summary = _Summary
    start_http_server = _start_http_server
except (ImportError, KeyError, ValueError, TypeError):

    class _NoopRegistry:

        def __init__(self):
            self._names_to_collectors: dict[str, object] = {}

        def register(self, *_, **__):
            pass

        def unregister(self, *_, **__):
            pass

    class _NoopMetric:

        def __init__(self, *_, **__):
            pass

        def labels(self, *_, **__):
            return self

        def set(self, *_):
            pass

        def inc(self, *_):
            pass

        def observe(self, *_):
            pass

    def _noop_start_http_server(*_, **__):
        pass
    PROMETHEUS_AVAILABLE = False
    REGISTRY = _NoopRegistry()

    class CollectorRegistry:

        def __init__(self, *_, **__):
            self._names_to_collectors: dict[str, object] = {}
    Gauge = Counter = Histogram = Summary = _NoopMetric
    start_http_server = _noop_start_http_server


def _ensure_names_map(registry):
    """Ensure ``registry`` exposes a ``_names_to_collectors`` mapping."""

    if registry is None:
        return _FallbackRegistry()
    if not hasattr(registry, "_names_to_collectors"):
        try:
            setattr(registry, "_names_to_collectors", {})
        except (AttributeError, TypeError):
            return _FallbackRegistry()
    return registry


REGISTRY = _ensure_names_map(REGISTRY)


_calculate_atr = None
_safe_divide = None

_RESET_HOOKS: list[Callable[[CollectorRegistry], None]] = []


def calculate_atr(*args, **kwargs):
    """Lazy import wrapper for ``monitoring.metrics.calculate_atr``."""
    global _calculate_atr
    if _calculate_atr is None:  # pragma: no cover - simple cache
        from ai_trading.monitoring.metrics import (
            calculate_atr as _calculate_atr_fn,
        )
        _calculate_atr = _calculate_atr_fn
    return _calculate_atr(*args, **kwargs)


def safe_divide(*args, **kwargs):
    """Lazy import wrapper for ``monitoring.metrics.safe_divide``."""
    global _safe_divide
    if _safe_divide is None:  # pragma: no cover - simple cache
        from ai_trading.monitoring.metrics import (
            safe_divide as _safe_divide_fn,
        )
        _safe_divide = _safe_divide_fn
    return _safe_divide(*args, **kwargs)

def compute_basic_metrics(data):
    """Return minimal metrics dict."""
    if hasattr(data, 'empty') and data.empty:
        return {'sharpe': 0.0, 'max_drawdown': 0.0}
    return {'sharpe': 0.0, 'max_drawdown': 0.0}


def reset_registry(registry: CollectorRegistry | None = None) -> CollectorRegistry:
    """Replace and return the active metrics registry.

    Tests may call this helper to ensure a clean registry between runs.
    When ``registry`` is ``None`` a new :class:`CollectorRegistry` is created.
    """
    global REGISTRY
    if registry is not None:
        REGISTRY = _ensure_names_map(registry)
    else:
        REGISTRY = _ensure_names_map(CollectorRegistry())  # type: ignore[call-arg]
    for hook in list(_RESET_HOOKS):
        hook(REGISTRY)
    return REGISTRY


def get_registry() -> CollectorRegistry:
    """Return the active metrics registry."""

    return REGISTRY  # type: ignore[return-value]


def register_reset_hook(callback: Callable[[CollectorRegistry], None]) -> None:
    """Register ``callback`` to run whenever the metrics registry resets."""

    if callback not in _RESET_HOOKS:
        _RESET_HOOKS.append(callback)


def _get_metric(metric_cls, name: str, documentation: str, *args, **kwargs):
    """Return an existing metric or create a new one in ``REGISTRY``.

    ``prometheus_client`` raises ``ValueError`` if a metric is registered more
    than once.  Older versions expose ``_names_to_collectors`` on the registry
    which we can query directly without requiring a sample to have been
    recorded.  This approach avoids duplicate registration even when the metric
    has not been used yet.
    """
    registry = kwargs.pop("registry", REGISTRY)
    collectors = getattr(registry, "_names_to_collectors", None)
    existing = collectors.get(name) if collectors and name else None
    if existing is not None:
        return existing
    metric = metric_cls(name, documentation, *args, registry=registry, **kwargs)
    if not hasattr(registry, "register"):
        if collectors is None:
            collectors = {}
            setattr(registry, "_names_to_collectors", collectors)
        if name:
            collectors[name] = metric
    return metric


def get_counter(name: str, documentation: str, *args, **kwargs):
    """Return a :class:`Counter` avoiding duplicate registration."""
    return _get_metric(Counter, name, documentation, *args, **kwargs)


def get_gauge(name: str, documentation: str, *args, **kwargs):
    """Return a :class:`Gauge` avoiding duplicate registration."""
    return _get_metric(Gauge, name, documentation, *args, **kwargs)


def get_histogram(name: str, documentation: str, *args, **kwargs):
    """Return a :class:`Histogram` avoiding duplicate registration."""
    return _get_metric(Histogram, name, documentation, *args, **kwargs)
__all__ = [
    'PROMETHEUS_AVAILABLE',
    'REGISTRY',
    'CollectorRegistry',
    'Gauge',
    'Counter',
    'Histogram',
    'Summary',
    'start_http_server',
    'safe_divide',
    'calculate_atr',
    'compute_basic_metrics',
    'reset_registry',
    'register_reset_hook',
    'get_registry',
    'get_counter',
    'get_gauge',
    'get_histogram',
]
