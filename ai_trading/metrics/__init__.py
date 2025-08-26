from __future__ import annotations
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
except (KeyError, ValueError, TypeError):

    class _NoopRegistry:

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
            pass
    Gauge = Counter = Histogram = Summary = _NoopMetric
    start_http_server = _noop_start_http_server


_calculate_atr = None
_safe_divide = None


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
]
