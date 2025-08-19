from __future__ import annotations

PROMETHEUS_AVAILABLE = False
REGISTRY = None
CollectorRegistry = None
Gauge = None
Counter = None
Histogram = None
Summary = None
start_http_server = None

try:  # new-style guarded import
    # Try common symbols from prometheus_client
    from prometheus_client import (
        REGISTRY as _REGISTRY,
    )
    from prometheus_client import (
        CollectorRegistry as _CollectorRegistry,
    )
    from prometheus_client import (
        Counter as _Counter,
    )
    from prometheus_client import (
        Gauge as _Gauge,
    )
    from prometheus_client import (
        Histogram as _Histogram,
    )
    from prometheus_client import (
        Summary as _Summary,
    )
    from prometheus_client import (
        start_http_server as _start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
    REGISTRY = _REGISTRY
    CollectorRegistry = _CollectorRegistry
    Gauge = _Gauge
    Counter = _Counter
    Histogram = _Histogram
    Summary = _Summary
    start_http_server = _start_http_server
except Exception:  # noqa: BLE001
    # Minimal no-op fallbacks so imports & tests never crash if the pkg is missing
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

    class CollectorRegistry:  # type: ignore[override]
        def __init__(self, *_, **__):
            pass

    Gauge = Counter = Histogram = Summary = _NoopMetric  # type: ignore
    start_http_server = _noop_start_http_server

# AI-AGENT-REF: expose basic metrics helpers under canonical package
from ai_trading.monitoring.metrics import safe_divide, calculate_atr  # noqa: E402


def compute_basic_metrics(data):
    """Return minimal metrics dict."""  # AI-AGENT-REF
    if hasattr(data, "empty") and data.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0}
    return {"sharpe": 0.0, "max_drawdown": 0.0}


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "REGISTRY",
    "CollectorRegistry",
    "Gauge",
    "Counter",
    "Histogram",
    "Summary",
    "start_http_server",
    "safe_divide",
    "calculate_atr",
    "compute_basic_metrics",
]
