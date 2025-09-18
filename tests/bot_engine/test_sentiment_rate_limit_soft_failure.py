from __future__ import annotations

import math
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def stub_numpy(monkeypatch):
    """Provide a lightweight numpy stub so bot_engine imports succeed."""

    numpy_stub = types.SimpleNamespace(
        ndarray=list,
        array=lambda data, dtype=None: list(data),
        asarray=lambda data, dtype=float: list(data),
        diff=lambda arr: [arr[i + 1] - arr[i] for i in range(len(arr) - 1)],
        where=lambda cond, x, y: [
            xi if cond_i else yi for cond_i, xi, yi in zip(cond, x, y, strict=False)
        ],
        zeros_like=lambda arr: [0 for _ in arr],
        float64=float,
        nan=float("nan"),
        NaN=float("nan"),
        isfinite=lambda value: math.isfinite(value)
        if isinstance(value, (int, float))
        else False,
        random=types.SimpleNamespace(seed=lambda *_, **__: None),
    )
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    yield


@pytest.fixture(autouse=True)
def stub_portalocker(monkeypatch):
    """Stub portalocker to avoid optional dependency imports."""

    portalocker_stub = types.SimpleNamespace(
        LOCK_EX=1,
        lock=lambda *_, **__: None,
        unlock=lambda *_, **__: None,
    )
    monkeypatch.setitem(sys.modules, "portalocker", portalocker_stub)
    yield


@pytest.fixture(autouse=True)
def stub_bs4(monkeypatch):
    """Stub BeautifulSoup dependency."""

    class _BeautifulSoup:  # pragma: no cover - trivial stub
        def __init__(self, *_, **__):
            pass

    module = types.ModuleType("bs4")
    module.BeautifulSoup = _BeautifulSoup
    monkeypatch.setitem(sys.modules, "bs4", module)
    yield


@pytest.fixture(autouse=True)
def reset_sentiment_state(stub_numpy, stub_portalocker, stub_bs4):
    """Ensure sentiment circuit breaker starts from a clean slate."""

    from ai_trading.core import bot_engine as be

    be._SENTIMENT_CACHE.clear()
    be._SENTIMENT_CIRCUIT_BREAKER.update(
        {
            "failures": 0,
            "last_failure": 0,
            "state": "closed",
            "next_retry": 0,
            "opened_at": 0,
        }
    )
    yield
    # Tests expect a pristine circuit breaker for other cases as well.
    be._SENTIMENT_CACHE.clear()
    be._SENTIMENT_CIRCUIT_BREAKER.update(
        {
            "failures": 0,
            "last_failure": 0,
            "state": "closed",
            "next_retry": 0,
            "opened_at": 0,
        }
    )


def test_rate_limit_soft_failure_defers_provider_disable(monkeypatch):
    """Simulate repeated 429 responses and ensure escalation is deferred."""

    from ai_trading.core import bot_engine as be

    provider_calls: list[tuple[str, str, str | None]] = []

    def record_failure(provider: str, reason: str, error: str | None = None) -> None:
        provider_calls.append((provider, reason, error))

    monkeypatch.setattr(
        be.provider_monitor,
        "record_failure",
        record_failure,
        raising=False,
    )

    threshold = be.SENTIMENT_FAILURE_THRESHOLD

    for attempt in range(1, threshold):
        be._record_sentiment_failure("rate_limit")
        assert be._SENTIMENT_CIRCUIT_BREAKER["failures"] == attempt
        assert be._SENTIMENT_CIRCUIT_BREAKER["state"] == "closed"
        assert provider_calls == []

    be._record_sentiment_failure("rate_limit")
    assert be._SENTIMENT_CIRCUIT_BREAKER["failures"] == threshold
    assert be._SENTIMENT_CIRCUIT_BREAKER["state"] == "open"
    assert provider_calls == [("sentiment", "rate_limit", None)]
