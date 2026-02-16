from __future__ import annotations

import pytest

from ai_trading.core.dependency_breakers import DependencyBreakers
from ai_trading.core.errors import classify_exception
from ai_trading.core.retry import retry_idempotent


def test_retry_idempotent_retries_transient_errors() -> None:
    breakers = DependencyBreakers()
    attempts = {"count": 0}

    def _fn() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary timeout")
        return "ok"

    result = retry_idempotent(
        _fn,
        dep="data_primary",
        breakers=breakers,
        classify_exception=classify_exception,
        max_attempts=3,
        max_total_seconds=10.0,
        base_delay=0.0,
        jitter=0.0,
        context={"symbol": "AAPL"},
    )
    assert result == "ok"
    assert attempts["count"] == 3


def test_retry_idempotent_does_not_retry_non_retryable() -> None:
    breakers = DependencyBreakers()
    attempts = {"count": 0}

    def _fn() -> None:
        attempts["count"] += 1
        raise TypeError("programming bug")

    with pytest.raises(TypeError):
        retry_idempotent(
            _fn,
            dep="data_primary",
            breakers=breakers,
            classify_exception=classify_exception,
            max_attempts=5,
            max_total_seconds=10.0,
            base_delay=0.0,
            jitter=0.0,
        )
    assert attempts["count"] == 1
