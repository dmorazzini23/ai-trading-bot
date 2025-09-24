"""Transient broker failures should retry before succeeding."""

from __future__ import annotations

from ai_trading.execution import live_trading as lt


def _engine_with_retry() -> lt.ExecutionEngine:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.retry_config = {"max_attempts": 3, "base_delay": 0.0, "max_delay": 0.0, "exponential_base": 1.0}
    engine.stats = {"retry_count": 0}
    engine.circuit_breaker = {"failure_count": 0, "is_open": False, "last_failure": None}
    engine._handle_nonretryable_api_error = lambda exc, *a, **k: None
    engine._handle_execution_failure = lambda exc: None
    return engine


def test_retry_until_success(monkeypatch):
    engine = _engine_with_retry()
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary")
        return "ok"

    result = engine._execute_with_retry(flaky)
    assert result == "ok"
    assert attempts["count"] == 3
    assert engine.stats["retry_count"] == 2


def test_nonretryable_error_stops_retry(monkeypatch):
    engine = _engine_with_retry()

    class StopError(lt.APIError):
        pass

    def fail():
        raise StopError("fatal")

    engine._handle_nonretryable_api_error = lambda exc, *a, **k: lt.NonRetryableBrokerError("fatal")

    try:
        engine._execute_with_retry(fail)
    except lt.NonRetryableBrokerError:
        pass
    else:  # pragma: no cover - safeguard
        raise AssertionError("Non-retryable error should propagate")

    assert engine.stats["retry_count"] == 0
