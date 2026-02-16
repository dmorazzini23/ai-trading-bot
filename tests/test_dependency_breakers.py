from __future__ import annotations

from ai_trading.core.dependency_breakers import DependencyBreakers
from ai_trading.core.errors import ErrorAction, ErrorCategory, ErrorInfo, ErrorScope


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _error_info(dep: str) -> ErrorInfo:
    return ErrorInfo(
        category=ErrorCategory.TRANSIENT_NETWORK,
        scope=ErrorScope.PROVIDER,
        action=ErrorAction.RETRY,
        retryable=True,
        dependency=dep,
        reason_code="NET_RETRY",
        details={},
    )


def test_breaker_opens_after_three_failures_in_60s(monkeypatch) -> None:
    clock = _Clock()
    breakers = DependencyBreakers()
    monkeypatch.setattr(breakers, "_monotonic", clock)

    dep = "data_primary"
    for timestamp in (0.0, 10.0, 20.0):
        clock.value = timestamp
        breakers.record_failure(dep, _error_info(dep))

    clock.value = 30.0
    assert breakers.allow(dep) is False
    assert breakers.open_reason(dep) == "CIRCUIT_OPEN_data_primary"

    clock.value = 90.1
    assert breakers.allow(dep) is True


def test_breaker_clears_after_success(monkeypatch) -> None:
    clock = _Clock()
    breakers = DependencyBreakers()
    monkeypatch.setattr(breakers, "_monotonic", clock)

    dep = "broker_positions"
    for timestamp in (0.0, 10.0, 20.0):
        clock.value = timestamp
        breakers.record_failure(dep, _error_info(dep))
    assert breakers.allow(dep) is False

    breakers.record_success(dep)
    assert breakers.allow(dep) is True
