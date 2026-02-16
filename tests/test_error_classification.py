from __future__ import annotations

from ai_trading.core.errors import (
    ErrorAction,
    ErrorCategory,
    classify_exception,
)


class _HttpError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_classify_auth_halts() -> None:
    info = classify_exception(_HttpError(401, "unauthorized"), dependency="broker_submit")
    assert info.category is ErrorCategory.AUTH
    assert info.action is ErrorAction.HALT_TRADING
    assert info.reason_code == "AUTH_HALT"


def test_classify_rate_limit_retries() -> None:
    info = classify_exception(_HttpError(429, "too many requests"), dependency="data_primary")
    assert info.category is ErrorCategory.RATE_LIMIT
    assert info.action is ErrorAction.RETRY
    assert info.retryable is True


def test_classify_programming_halts() -> None:
    info = classify_exception(TypeError("bad key"), dependency="data_primary", symbol="AAPL")
    assert info.category is ErrorCategory.PROGRAMMING_ERROR
    assert info.action is ErrorAction.HALT_TRADING


def test_classify_invariant_halts() -> None:
    info = classify_exception(ValueError("nan encountered in netting"), dependency="broker_positions")
    assert info.category is ErrorCategory.INVARIANT_VIOLATION
    assert info.action is ErrorAction.HALT_TRADING
