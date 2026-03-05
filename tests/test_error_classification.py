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
    assert info.reason_code.startswith("AUTH_BROKER_HALT")


def test_classify_data_auth_disables_provider() -> None:
    info = classify_exception(_HttpError(403, "forbidden"), dependency="data_primary")
    assert info.category is ErrorCategory.AUTH
    assert info.action is ErrorAction.DISABLE_PROVIDER
    assert info.reason_code.startswith("AUTH_PROVIDER_DISABLE")


def test_classify_auth_credentials_missing_reason() -> None:
    info = classify_exception(
        RuntimeError("missing credentials for broker"),
        dependency="broker_submit",
    )
    assert info.category is ErrorCategory.AUTH
    assert info.reason_code.endswith("CREDENTIALS_MISSING")


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


def test_auth_matcher_does_not_trigger_on_non_auth_words() -> None:
    info = classify_exception(RuntimeError("authoritative source mismatch"), dependency="data_primary")
    assert info.category is ErrorCategory.UNKNOWN
    assert info.action is ErrorAction.HALT_TRADING


def test_classify_broker_capacity_forbidden_as_order_reject() -> None:
    info = classify_exception(
        _HttpError(403, "insufficient buying power"),
        dependency="broker_submit",
        symbol="AAPL",
    )
    assert info.category is ErrorCategory.ORDER_REJECTED
    assert info.action is ErrorAction.SKIP_SYMBOL
    assert info.reason_code == "ORDER_REJECTED_SKIP"


def test_classify_broker_short_restriction_forbidden_as_order_reject() -> None:
    info = classify_exception(
        _HttpError(403, "shorting_not_permitted"),
        dependency="broker_submit",
        symbol="AAPL",
    )
    assert info.category is ErrorCategory.ORDER_REJECTED
    assert info.action is ErrorAction.SKIP_SYMBOL
    assert info.reason_code == "ORDER_REJECTED_SKIP"
