"""Centralized exception taxonomy for trading runtime decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    TRANSIENT_NETWORK = "TRANSIENT_NETWORK"
    RATE_LIMIT = "RATE_LIMIT"
    AUTH = "AUTH"
    BAD_DATA = "BAD_DATA"
    PROVIDER_SCHEMA = "PROVIDER_SCHEMA"
    ORDER_REJECTED = "ORDER_REJECTED"
    BROKER_STATE = "BROKER_STATE"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    PROGRAMMING_ERROR = "PROGRAMMING_ERROR"
    UNKNOWN = "UNKNOWN"


class ErrorScope(str, Enum):
    SYMBOL = "SYMBOL"
    PROVIDER = "PROVIDER"
    GLOBAL = "GLOBAL"


class ErrorAction(str, Enum):
    RETRY = "RETRY"
    SKIP_SYMBOL = "SKIP_SYMBOL"
    DISABLE_PROVIDER = "DISABLE_PROVIDER"
    HALT_TRADING = "HALT_TRADING"


@dataclass(slots=True)
class ErrorInfo:
    category: ErrorCategory
    scope: ErrorScope
    action: ErrorAction
    retryable: bool
    dependency: str
    reason_code: str
    details: dict[str, Any] = field(default_factory=dict)


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        raw = getattr(exc, attr, None)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and raw.isdigit():
            return int(raw)
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _has_text(exc: Exception, token: str) -> bool:
    return token.lower() in str(exc).lower()


def _is_transient_network(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = exc.__class__.__name__.lower()
    return "timeout" in name or "connection" in name


def _is_auth(exc: Exception) -> bool:
    code = _status_code(exc)
    if code in {401, 403}:
        return True
    return any(_has_text(exc, token) for token in ("unauthorized", "forbidden", "auth"))


def _is_rate_limit(exc: Exception) -> bool:
    if _status_code(exc) == 429:
        return True
    return any(_has_text(exc, token) for token in ("rate limit", "too many requests", "throttled"))


def _is_invariant_violation(exc: Exception) -> bool:
    if isinstance(exc, FloatingPointError):
        return True
    lowered = str(exc).lower()
    return any(token in lowered for token in ("nan", "inf", "infinity", "negative price"))


def _is_provider_schema(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return any(token in lowered for token in ("schema", "missing field", "missing column", "unexpected payload"))


def _is_bad_data(exc: Exception) -> bool:
    if isinstance(exc, ValueError):
        return True
    lowered = str(exc).lower()
    return any(token in lowered for token in ("bad data", "invalid data", "validation"))


def _is_order_rejected(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "reject" in lowered or "insufficient" in lowered


def _is_broker_state(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return any(token in lowered for token in ("position", "order state", "account state"))


def _derive_scope(dependency: str, symbol: str | None) -> ErrorScope:
    if symbol:
        return ErrorScope.SYMBOL
    dep = (dependency or "").lower()
    if dep.startswith("data_") or dep.startswith("quotes_") or dep.startswith("broker_"):
        return ErrorScope.PROVIDER
    return ErrorScope.GLOBAL


def classify_exception(
    exc: Exception,
    *,
    dependency: str,
    symbol: str | None = None,
    sleeve: str | None = None,
) -> ErrorInfo:
    """Classify an exception into deterministic handling instructions."""

    scope = _derive_scope(dependency, symbol)
    details: dict[str, Any] = {
        "exception_type": exc.__class__.__name__,
        "message": str(exc),
    }
    if symbol:
        details["symbol"] = symbol
    if sleeve:
        details["sleeve"] = sleeve
    status_code = _status_code(exc)
    if status_code is not None:
        details["status_code"] = status_code

    if _is_auth(exc):
        return ErrorInfo(
            category=ErrorCategory.AUTH,
            scope=scope,
            action=ErrorAction.HALT_TRADING,
            retryable=False,
            dependency=dependency,
            reason_code="AUTH_HALT",
            details=details,
        )
    if _is_rate_limit(exc):
        return ErrorInfo(
            category=ErrorCategory.RATE_LIMIT,
            scope=scope,
            action=ErrorAction.RETRY,
            retryable=True,
            dependency=dependency,
            reason_code="RATE_LIMIT_RETRY",
            details=details,
        )
    if _is_transient_network(exc):
        return ErrorInfo(
            category=ErrorCategory.TRANSIENT_NETWORK,
            scope=scope,
            action=ErrorAction.RETRY,
            retryable=True,
            dependency=dependency,
            reason_code="NET_RETRY",
            details=details,
        )
    if _is_invariant_violation(exc):
        return ErrorInfo(
            category=ErrorCategory.INVARIANT_VIOLATION,
            scope=scope,
            action=ErrorAction.HALT_TRADING,
            retryable=False,
            dependency=dependency,
            reason_code="INVARIANT_HALT",
            details=details,
        )
    if isinstance(exc, (TypeError, KeyError)):
        return ErrorInfo(
            category=ErrorCategory.PROGRAMMING_ERROR,
            scope=scope,
            action=ErrorAction.HALT_TRADING,
            retryable=False,
            dependency=dependency,
            reason_code="PROGRAMMING_HALT",
            details=details,
        )
    if _is_provider_schema(exc):
        return ErrorInfo(
            category=ErrorCategory.PROVIDER_SCHEMA,
            scope=scope,
            action=ErrorAction.DISABLE_PROVIDER,
            retryable=False,
            dependency=dependency,
            reason_code="PROVIDER_SCHEMA_DISABLE",
            details=details,
        )
    if _is_bad_data(exc):
        return ErrorInfo(
            category=ErrorCategory.BAD_DATA,
            scope=scope,
            action=ErrorAction.SKIP_SYMBOL,
            retryable=False,
            dependency=dependency,
            reason_code="BAD_DATA_SKIP",
            details=details,
        )
    if _is_order_rejected(exc):
        return ErrorInfo(
            category=ErrorCategory.ORDER_REJECTED,
            scope=scope,
            action=ErrorAction.SKIP_SYMBOL,
            retryable=False,
            dependency=dependency,
            reason_code="ORDER_REJECTED_SKIP",
            details=details,
        )
    if _is_broker_state(exc):
        return ErrorInfo(
            category=ErrorCategory.BROKER_STATE,
            scope=scope,
            action=ErrorAction.RETRY,
            retryable=True,
            dependency=dependency,
            reason_code="BROKER_STATE_RETRY",
            details=details,
        )
    return ErrorInfo(
        category=ErrorCategory.UNKNOWN,
        scope=scope,
        action=ErrorAction.HALT_TRADING,
        retryable=False,
        dependency=dependency,
        reason_code="UNKNOWN_HALT",
        details=details,
    )
