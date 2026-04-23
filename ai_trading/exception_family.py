"""Shared non-fatal exception families for defensive runtime fallbacks."""

from __future__ import annotations

AI_TRADING_FALLBACK_EXCEPTIONS: tuple[type[Exception], ...] = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ConnectionError,
    EOFError,
    ImportError,
    LookupError,
    NameError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

OPTIONAL_DEPENDENCY_EXCEPTIONS: tuple[type[Exception], ...] = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
)
