"""Fail-fast runtime contract helpers for paper/live execution."""
from __future__ import annotations

from typing import Any, Mapping

from ai_trading.config.management import get_env


def _execution_mode(execution_mode: str | None = None) -> str:
    if execution_mode:
        return str(execution_mode).strip().lower()
    return str(get_env("EXECUTION_MODE", "sim")).strip().lower()


def is_testing_mode() -> bool:
    """Return True when explicit test mode is active."""

    if bool(get_env("AI_TRADING_TESTING", "0", cast=bool)):
        return True
    if bool(get_env("PYTEST_RUNNING", "0", cast=bool)):
        return True
    return bool(get_env("TESTING", "0", cast=bool))


def _should_enforce(execution_mode: str | None = None) -> bool:
    mode = _execution_mode(execution_mode)
    return mode in {"paper", "live"} and not is_testing_mode()


def require_dependency(condition: bool, message: str, *, execution_mode: str | None = None) -> None:
    """Raise when a required dependency is missing in enforced modes."""

    if condition:
        return
    if _should_enforce(execution_mode):
        raise RuntimeError(message)


def _value_is_stub(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return bool(getattr(value, "_IS_STUB", False))


def require_no_stubs(ctx: Any, *, execution_mode: str | None = None) -> None:
    """Raise when known stub markers are reachable in paper/live execution."""

    if not _should_enforce(execution_mode):
        return

    violations: list[str] = []
    if isinstance(ctx, Mapping):
        for name, value in ctx.items():
            key = str(name)
            if key.endswith("_STUB") and _value_is_stub(value):
                violations.append(key)
            elif _value_is_stub(value):
                violations.append(key)
    else:
        for attr in ("_REQUESTS_STUB", "_HTTP_SESSION_STUB", "_IS_STUB"):
            try:
                if _value_is_stub(getattr(ctx, attr, False)):
                    violations.append(attr)
            except Exception:
                continue

    runtime = None
    if isinstance(ctx, Mapping):
        runtime = ctx.get("runtime")
    elif hasattr(ctx, "execution_engine") or hasattr(ctx, "exec_engine"):
        runtime = ctx

    if runtime is not None:
        for attr in ("execution_engine", "exec_engine"):
            engine = getattr(runtime, attr, None)
            if _value_is_stub(engine):
                violations.append(attr)

    if violations:
        ordered = ", ".join(sorted(set(violations)))
        raise RuntimeError(f"Runtime contract violation: stub dependency active ({ordered})")
