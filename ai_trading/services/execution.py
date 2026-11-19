from __future__ import annotations

from typing import Any

from ai_trading.config.management import get_env, is_test_runtime
from ai_trading.core.runtime_contract import (
    UnknownExecutionModeError,
    normalize_execution_mode,
)


class NonNettingLiveExecutionBlockedError(RuntimeError):
    """Raised when non-netting execution is attempted in live mode."""


def _execution_mode(ctx: Any) -> str:
    ctx_mode = str(getattr(ctx, "execution_mode", "") or "").strip().lower()
    if ctx_mode:
        return normalize_execution_mode(ctx_mode)
    return normalize_execution_mode(get_env("EXECUTION_MODE", "paper") or "paper")


def _non_netting_live_execution_allowed() -> bool:
    return bool(
        is_test_runtime()
        and get_env("AI_TRADING_ENABLE_NON_NETTING_LIVE_EXECUTION", False, cast=bool)
    )


class ExecutionService:
    """Canonical execution service facade for operator/runtime callers."""

    boundary_type = "facade"
    canonical_runtime_owner = (
        "ai_trading.core.submit_runtime.submit_order_runtime",
    )

    @staticmethod
    def _require_supported_mode(*, ctx: Any, operation: str) -> None:
        mode = _execution_mode(ctx)
        if mode == "live" and not _non_netting_live_execution_allowed():
            raise NonNettingLiveExecutionBlockedError(
                f"{operation} is blocked for live non-netting execution. "
                "Use the canonical OMS/netting live path instead."
            )

    def submit_order(
        self,
        ctx: Any,
        symbol: str,
        qty: int,
        side: str,
        *,
        price: float | None = None,
        **exec_kwargs: Any,
    ) -> Any | None:
        """Submit through the shared non-netting runtime only outside blocked live mode."""

        self._require_supported_mode(ctx=ctx, operation="submit_order")
        from ai_trading.core.submit_runtime import submit_order_runtime

        return submit_order_runtime(
            ctx,
            symbol,
            qty,
            side,
            price=price,
            **exec_kwargs,
        )


def submit_order(
    ctx: Any,
    symbol: str,
    qty: int,
    side: str,
    *,
    price: float | None = None,
    **exec_kwargs: Any,
) -> Any | None:
    return ExecutionService().submit_order(
        ctx,
        symbol,
        qty,
        side,
        price=price,
        **exec_kwargs,
    )


__all__ = [
    "ExecutionService",
    "NonNettingLiveExecutionBlockedError",
    "UnknownExecutionModeError",
    "normalize_execution_mode",
    "submit_order",
]
