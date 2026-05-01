from __future__ import annotations

from json import JSONDecodeError
from typing import Any

from ai_trading.config.management import get_env, is_test_runtime

_NON_ACCEPTED_ORDER_STATUSES = {"rejected", "canceled", "cancelled", "expired", "done_for_day"}


class NonNettingLiveExecutionBlockedError(RuntimeError):
    """Raised when non-netting execution is attempted in live mode."""


def _order_status_token(order: Any) -> str:
    status = getattr(order, "status", None)
    status_value = getattr(status, "value", status)
    return str(status_value or "").strip().lower()


def _execution_mode(ctx: Any) -> str:
    ctx_mode = str(getattr(ctx, "execution_mode", "") or "").strip().lower()
    if ctx_mode:
        return ctx_mode
    return str(get_env("EXECUTION_MODE", "paper") or "paper").strip().lower()


def _non_netting_live_execution_allowed() -> bool:
    return bool(
        is_test_runtime()
        and get_env("AI_TRADING_ENABLE_NON_NETTING_LIVE_EXECUTION", False, cast=bool)
    )


def _market_order_request(symbol: str, qty: int, side: str) -> Any:
    from types import SimpleNamespace

    from ai_trading.alpaca_api import MarketOrderRequest

    if MarketOrderRequest is None:
        return SimpleNamespace(
            symbol=str(symbol),
            qty=str(qty),
            side=str(side),
            time_in_force="day",
        )
    return MarketOrderRequest(
        symbol=str(symbol),
        qty=str(qty),
        side=str(side),
        time_in_force="day",
    )


class ExecutionService:
    """Canonical execution service facade for operator/runtime callers."""

    boundary_type = "facade"
    canonical_runtime_owner = (
        "ai_trading.core.submit_runtime.submit_order_runtime",
        "ai_trading.core.trade_cycle.execute_trade_logic",
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

    def execute_trade_cycle(
        self,
        ctx: Any,
        state: Any,
        symbol: str,
        balance: float,
        model: Any,
        regime_ok: bool,
        *,
        price_df: Any = None,
        now_provider: Any = None,
    ) -> bool:
        """Run the shared non-netting trade cycle only outside blocked live mode."""

        self._require_supported_mode(ctx=ctx, operation="trade_logic")
        from ai_trading.core.trade_cycle import execute_trade_logic

        return bool(
            execute_trade_logic(
                ctx,
                state,
                symbol,
                balance,
                model,
                regime_ok,
                price_df=price_df,
                now_provider=now_provider,
            )
        )

    def execute_signal_orders(
        self,
        ctx: Any,
        signals: Any,
        *,
        logger: Any,
    ) -> list[tuple[str, str]]:
        """Submit simple directional test/runtime orders through the attached broker client."""

        self._require_supported_mode(ctx=ctx, operation="execute_signal_orders")
        orders: list[tuple[str, str]] = []
        items = getattr(signals, "items", None)
        if not callable(items):
            return orders
        for symbol, sig in items():
            if sig == 0:
                continue
            side = "buy" if sig > 0 else "sell"
            api = getattr(ctx, "api", None)
            if api is not None and hasattr(api, "submit_order"):
                try:
                    order = api.submit_order(order_data=_market_order_request(str(symbol), 1, side))
                    status_token = _order_status_token(order)
                    if status_token in _NON_ACCEPTED_ORDER_STATUSES:
                        logger.error(
                            "Broker did not accept test order for %s %s: status=%s",
                            symbol,
                            side,
                            status_token,
                        )
                        continue
                    orders.append((str(symbol), side))
                except (
                    FileNotFoundError,
                    PermissionError,
                    IsADirectoryError,
                    JSONDecodeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                ) as exc:
                    logger.error(
                        "Failed to submit test order for %s %s: %s",
                        symbol,
                        side,
                        exc,
                    )
        return orders


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


def execute_trade_cycle(
    ctx: Any,
    state: Any,
    symbol: str,
    balance: float,
    model: Any,
    regime_ok: bool,
    *,
    price_df: Any = None,
    now_provider: Any = None,
) -> bool:
    return ExecutionService().execute_trade_cycle(
        ctx,
        state,
        symbol,
        balance,
        model,
        regime_ok,
        price_df=price_df,
        now_provider=now_provider,
    )


def execute_signal_orders(
    ctx: Any,
    signals: Any,
    *,
    logger: Any,
) -> list[tuple[str, str]]:
    return ExecutionService().execute_signal_orders(ctx, signals, logger=logger)


__all__ = [
    "ExecutionService",
    "NonNettingLiveExecutionBlockedError",
    "execute_signal_orders",
    "execute_trade_cycle",
    "submit_order",
]
