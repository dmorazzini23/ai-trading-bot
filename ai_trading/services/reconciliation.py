from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from collections.abc import Mapping
from json import JSONDecodeError
from typing import Any

from ai_trading.contracts import position_snapshot_from_position


def _broker_positions_reader(api: Any) -> Any | None:
    for name in ("get_all_positions", "list_positions"):
        reader = getattr(api, name, None)
        if callable(reader):
            return reader
    return None


class ReconciliationService:
    """Canonical reconciliation service facade over runtime cleanup and error gates."""

    boundary_type = "facade"
    canonical_runtime_owner = (
        "ai_trading.services.reconciliation.ReconciliationService.reconcile_position_targets",
        "ai_trading.services.reconciliation.ReconciliationService.require_success",
    )

    def reconcile_position_targets(
        self,
        ctx: Any,
        *,
        logger: Any,
        targets_lock: Any,
        warned: bool = False,
    ) -> bool:
        """Prune stale stop/take targets against live broker positions."""

        if not getattr(ctx, "api", None):
            if not warned:
                logger.warning("Skipping reconciliation: no broker client")
                return True
            return warned
        positions_reader = _broker_positions_reader(ctx.api)
        if positions_reader is None:
            if not warned:
                logger.warning(
                    "Skipping reconciliation: broker client missing positions method"
                )
                return True
            return warned
        try:
            live_position_snapshots = {
                snapshot.symbol: snapshot
                for snapshot in (
                    position_snapshot_from_position(pos, provider="alpaca")
                    for pos in positions_reader()
                )
                if snapshot is not None
            }
            live_positions = {
                symbol: int(snapshot.qty)
                for symbol, snapshot in live_position_snapshots.items()
            }
            try:
                setattr(
                    ctx,
                    "_reconciliation_position_snapshots",
                    {
                        symbol: snapshot.to_dict()
                        for symbol, snapshot in live_position_snapshots.items()
                    },
                )
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                logger.debug("RECONCILIATION_POSITION_SNAPSHOT_SET_FAILED", exc_info=True)
            with targets_lock:
                symbols_with_targets = list(getattr(ctx, "stop_targets", {}).keys()) + list(
                    getattr(ctx, "take_profit_targets", {}).keys()
                )
                for symbol in symbols_with_targets:
                    if symbol not in live_positions or live_positions[symbol] == 0:
                        ctx.stop_targets.pop(symbol, None)
                        ctx.take_profit_targets.pop(symbol, None)
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
            logger.exception("reconcile_positions failed", exc_info=exc)
            try:
                setattr(ctx, "_reconciliation_error", str(exc))
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                logger.debug("RECONCILIATION_ERROR_SET_FAILED", exc_info=True)
            raise RuntimeError("position target reconciliation failed") from exc
        return warned

    @staticmethod
    def require_success(
        context: Mapping[str, Any] | None,
        *,
        scope: str,
        logger: Any,
    ) -> None:
        """Raise when reconciliation context contains explicit error fields."""

        payload = dict(context or {})
        errors = {
            str(key): str(value)
            for key, value in payload.items()
            if str(key).endswith("_error") and str(value or "").strip()
        }
        if not errors:
            return
        logger.error(
            "RECONCILIATION_SERVICE_FAILED",
            extra={"scope": scope, "errors": dict(errors)},
        )
        raise RuntimeError(f"{scope} reconciliation failed")


def reconcile_position_targets(
    ctx: Any,
    *,
    logger: Any,
    targets_lock: Any,
    warned: bool = False,
) -> bool:
    return ReconciliationService().reconcile_position_targets(
        ctx,
        logger=logger,
        targets_lock=targets_lock,
        warned=warned,
    )


def require_success(
    context: Mapping[str, Any] | None,
    *,
    scope: str,
    logger: Any,
) -> None:
    ReconciliationService.require_success(context, scope=scope, logger=logger)


__all__ = ["ReconciliationService", "reconcile_position_targets", "require_success"]
