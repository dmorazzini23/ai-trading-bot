"""Governance and reconciliation helpers extracted from ``bot_engine.py``."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_trading.config.runtime import TradingConfig


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _normalize_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _normalize_positions(raw_positions: Any) -> dict[str, float]:
    positions: dict[str, float] = {}
    if not isinstance(raw_positions, Mapping):
        return positions
    for raw_symbol, raw_value in dict(raw_positions).items():
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            continue
        try:
            positions[symbol] = float(raw_value or 0.0)
        except (TypeError, ValueError):
            positions[symbol] = 0.0
    return positions


@dataclass(slots=True)
class NettingGovernanceSnapshot:
    now: datetime
    market_open_now: bool


def run_netting_cycle_governance(state: Any) -> NettingGovernanceSnapshot:
    """Apply governance and learning schedules at the start of a netting cycle."""

    be = _bot_engine()
    now = datetime.now(UTC)
    market_open_now = False
    step = "market_is_open"
    try:
        market_open_now = bool(be.market_is_open(now))
        step = "post_trade_learning_update"
        be._run_post_trade_learning_update(
            state,
            now=now,
            market_open_now=market_open_now,
        )
        step = "tca_cost_calibration"
        be._run_tca_cost_calibration(
            state,
            now=now,
            market_open_now=market_open_now,
        )
        step = "policy_ablation_rollback"
        be._run_policy_ablation_rollback(
            state,
            now=now,
            market_open_now=market_open_now,
        )
        step = "replay_governance"
        be._run_replay_governance(state, now=now, market_open_now=market_open_now)
        step = "walk_forward_governance"
        be._run_walk_forward_governance(
            state,
            now=now,
            market_open_now=market_open_now,
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        state.halt_trading = True
        state.halt_reason = str(exc) or "GOVERNANCE_GUARD_FAILED"
        be.logger.error(
            "NETTING_GOVERNANCE_GUARD_FAILED",
            extra={
                "error_type": exc.__class__.__name__,
                "detail": str(exc),
                "step": step,
            },
        )
    return NettingGovernanceSnapshot(now=now, market_open_now=market_open_now)


def run_reconciliation_if_due(
    state: Any,
    runtime: Any,
    cfg: TradingConfig,
    now: datetime,
) -> bool:
    """Reconcile internal and broker positions on the configured schedule."""

    be = _bot_engine()
    if not bool(getattr(cfg, "recon_enabled", False)):
        return True
    interval = int(getattr(cfg, "recon_interval_seconds", 300))
    last_ts = _normalize_timestamp(getattr(state, "last_recon_ts", None))
    if last_ts is not None and (now - last_ts).total_seconds() < interval:
        return not bool(getattr(state, "recon_halt", False))

    breakers = be._dependency_breakers(state)
    if not breakers.allow("broker_positions"):
        state.recon_halt = True
        state.halt_reason = breakers.open_reason("broker_positions") or "CIRCUIT_OPEN_broker_positions"
        state.last_recon_ts = now
        be.logger.warning(
            "RECONCILIATION_SKIPPED_CIRCUIT_OPEN",
            extra={"reason_code": state.halt_reason},
        )
        return False

    try:
        from ai_trading.oms.reconcile import fetch_broker_positions, reconcile

        cached_positions = getattr(state, "position_cache", None)
        if isinstance(cached_positions, Mapping) and cached_positions:
            internal_positions = dict(cached_positions)
        else:
            internal_positions = be.retry_idempotent(
                lambda: be.compute_current_positions(runtime),
                dep="broker_positions",
                breakers=breakers,
                classify_exception=be.classify_exception,
                max_attempts=3,
                max_total_seconds=5.0,
                base_delay=0.2,
                jitter=0.1,
                context={"scope": "reconcile"},
            )

        broker_positions = be.retry_idempotent(
            lambda: fetch_broker_positions(getattr(runtime, "api", None)),
            dep="broker_positions",
            breakers=breakers,
            classify_exception=be.classify_exception,
            max_attempts=3,
            max_total_seconds=5.0,
            base_delay=0.2,
            jitter=0.1,
            context={"scope": "reconcile"},
        )
        result = reconcile(
            _normalize_positions(internal_positions),
            _normalize_positions(broker_positions),
            tolerance_shares=0.0,
        )
        state.last_recon_ts = now
        if not result.ok:
            state.recon_halt = True
            state.halt_reason = "RECON_MISMATCH_HALT"
            be.logger.error(
                "RECON_MISMATCH_HALT",
                extra={
                    "mismatches": result.mismatches,
                    "summary": result.summary,
                },
            )
            if bool(
                be.get_env(
                    "AI_TRADING_CANCEL_ALL_ON_RECON_MISMATCH",
                    "0",
                    cast=bool,
                )
            ):
                result_cancel = be.cancel_all_open_orders_oms(runtime)
                be.logger.warning(
                    "CANCEL_ALL_TRIGGERED",
                    extra={
                        "reason_code": "RECON_MISMATCH_HALT",
                        "cancelled": result_cancel.cancelled,
                        "failed": result_cancel.failed,
                    },
                )
            return False
        breakers.record_success("broker_positions")
        state.recon_halt = False
        return True
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        error_info = be.classify_exception(exc, dependency="broker_positions")
        breakers.record_failure("broker_positions", error_info)
        state.recon_halt = True
        state.halt_reason = str(error_info.reason_code or "RECON_ERROR_HALT")
        be.logger.error(
            "RECONCILIATION_ERROR_HALT",
            extra={
                "reason_code": state.halt_reason,
                "error_type": exc.__class__.__name__,
                "detail": str(exc),
            },
        )
        be._handle_error(error_info, state=state, ctx=runtime)
        state.last_recon_ts = now
        return False


__all__ = [
    "NettingGovernanceSnapshot",
    "run_netting_cycle_governance",
    "run_reconciliation_if_due",
]
