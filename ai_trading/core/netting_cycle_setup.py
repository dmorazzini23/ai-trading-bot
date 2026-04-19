"""Preparation helpers for the live netting cycle."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any, Callable, Mapping, Sequence


@dataclass(slots=True)
class NettingCycleInputs:
    sleeves: list[Any]
    sleeve_snapshot: dict[str, Any]
    allocation_weights: dict[str, float]
    learned_overrides: dict[str, Any]
    symbols: list[str]
    positions: dict[str, float]
    max_new_orders_per_cycle: int | None


@dataclass(slots=True)
class NettingPreparationError(RuntimeError):
    reason_code: str

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.reason_code)


def prepare_netting_cycle_inputs(
    *,
    state: Any,
    runtime: Any,
    cfg: Any,
    now: datetime,
    market_open_now: bool,
    breakers: Any,
    logger: Any,
    get_env: Callable[..., Any],
    build_sleeve_configs_func: Callable[[Any], Sequence[Any]],
    resolve_runtime_sleeve_whitelist_func: Callable[[], set[str]],
    maybe_update_allocation_state_func: Callable[..., None],
    allocation_weights_for_sleeves_func: Callable[[Any, Sequence[Any]], dict[str, float]],
    load_learned_overrides_func: Callable[[Any], dict[str, Any]],
    load_candidate_universe_func: Callable[[Any], list[str]],
    pre_rank_execution_candidates_func: Callable[[Sequence[str]], list[str]],
    ensure_data_fetcher_func: Callable[[Any], Any],
    retry_idempotent_func: Callable[..., Any],
    compute_current_positions_func: Callable[[Any], Mapping[str, Any]],
    classify_exception_func: Callable[..., Any],
    handle_error_func: Callable[..., None],
    merge_managed_position_symbols_func: Callable[[Sequence[str], Mapping[str, float]], list[str]],
    pending_orders_block_scope_func: Callable[[], str],
    get_cycle_budget_context_func: Callable[[], Any],
    resolve_adaptive_order_cap_func: Callable[..., tuple[int | None, dict[str, Any]]],
    select_symbols_with_budget_rotation_func: Callable[..., tuple[list[str], int, int]],
    pending_order_blocked_symbols_attr: str,
    pending_order_sample_limit: int,
) -> NettingCycleInputs | None:
    sleeves = [s for s in build_sleeve_configs_func(cfg) if s.enabled]
    sleeve_whitelist = resolve_runtime_sleeve_whitelist_func()
    if sleeve_whitelist:
        sleeves_before = list(sleeves)
        sleeves = [s for s in sleeves if str(getattr(s, "name", "") or "") in sleeve_whitelist]
        logger.info(
            "RUNTIME_SLEEVE_WHITELIST_APPLIED",
            extra={
                "requested": sorted(sleeve_whitelist),
                "before": [str(getattr(s, "name", "") or "") for s in sleeves_before],
                "after": [str(getattr(s, "name", "") or "") for s in sleeves],
            },
        )
    if not sleeves:
        if sleeve_whitelist:
            logger.warning(
                "RUNTIME_SLEEVE_WHITELIST_EMPTY",
                extra={"requested": sorted(sleeve_whitelist)},
            )
        else:
            logger.warning("HORIZONS_EMPTY")
        return None

    sleeve_configs_map = {str(s.name): s for s in sleeves}
    sleeve_snapshot = {
        str(name): {
            "timeframe": str(getattr(sleeve_cfg, "timeframe", "")),
            "entry_threshold": float(getattr(sleeve_cfg, "entry_threshold", 0.0)),
            "exit_threshold": float(getattr(sleeve_cfg, "exit_threshold", 0.0)),
            "flip_threshold": float(getattr(sleeve_cfg, "flip_threshold", 0.0)),
            "reentry_threshold": float(getattr(sleeve_cfg, "reentry_threshold", 0.0)),
            "deadband_dollars": float(getattr(sleeve_cfg, "deadband_dollars", 0.0)),
            "deadband_shares": float(getattr(sleeve_cfg, "deadband_shares", 0.0)),
            "cost_k": float(getattr(sleeve_cfg, "cost_k", 0.0)),
            "edge_scale_bps": float(getattr(sleeve_cfg, "edge_scale_bps", 0.0)),
            "turnover_cap_dollars": float(getattr(sleeve_cfg, "turnover_cap_dollars", 0.0)),
            "max_symbol_dollars": float(getattr(sleeve_cfg, "max_symbol_dollars", 0.0)),
            "max_gross_dollars": float(getattr(sleeve_cfg, "max_gross_dollars", 0.0)),
        }
        for name, sleeve_cfg in sleeve_configs_map.items()
    }
    maybe_update_allocation_state_func(
        state,
        now=now,
        market_open_now=market_open_now,
        sleeves=sleeves,
    )
    allocation_weights = allocation_weights_for_sleeves_func(cfg, sleeves)
    learned_overrides = load_learned_overrides_func(cfg)

    symbols = list(getattr(runtime, "tickers", []) or [])
    if not symbols:
        symbols = list(getattr(runtime, "universe_tickers", []) or [])
    if not symbols:
        try:
            symbols = load_candidate_universe_func(runtime)
        except Exception as exc:
            logger.warning(
                "NETTING_UNIVERSE_LOAD_FAILED",
                extra={"error": str(exc)},
            )
            symbols = []
    if not symbols:
        if bool(get_env("AI_TRADING_WARMUP_MODE", False, cast=bool)):
            logger.debug("NETTING_NO_SYMBOLS")
        else:
            logger.warning("NETTING_NO_SYMBOLS")
        return None

    canary_raw = str(get_env("AI_TRADING_CANARY_SYMBOLS", "") or "").strip()
    canary_percent = max(
        0.0,
        min(
            1.0,
            float(get_env("AI_TRADING_CANARY_PERCENT", 0.0, cast=float)),
        ),
    )
    if canary_raw or (0.0 < canary_percent < 1.0):
        selected_symbols = list(symbols)
        if canary_raw:
            canary_symbols = {
                token.strip().upper()
                for token in canary_raw.split(",")
                if token and token.strip()
            }
            selected_symbols = [
                symbol for symbol in selected_symbols if str(symbol).upper() in canary_symbols
            ]
        if 0.0 < canary_percent < 1.0:
            selected_symbols = [
                symbol
                for symbol in selected_symbols
                if (
                    int.from_bytes(
                        hashlib.blake2b(
                            str(symbol).upper().encode("utf-8"),
                            digest_size=8,
                        ).digest(),
                        byteorder="big",
                        signed=False,
                    )
                    / float(2**64)
                )
                < canary_percent
            ]
        symbols = selected_symbols
        if not state.canary_mode_logged:
            logger.warning(
                "CANARY_MODE_ACTIVE",
                extra={
                    "symbols": sorted(str(symbol).upper() for symbol in symbols),
                    "canary_percent": float(canary_percent),
                    "explicit_canary_symbols": bool(canary_raw),
                },
            )
            state.canary_mode_logged = True
        if not symbols:
            logger.warning("CANARY_MODE_EMPTY_UNIVERSE")
            return None

    symbols = pre_rank_execution_candidates_func(symbols)
    if not symbols:
        if bool(get_env("AI_TRADING_WARMUP_MODE", False, cast=bool)):
            logger.debug("NETTING_NO_SYMBOLS")
        else:
            logger.warning("NETTING_NO_SYMBOLS")
        return None

    ensure_data_fetcher_func(runtime)
    try:
        positions_raw = retry_idempotent_func(
            lambda: compute_current_positions_func(runtime),
            dep="broker_positions",
            breakers=breakers,
            classify_exception=classify_exception_func,
            max_attempts=3,
            max_total_seconds=5.0,
            base_delay=0.2,
            jitter=0.1,
            context={"scope": "netting"},
        )
    except Exception as exc:
        error_info = classify_exception_func(exc, dependency="broker_positions")
        breakers.record_failure("broker_positions", error_info)
        handle_error_func(error_info, state=state, ctx=runtime)
        raise NettingPreparationError(str(error_info.reason_code))

    positions: dict[str, float] = {}
    for raw_symbol, raw_position in dict(positions_raw or {}).items():
        symbol_key = str(raw_symbol).strip().upper()
        if not symbol_key:
            continue
        try:
            positions[symbol_key] = float(raw_position or 0.0)
        except (TypeError, ValueError):
            positions[symbol_key] = 0.0

    baseline_symbols = {str(sym).strip().upper() for sym in symbols if str(sym).strip()}
    symbols = merge_managed_position_symbols_func(symbols, positions)
    managed_symbols_added = [sym for sym in symbols if sym not in baseline_symbols]
    if managed_symbols_added:
        logger.info(
            "NETTING_MANAGED_POSITIONS_INCLUDED",
            extra={
                "added": len(managed_symbols_added),
                "sample": managed_symbols_added[:10],
                "positions_total": len(positions),
                "symbols_total": len(symbols),
            },
        )

    blocked_symbols = {
        str(sym).strip().upper()
        for sym in getattr(runtime, pending_order_blocked_symbols_attr, ())
        if str(sym).strip()
    }
    if blocked_symbols and pending_orders_block_scope_func() == "symbol":
        pre_filter_count = len(symbols)
        symbols = [sym for sym in symbols if sym not in blocked_symbols]
        if len(symbols) != pre_filter_count:
            logger.info(
                "NETTING_PENDING_SYMBOL_FILTER_APPLIED",
                extra={
                    "before": pre_filter_count,
                    "after": len(symbols),
                    "blocked_symbols_count": len(blocked_symbols),
                    "blocked_symbols": sorted(blocked_symbols)[:pending_order_sample_limit],
                },
            )
        if not symbols:
            logger.info(
                "NETTING_NO_ACTIONABLE_SYMBOLS",
                extra={
                    "reason": "pending_symbol_block",
                    "blocked_symbols_count": len(blocked_symbols),
                },
            )
            return None

    exec_engine = getattr(runtime, "execution_engine", None)
    if exec_engine is not None:
        cycle_budget = get_cycle_budget_context_func()
        adaptive_cap, adaptive_details = resolve_adaptive_order_cap_func(
            cycle_budget=cycle_budget,
            last_loop_duration_s=getattr(state, "last_loop_duration", 0.0),
        )
        try:
            setattr(exec_engine, "_adaptive_new_orders_cap", adaptive_cap)
            setattr(exec_engine, "_adaptive_new_orders_details", adaptive_details)
        except Exception:
            logger.debug("ADAPTIVE_ORDER_CAP_SET_FAILED", exc_info=True)
        adaptive_signature = (
            adaptive_cap,
            str(adaptive_details.get("mode") or "none"),
            round(float(adaptive_details.get("headroom_ratio", 1.0) or 1.0), 3),
        )
        if adaptive_signature != getattr(state, "_last_adaptive_order_cap_signature", None):
            if adaptive_cap is not None:
                logger.info(
                    "ADAPTIVE_ORDER_CAP_APPLIED",
                    extra={
                        "cap": int(adaptive_cap),
                        "mode": adaptive_details.get("mode"),
                        "headroom_ratio": adaptive_details.get("headroom_ratio"),
                        "loop_headroom_ratio": adaptive_details.get("loop_headroom_ratio"),
                        "budget_headroom_ratio": adaptive_details.get("budget_headroom_ratio"),
                    },
                )
            state._last_adaptive_order_cap_signature = adaptive_signature

    max_new_orders_per_cycle: int | None = None
    cap_source = "none"
    if exec_engine is not None:
        resolve_submit_cap = getattr(exec_engine, "_resolve_order_submit_cap", None)
        if callable(resolve_submit_cap):
            try:
                max_new_orders_per_cycle, cap_source = resolve_submit_cap()
            except Exception:
                logger.debug("NETTING_ORDER_CAP_RESOLVE_FAILED", exc_info=True)
                max_new_orders_per_cycle = None
                cap_source = "error"

    try:
        symbols_per_order = int(get_env("AI_TRADING_EXEC_SYMBOLS_PER_ORDER", 6, cast=int))
    except Exception:
        symbols_per_order = 6
    try:
        symbol_budget_min = int(get_env("AI_TRADING_EXEC_SYMBOL_BUDGET_MIN", 12, cast=int))
    except Exception:
        symbol_budget_min = 12
    try:
        symbol_budget_max = int(get_env("AI_TRADING_EXEC_SYMBOL_BUDGET_MAX", 120, cast=int))
    except Exception:
        symbol_budget_max = 120
    symbols_per_order = max(1, min(symbols_per_order, 50))
    symbol_budget_min = max(1, min(symbol_budget_min, 500))
    symbol_budget_max = max(symbol_budget_min, min(symbol_budget_max, 500))
    if max_new_orders_per_cycle is not None:
        try:
            order_cap_value = max(1, int(max_new_orders_per_cycle))
        except (TypeError, ValueError):
            order_cap_value = 1
        symbol_budget_target = order_cap_value * symbols_per_order
        symbol_budget = max(symbol_budget_min, min(symbol_budget_target, symbol_budget_max))
        if len(symbols) > symbol_budget:
            selected_symbols, held_cursor_start, held_symbols_kept = (
                select_symbols_with_budget_rotation_func(
                    symbols,
                    positions,
                    symbol_budget=symbol_budget,
                    state=state,
                )
            )
            dropped_count = max(0, len(symbols) - len(selected_symbols))
            if dropped_count > 0:
                logger.info(
                    "NETTING_SYMBOL_BUDGET_APPLIED",
                    extra={
                        "before": len(symbols),
                        "after": len(selected_symbols),
                        "dropped": dropped_count,
                        "symbol_budget": symbol_budget,
                        "order_cap": order_cap_value,
                        "symbols_per_order": symbols_per_order,
                        "cap_source": cap_source,
                        "held_symbols_kept": held_symbols_kept,
                        "held_cursor_start": held_cursor_start,
                        "held_cursor_next": int(
                            getattr(state, "netting_symbol_budget_cursor", 0)
                        ),
                        "selected_sample": selected_symbols[:10],
                    },
                )
            symbols = selected_symbols

    return NettingCycleInputs(
        sleeves=list(sleeves),
        sleeve_snapshot=sleeve_snapshot,
        allocation_weights=allocation_weights,
        learned_overrides=learned_overrides,
        symbols=list(symbols),
        positions=positions,
        max_new_orders_per_cycle=max_new_orders_per_cycle,
    )


__all__ = ["NettingCycleInputs", "NettingPreparationError", "prepare_netting_cycle_inputs"]
