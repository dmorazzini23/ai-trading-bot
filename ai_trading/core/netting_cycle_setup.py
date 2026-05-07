"""Preparation helpers for the live netting cycle."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


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


def _symbol_set(raw_value: Any) -> set[str]:
    return {
        token.strip().upper()
        for token in str(raw_value or "").split(",")
        if token and token.strip()
    }


def _ordered_symbol_tokens(raw_value: Any) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for token in str(raw_value or "").split(","):
        symbol = token.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _read_json_mapping(path: Any) -> Mapping[str, Any]:
    try:
        payload = json.loads(resolve_runtime_artifact_path(
            str(path),
            default_relative=str(path),
        ).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _research_universe_symbols(
    *,
    get_env: Callable[..., Any],
) -> set[str]:
    if not bool(get_env("AI_TRADING_UNIVERSE_MISMATCH_ALERT_ENABLED", True, cast=bool)):
        return set()
    configured_path = str(
        get_env(
            "AI_TRADING_DAILY_RESEARCH_REPORT_PATH",
            "runtime/research_reports/latest/daily_research_latest.json",
            cast=str,
        )
        or "runtime/research_reports/latest/daily_research_latest.json"
    ).strip()
    payload = _read_json_mapping(configured_path)
    raw_symbols = payload.get("symbols")
    if isinstance(raw_symbols, str):
        return _symbol_set(raw_symbols)
    if isinstance(raw_symbols, list):
        return {
            str(symbol).strip().upper()
            for symbol in raw_symbols
            if str(symbol).strip()
        }
    symbol_actions = payload.get("symbol_actions")
    if isinstance(symbol_actions, Mapping):
        rows = symbol_actions.get("symbols")
        if isinstance(rows, list):
            return {
                str(row.get("symbol") or "").strip().upper()
                for row in rows
                if isinstance(row, Mapping) and str(row.get("symbol") or "").strip()
            }
    return set()


def _emit_universe_mismatch_alert(
    *,
    state: Any,
    logger: Any,
    get_env: Callable[..., Any],
    source_symbols: Sequence[str],
    executable_symbols: Sequence[str],
    canary_symbols: set[str],
) -> None:
    research_symbols = _research_universe_symbols(get_env=get_env)
    shadow_symbols = _symbol_set(
        get_env("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", "", cast=str)
    )
    executable = {
        str(symbol).strip().upper()
        for symbol in executable_symbols
        if str(symbol).strip()
    }
    source = {
        str(symbol).strip().upper()
        for symbol in source_symbols
        if str(symbol).strip()
    }
    configured = (research_symbols | shadow_symbols | canary_symbols) & (source | research_symbols | shadow_symbols)
    missing = sorted(symbol for symbol in configured if symbol not in executable)
    if not missing:
        return
    signature = ",".join(missing)
    if getattr(state, "_last_universe_mismatch_signature", "") == signature:
        return
    try:
        setattr(state, "_last_universe_mismatch_signature", signature)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        pass
    logger.warning(
        "UNIVERSE_MISMATCH_ALERT",
        extra={
            "missing_executable_symbols": missing,
            "executable_symbols": sorted(executable),
            "research_symbols": sorted(research_symbols),
            "shadow_symbols": sorted(shadow_symbols),
            "canary_symbols": sorted(canary_symbols),
            "source_symbols": sorted(source),
            "reason": "configured_or_researched_symbols_not_executable",
        },
    )


def _scorecard_symbol_modes(
    *,
    get_env: Callable[..., Any],
    logger: Any,
) -> dict[str, str]:
    if not bool(get_env("AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_ENABLED", True, cast=bool)):
        return {}
    configured_path = str(
        get_env(
            "AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_PATH",
            "runtime/symbol_universe_scorecard_latest.json",
            cast=str,
        )
        or "runtime/symbol_universe_scorecard_latest.json"
    ).strip()
    path = resolve_runtime_artifact_path(
        configured_path,
        default_relative="runtime/symbol_universe_scorecard_latest.json",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if path.exists():
            logger.warning(
                "SYMBOL_UNIVERSE_SCORECARD_READ_FAILED",
                extra={"path": str(path), "error": str(exc)},
            )
        return {}
    if not isinstance(payload, Mapping):
        return {}
    status = payload.get("status")
    if not isinstance(status, Mapping) or not bool(status.get("available")):
        return {}
    rows = payload.get("symbols")
    if not isinstance(rows, list):
        return {}
    modes: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        mode = str(row.get("effective_mode") or "allow").strip().lower()
        if symbol and mode in {"allow", "shadow_only", "disabled"}:
            modes[symbol] = mode
    return modes


def _apply_symbol_universe_pruning(
    symbols: Sequence[str],
    *,
    get_env: Callable[..., Any],
    logger: Any,
) -> list[str]:
    original = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not original or not bool(get_env("AI_TRADING_SYMBOL_PRUNE_ENABLED", False, cast=bool)):
        return original
    mode = str(get_env("AI_TRADING_SYMBOL_PRUNE_MODE", "shadow", cast=str) or "shadow").strip().lower()
    enforce = mode in {"enforce", "block", "live"}
    disabled = _symbol_set(get_env("AI_TRADING_SYMBOL_PRUNE_DISABLED_SYMBOLS", "", cast=str))
    allowlist = _symbol_set(get_env("AI_TRADING_SYMBOL_PRUNE_ALLOWLIST", "", cast=str))
    scorecard_modes = _scorecard_symbol_modes(get_env=get_env, logger=logger)
    prune_shadow_only = bool(
        get_env("AI_TRADING_SYMBOL_PRUNE_SHADOW_ONLY_ENABLED", True, cast=bool)
    )
    disabled.update(symbol for symbol, symbol_mode in scorecard_modes.items() if symbol_mode == "disabled")
    if prune_shadow_only:
        disabled.update(
            symbol for symbol, symbol_mode in scorecard_modes.items() if symbol_mode == "shadow_only"
        )

    kept: list[str] = []
    pruned: list[str] = []
    for symbol in original:
        if allowlist and symbol not in allowlist:
            pruned.append(symbol)
            continue
        if symbol in disabled:
            pruned.append(symbol)
            continue
        kept.append(symbol)
    if pruned or allowlist:
        logger.info(
            "SYMBOL_UNIVERSE_PRUNE_EVALUATED",
            extra={
                "mode": mode,
                "enforced": bool(enforce),
                "before": len(original),
                "after": len(kept) if enforce else len(original),
                "pruned_count": len(pruned),
                "pruned_sample": sorted(set(pruned))[:10],
                "allowlist_count": len(allowlist),
                "scorecard_modes_count": len(scorecard_modes),
            },
        )
    if not enforce:
        return original
    return kept


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
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
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

    pre_canary_symbols = list(symbols)
    canary_raw = str(get_env("AI_TRADING_CANARY_SYMBOLS", "") or "").strip()
    active_canary_symbols: set[str] = set()
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
            canary_symbol_order = _ordered_symbol_tokens(canary_raw)
            canary_symbols = set(canary_symbol_order)
            active_canary_symbols = set(canary_symbols)
            existing_symbols = {
                str(symbol).strip().upper()
                for symbol in selected_symbols
                if str(symbol).strip()
            }
            added_canary_symbols = [
                symbol for symbol in canary_symbol_order if symbol not in existing_symbols
            ]
            if added_canary_symbols:
                selected_symbols.extend(added_canary_symbols)
                logger.info(
                    "CANARY_SYMBOLS_ADDED_TO_RUNTIME_UNIVERSE",
                    extra={
                        "added": added_canary_symbols,
                        "runtime_symbols_before": len(existing_symbols),
                        "runtime_symbols_after": len(existing_symbols) + len(added_canary_symbols),
                    },
                )
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
            logger.info(
                "CANARY_MODE_ACTIVE",
                extra={
                    "symbols": sorted(str(symbol).upper() for symbol in symbols),
                    "canary_percent": float(canary_percent),
                    "explicit_canary_symbols": bool(canary_raw),
                },
            )
            state.canary_mode_logged = True
    _emit_universe_mismatch_alert(
        state=state,
        logger=logger,
        get_env=get_env,
        source_symbols=pre_canary_symbols,
        executable_symbols=symbols,
        canary_symbols=active_canary_symbols,
    )
    if not symbols:
        logger.warning("CANARY_MODE_EMPTY_UNIVERSE")
        return None

    symbols = _apply_symbol_universe_pruning(
        symbols,
        get_env=get_env,
        logger=logger,
    )
    if not symbols:
        logger.info("NETTING_NO_ACTIONABLE_SYMBOLS", extra={"reason": "symbol_universe_prune"})
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
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
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
        except AI_TRADING_FALLBACK_EXCEPTIONS:
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
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                logger.debug("NETTING_ORDER_CAP_RESOLVE_FAILED", exc_info=True)
                max_new_orders_per_cycle = None
                cap_source = "error"

    try:
        symbols_per_order = int(get_env("AI_TRADING_EXEC_SYMBOLS_PER_ORDER", 6, cast=int))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        symbols_per_order = 6
    try:
        symbol_budget_min = int(get_env("AI_TRADING_EXEC_SYMBOL_BUDGET_MIN", 12, cast=int))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        symbol_budget_min = 12
    try:
        symbol_budget_max = int(get_env("AI_TRADING_EXEC_SYMBOL_BUDGET_MAX", 120, cast=int))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
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
