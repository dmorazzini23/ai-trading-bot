"""Prepare-run runtime helpers extracted from ``bot_engine.py``."""

from __future__ import annotations

import importlib
from typing import Any


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _set_runtime_data_degraded(
    runtime: Any,
    *,
    degraded_cycle: bool,
    degrade_reason: str | None,
    degrade_fatal: bool,
) -> None:
    setattr(runtime, "_data_degraded", bool(degraded_cycle))
    if degrade_reason:
        setattr(runtime, "_data_degraded_reason", degrade_reason)
    elif hasattr(runtime, "_data_degraded_reason"):
        delattr(runtime, "_data_degraded_reason")
    if degraded_cycle:
        setattr(runtime, "_data_degraded_fatal", bool(degrade_fatal))
    elif hasattr(runtime, "_data_degraded_fatal"):
        delattr(runtime, "_data_degraded_fatal")


def truncate_degraded_candidates(
    symbols: list[str],
    runtime: Any,
    *,
    reason: str | None = None,
) -> list[str]:
    """Limit candidate list when the primary provider is degraded."""

    be = _bot_engine()
    if not symbols:
        return symbols
    cfg_limit = None
    try:
        cfg_obj = getattr(runtime, "cfg", None)
        if cfg_obj is not None:
            cfg_limit = getattr(cfg_obj, "degraded_max_candidates", None)
    except Exception:
        cfg_limit = None

    max_candidates: int | None = None
    if cfg_limit not in (None, "", 0):
        try:
            max_candidates = int(cfg_limit)
        except (TypeError, ValueError):
            max_candidates = None
    if max_candidates is None:
        try:
            max_candidates = int(
                be.get_env("TRADING__DEGRADED_MAX_CANDIDATES", "3", cast=int)
            )
        except Exception:
            max_candidates = 3
    if max_candidates <= 0:
        max_candidates = 1
    if len(symbols) <= max_candidates:
        return symbols
    penalty_factor = round(max_candidates / len(symbols), 4)
    be.logger.warning(
        "SCREEN_DEGRADED_INPUTS",
        extra={
            "penalty_factor": penalty_factor,
            "reason": reason or "provider_degraded",
        },
    )
    be.logger.warning(
        "DEGRADED_CANDIDATES_TRUNCATED",
        extra={
            "original": len(symbols),
            "truncated": max_candidates,
            "penalty_factor": penalty_factor,
            "reason": reason or "provider_degraded",
        },
    )
    return symbols[:max_candidates]


def prepare_run(
    runtime: Any,
    state: Any,
    tickers: list[str] | None,
) -> tuple[float, bool, list[str]]:
    """Prepare a trading cycle by syncing state, screening, and health-gating."""

    be = _bot_engine()
    from ai_trading.utils import portfolio_lock
    from ai_trading.data.dynamic_universe import build_dynamic_universe

    try:
        be.ensure_data_fetcher(runtime)
        cleanup_done = getattr(state, "_open_order_cleanup_done", False)
        if not cleanup_done:
            be.cancel_all_open_orders(runtime)
            try:
                setattr(state, "_open_order_cleanup_done", True)
            except Exception:
                # Best effort bookkeeping; do not block startup if state is immutable.
                be.logger.debug("OPEN_ORDER_CLEANUP_BOOKKEEPING_FAILED", exc_info=True)
        be.audit_positions(runtime)
    except be.APIError as exc:
        be.logger.warning(
            "PREPARE_RUN_API_ERROR",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        raise be.DataFetchError("api error during pre-run") from exc
    try:
        acct = be.safe_alpaca_get_account(runtime)
        if acct:
            current_cash = float(
                getattr(acct, "buying_power", getattr(acct, "cash", 0.0))
            )
            equity = float(getattr(acct, "equity", current_cash))
        else:
            current_cash = 0.0
            equity = 0.0
    except (be.APIError, TimeoutError, ConnectionError) as exc:
        be.logger.warning(
            "ACCOUNT_INFO_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        current_cash = 0.0
        equity = 0.0
    be.update_if_present(runtime, equity)
    be.params["get_capital_cap()"] = be._param(runtime, "get_capital_cap()", 0.25)
    be.compute_spy_vol_stats(runtime)

    base_watchlist = be.load_candidate_universe(runtime, tickers)
    setattr(runtime, "base_universe_tickers", list(base_watchlist))
    dynamic_result = build_dynamic_universe(runtime, list(base_watchlist))
    full_watchlist = list(dynamic_result.merged_symbols)
    setattr(runtime, "universe_tickers", list(full_watchlist))
    setattr(runtime, "dynamic_universe_metadata", dict(dynamic_result.metadata))
    degraded_snapshot = be._resolve_data_provider_degraded()
    degraded_cycle, degrade_reason, degrade_fatal = be._degrade_state(degraded_snapshot)
    try:
        cfg_obj = be.get_trading_config()
        skip_on_disabled = bool(
            getattr(cfg_obj, "skip_compute_when_provider_disabled", False)
        )
        degraded_mode = str(
            getattr(cfg_obj, "degraded_feed_mode", "block") or "block"
        ).strip().lower()
        if degraded_mode not in {"block", "widen", "hard_block"}:
            degraded_mode = "block"
        failsoft_enabled = bool(getattr(cfg_obj, "safe_mode_failsoft", True))
        if not failsoft_enabled and degraded_mode not in {"block", "hard_block"}:
            degraded_mode = "block"
    except Exception:
        skip_on_disabled = False
        degraded_mode = "block"
    failsoft_cycle = be._failsoft_mode_active()
    hard_block_cycle = degraded_cycle and degraded_mode == "hard_block"
    fatal_skip_cycle = degraded_cycle and skip_on_disabled and not failsoft_cycle and (
        be._reason_implies_fatal(degrade_reason) or degrade_fatal
    )
    _set_runtime_data_degraded(
        runtime,
        degraded_cycle=bool(degraded_cycle),
        degrade_reason=degrade_reason,
        degrade_fatal=bool(degrade_fatal),
    )
    if hard_block_cycle or fatal_skip_cycle:
        reason_label = degrade_reason or be.safe_mode_reason() or "provider_disabled"
        if hard_block_cycle and not degrade_reason:
            reason_label = "degraded_feed_hard_block"
        be.logger.warning(
            "PRIMARY_PROVIDER_DISABLED_CYCLE_SKIP",
            extra={
                "reason": reason_label,
                "symbol_budget": len(full_watchlist),
                "degraded_mode": degraded_mode,
                "hard_block": bool(hard_block_cycle),
            },
        )
        return current_cash, False, []
    try:
        be.pretrade_data_health(runtime, full_watchlist)
    except be.DataFetchError:
        be.time.sleep(1.0)
        return current_cash, False, []
    symbols = be.screen_candidates(runtime, full_watchlist)
    be.logger.info("Number of screened candidates: %s", len(symbols))
    be.logger.info(
        "CANDIDATE_PIPELINE",
        extra={
            "base_universe_count": len(base_watchlist),
            "universe_count": len(full_watchlist),
            "screened_count": len(symbols),
            "dynamic_overlay_count": len(dynamic_result.additions),
            "degraded_cycle": bool(degraded_cycle),
            "degraded_reason": degrade_reason,
        },
    )
    if not symbols:
        be.logger.warning(
            "No candidates found after filtering, using top 5 tickers fallback."
        )
        if not full_watchlist:
            full_watchlist = be.load_universe() or be.FALLBACK_SYMBOLS
        symbols = full_watchlist[:5]
    be.logger.info("CANDIDATES_SCREENED", extra={"tickers": symbols})
    runtime.tickers = symbols
    if degraded_cycle and symbols:
        symbols = be._truncate_degraded_candidates(
            symbols,
            runtime,
            reason=degrade_reason,
        )
        runtime.tickers = symbols
    if symbols:
        pre_ranked_symbols = be._pre_rank_execution_candidates(symbols, runtime=runtime)
        pre_ranked_symbols = be._apply_acceptance_rate_governor_symbol_cap(
            state,
            pre_ranked_symbols,
        )
        prepare_limit = be._resolve_prepare_symbol_limit()
        if prepare_limit is not None and len(pre_ranked_symbols) > prepare_limit:
            dropped_count = len(pre_ranked_symbols) - prepare_limit
            symbols = pre_ranked_symbols[:prepare_limit]
            be.logger.info(
                "PREPARE_SYMBOL_LIMIT_APPLIED",
                extra={
                    "before": len(pre_ranked_symbols),
                    "after": len(symbols),
                    "limit": prepare_limit,
                    "dropped": dropped_count,
                    "selected_sample": symbols[:10],
                },
            )
        else:
            symbols = pre_ranked_symbols
        runtime.tickers = symbols
    be.guard_begin_cycle(universe_size=len(symbols), degraded=degraded_cycle)
    try:
        summary = be.pre_trade_health_check(runtime, symbols)
        be.logger.info("PRE_TRADE_HEALTH", extra=summary)
        if be._pre_trade_gate():
            return current_cash, False, []
    except (
        be.APIError,
        TimeoutError,
        ConnectionError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        be.logger.warning(
            "HEALTH_CHECK_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return current_cash, False, []
    with portfolio_lock:
        runtime.portfolio_weights = be.portfolio.compute_portfolio_weights(
            runtime,
            symbols,
        )
    acct = be.safe_alpaca_get_account(runtime)
    if acct:
        current_cash = float(getattr(acct, "buying_power", getattr(acct, "cash", 0.0)))
    else:
        be.logger.error("Failed to get account information from Alpaca")
        return current_cash, False, []
    regime_ok = be.check_market_regime(runtime, state)
    return current_cash, regime_ok, symbols


__all__ = [
    "prepare_run",
    "truncate_degraded_candidates",
]
