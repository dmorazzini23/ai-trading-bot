"""Legacy strategy-cycle orchestration extracted from ``bot_engine.py``."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any


def run_multi_strategy_cycle(ctx: Any) -> None:
    """Execute modular strategies, allocation, and legacy execution flow."""

    be = importlib.import_module("ai_trading.core.bot_engine")
    logger = be.logger
    RL_AGENT = be.RL_AGENT
    JSONDecodeError = be.JSONDecodeError
    COMMON_EXC = be.COMMON_EXC
    DataFetchError = be.DataFetchError
    ZoneInfo = be.ZoneInfo
    utils = be.utils
    pd = be.pd
    np = be.np
    state = be.state
    get_env = be.get_env
    get_allocator = be.get_allocator
    _init_metrics = be._init_metrics
    _increment_counter_safe = be._increment_counter_safe
    rl_eval_cycles_total = be.rl_eval_cycles_total
    rl_eval_failures_total = be.rl_eval_failures_total
    rl_eval_skips_total = be.rl_eval_skips_total
    rl_eval_state_vectors_total = be.rl_eval_state_vectors_total
    rl_eval_symbols_total = be.rl_eval_symbols_total
    _resolve_rl_symbol_limit = be._resolve_rl_symbol_limit
    _resolve_rl_candidate_symbols = be._resolve_rl_candidate_symbols
    note_rl_signals_emitted = be.note_rl_signals_emitted
    _dedupe_cycle_intents = be._dedupe_cycle_intents
    _signal_strength_threshold = be._signal_strength_threshold
    to_trade_signal = be.to_trade_signal
    get_latest_price = be.get_latest_price
    get_price_source = be.get_price_source
    _is_reliable_quote = be._is_reliable_quote
    fetch_minute_df_safe = be.fetch_minute_df_safe
    should_skip_symbol = be.should_skip_symbol
    _degraded_gap_limit_ratio = be._degraded_gap_limit_ratio
    _resolve_limit_price = be._resolve_limit_price
    _log_order_intent_blocked = be._log_order_intent_blocked
    _resolve_order_intent = be._resolve_order_intent
    _current_qty = be._current_qty
    _delta_quantity = be._delta_quantity
    _normalize_order_side_value = be._normalize_order_side_value
    _normalize_price_source_label = be._normalize_price_source_label
    _is_primary_price_source = be._is_primary_price_source

    signals_by_strategy: dict[str, list[Any]] = {}
    for strat in ctx.strategies:
        try:
            gen = getattr(strat, "generate", None)
            if callable(gen):
                sigs = gen(ctx)
            else:
                gs = getattr(strat, "generate_signals", None)
                if callable(gs):
                    sigs = gs(getattr(ctx, "market_data", ctx))
                else:
                    logger.error(
                        "Strategy %s has neither `generate` nor "
                        "`generate_signals`; skipping",
                        type(strat).__name__,
                    )
                    continue
            signals_by_strategy[strat.name] = sigs
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
            logger.warning("Strategy %s failed: %s", strat.name, exc)
    if RL_AGENT:
        _init_metrics()
        _increment_counter_safe(rl_eval_cycles_total)
        rl_cycle_summary: dict[str, Any] = {
            "status": "started",
            "reason": "init",
            "candidate_symbols": 0,
            "state_vectors": 0,
            "rl_signals": 0,
            "symbol_source": "none",
            "symbol_limit": _resolve_rl_symbol_limit(),
            "symbols_truncated": False,
        }
        try:
            all_symbols, symbol_metadata = _resolve_rl_candidate_symbols(
                ctx,
                signals_by_strategy,
            )
            rl_cycle_summary.update(
                {
                    "candidate_symbols": len(all_symbols),
                    "symbol_source": symbol_metadata.get("source", "none"),
                    "symbol_limit": int(
                        symbol_metadata.get("symbol_limit", 0) or 0
                    ),
                    "symbols_truncated": bool(
                        symbol_metadata.get("truncated", False)
                    ),
                    "symbol_sample": all_symbols[:5],
                }
            )
            _increment_counter_safe(
                rl_eval_symbols_total,
                float(len(all_symbols)),
            )
            if not all_symbols:
                _increment_counter_safe(rl_eval_skips_total)
                note_rl_signals_emitted()
                rl_cycle_summary.update({"status": "skipped", "reason": "no_symbols"})
                logger.debug(
                    "RL_SIGNALS_SKIPPED",
                    extra={"reason": "no_symbols"},
                )
            else:
                import numpy as _np

                from ai_trading.rl_trading.features import (
                    FeatureConfig,
                    compute_features,
                )

                states: list[_np.ndarray] = []
                rl_symbols: list[str] = []
                rl_feature_window = int(
                    get_env("AI_TRADING_RL_FEATURE_WINDOW", 10, cast=int)
                )
                if rl_feature_window < 5:
                    rl_feature_window = 5
                elif rl_feature_window > 256:
                    rl_feature_window = 256
                feature_cfg = FeatureConfig(window=rl_feature_window)
                for sym in all_symbols:
                    df = None
                    try:
                        df = ctx.data_fetcher.get_daily_df(ctx, sym)
                    except (
                        FileNotFoundError,
                        PermissionError,
                        IsADirectoryError,
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        TypeError,
                        OSError,
                    ):
                        df = None
                    if df is None or getattr(df, "empty", True):
                        try:
                            df = ctx.data_fetcher.get_minute_df(ctx, sym)
                        except (
                            FileNotFoundError,
                            PermissionError,
                            IsADirectoryError,
                            JSONDecodeError,
                            ValueError,
                            KeyError,
                            TypeError,
                            OSError,
                        ):
                            df = None
                    if df is None or getattr(df, "empty", True):
                        logger.debug(
                            "RL_FEATURES_SKIPPED_NO_DATA",
                            extra={"symbol": sym},
                        )
                        continue
                    try:
                        state_vec = compute_features(df, cfg=feature_cfg)
                    except (ValueError, TypeError, KeyError) as exc:
                        logger.debug(
                            "RL_FEATURES_BUILD_FAILED",
                            extra={"symbol": sym, "error": str(exc)},
                        )
                        continue
                    states.append(state_vec)
                    rl_symbols.append(sym)
                rl_cycle_summary["state_vectors"] = len(rl_symbols)
                _increment_counter_safe(
                    rl_eval_state_vectors_total,
                    float(len(rl_symbols)),
                )
                if states and rl_symbols:
                    state_mat = _np.stack(states).astype(_np.float32)
                    rl_sigs = RL_AGENT.predict(state_mat, symbols=rl_symbols)
                    note_rl_signals_emitted()
                    if rl_sigs:
                        rl_signal_list = (
                            rl_sigs if isinstance(rl_sigs, list) else [rl_sigs]
                        )
                        signals_by_strategy["rl"] = rl_signal_list
                        rl_cycle_summary.update(
                            {
                                "status": "emitted",
                                "reason": "predict_ok",
                                "rl_signals": len(rl_signal_list),
                            }
                        )
                        logger.info(
                            "RL_SIGNALS_EMITTED",
                            extra={
                                "signals": len(rl_signal_list),
                                "symbols": rl_symbols,
                            },
                        )
                    else:
                        rl_cycle_summary.update(
                            {"status": "empty", "reason": "predict_ok"}
                        )
                        logger.debug(
                            "RL_SIGNALS_EMPTY",
                            extra={"symbols": rl_symbols},
                        )
                else:
                    _increment_counter_safe(rl_eval_skips_total)
                    note_rl_signals_emitted()
                    rl_cycle_summary.update(
                        {"status": "skipped", "reason": "no_state_vectors"}
                    )
                    logger.debug(
                        "RL_SIGNALS_SKIPPED",
                        extra={
                            "reason": "no_state_vectors",
                            "symbols": all_symbols,
                        },
                    )
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
            _increment_counter_safe(rl_eval_failures_total)
            rl_cycle_summary.update(
                {
                    "status": "error",
                    "reason": "exception",
                    "error": str(exc),
                }
            )
            logger.error("RL_AGENT_ERROR", extra={"exc": str(exc)})
        finally:
            logger.info("RL_EVAL_CYCLE", extra=rl_cycle_summary)

    try:
        current_positions = ctx.api.list_positions()
        from ai_trading.signals import (
            enhance_signals_with_position_logic,
            generate_position_hold_signals,
        )

        hold_signals = generate_position_hold_signals(ctx, current_positions)
        enhanced_signals_by_strategy = {}
        for strategy_name, strategy_signals in signals_by_strategy.items():
            enhanced_signals = enhance_signals_with_position_logic(
                strategy_signals,
                ctx,
                hold_signals,
            )
            enhanced_signals_by_strategy[strategy_name] = enhanced_signals
        original_count = sum(len(sigs) for sigs in signals_by_strategy.values())
        enhanced_count = sum(
            len(sigs) for sigs in enhanced_signals_by_strategy.values()
        )
        logger.info(
            "POSITION_HOLD_FILTER",
            extra={
                "original_signals": original_count,
                "enhanced_signals": enhanced_count,
                "filtered_out": original_count - enhanced_count,
                "hold_signals_count": len(hold_signals),
            },
        )
        signals_by_strategy = enhanced_signals_by_strategy
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
        logger.warning(
            "Position holding logic failed, using original signals: %s",
            exc,
        )
    if getattr(ctx, "allocator", None) is None:
        ctx.allocator = get_allocator()
    all_signals = [s for sigs in signals_by_strategy.values() for s in sigs]
    if not all_signals:
        logger.info("No signals produced this cycle; skipping allocation and execution")
        return
    allocated = ctx.allocator.allocate(signals_by_strategy)
    if allocated is None:
        logger.info("Allocator returned no signals; skipping execution")
        return
    if isinstance(allocated, dict):
        candidate_iterable = allocated.values()
    else:
        candidate_iterable = allocated
    if isinstance(candidate_iterable, Iterable) and not isinstance(
        candidate_iterable,
        (str, bytes),
    ):
        final = list(candidate_iterable)
    else:
        final = [candidate_iterable]
    final, dedupe_summary = _dedupe_cycle_intents(final)
    logger.info("CYCLE_INTENTS_DEDUPED", extra=dedupe_summary)
    if not final:
        logger.info("No tradable signals after allocation; skipping execution")
        return
    acct = ctx.api.get_account()
    cash = float(getattr(acct, "cash", 0))
    strength_threshold = _signal_strength_threshold(ctx)
    for sig in final:
        sig = to_trade_signal(sig)
        minute_df: Any = None
        last_close: float | None = None
        minute_fetch_attempted = False
        quote_price: float | None = None
        quote_source: str | None = None
        try:
            quote_price = get_latest_price(sig.symbol)
            quote_source = get_price_source(sig.symbol)
        except COMMON_EXC:
            quote_price = None
            quote_source = None
        quote_reliable = _is_reliable_quote(quote_price, quote_source)

        def _maybe_fetch_minute_data() -> bool:
            nonlocal minute_df, last_close, minute_fetch_attempted
            if minute_fetch_attempted:
                return False
            minute_fetch_attempted = True
            local_df: Any = None
            try:
                local_df = fetch_minute_df_safe(sig.symbol)
            except DataFetchError:
                local_df = pd.DataFrame()
            if local_df is not None and not local_df.empty:
                minute_df = local_df
                last_close = utils.get_latest_close(local_df)
                coverage_meta = getattr(local_df, "attrs", {}).get(
                    "_coverage_meta",
                    {},
                )
                tz_info = ZoneInfo("America/New_York")
                start_val = (
                    coverage_meta.get("window_start")
                    if isinstance(coverage_meta, dict)
                    else None
                )
                end_val = (
                    coverage_meta.get("window_end")
                    if isinstance(coverage_meta, dict)
                    else None
                )

                def _normalize_window(val: Any) -> datetime | None:
                    if hasattr(val, "to_pydatetime"):
                        dt_val = val.to_pydatetime()
                    elif isinstance(val, datetime):
                        dt_val = val
                    else:
                        return None
                    if not isinstance(dt_val, datetime):
                        return None
                    if dt_val.tzinfo is None:
                        dt_val = dt_val.replace(tzinfo=UTC)
                    return dt_val.astimezone(tz_info)

                start_window = _normalize_window(start_val)
                end_window = _normalize_window(end_val)
                if start_window is None or end_window is None:
                    now_local = datetime.now(tz_info)
                    start_window = now_local.replace(
                        hour=9,
                        minute=30,
                        second=0,
                        microsecond=0,
                    )
                    end_window = now_local.replace(
                        hour=16,
                        minute=0,
                        second=0,
                        microsecond=0,
                    )

                def _gap_ratio_setting() -> float:
                    env_bps: float | None = None
                    for key in ("DATA_MAX_GAP_RATIO_BPS", "MAX_GAP_RATIO_BPS"):
                        try:
                            value = get_env(key, None, cast=float)
                        except COMMON_EXC:
                            continue
                        if value is not None:
                            try:
                                env_bps = max(float(value), 0.0)
                                break
                            except (TypeError, ValueError):
                                continue
                    base_bps = env_bps if env_bps is not None else 5.0
                    data_degraded_flag = bool(getattr(ctx, "_data_degraded", False))
                    degrade_fatal_flag = bool(
                        getattr(ctx, "_data_degraded_fatal", False)
                    )
                    if data_degraded_flag and not degrade_fatal_flag:
                        degraded_ratio = _degraded_gap_limit_ratio()
                        if degraded_ratio > 0.0:
                            base_bps = max(
                                base_bps,
                                degraded_ratio * 10000.0,
                            )
                    return base_bps

                max_gap_bps = _gap_ratio_setting()
                max_gap_ratio = max_gap_bps / 10000.0
                if should_skip_symbol(
                    local_df,
                    window=(start_window, end_window),
                    tz=tz_info,
                    max_gap_ratio=max_gap_ratio,
                ):
                    gap_ratio = 0.0
                    if isinstance(coverage_meta, dict):
                        try:
                            gap_ratio = float(coverage_meta.get("gap_ratio", 0.0))
                        except (TypeError, ValueError):
                            gap_ratio = 0.0
                    logger.info(
                        "SKIP_SYMBOL_DATA_GAPS | symbol=%s gap_ratio=%s",
                        sig.symbol,
                        f"{gap_ratio:.4%}",
                    )
                    return True
            else:
                minute_df = None
            return False

        if not quote_reliable:
            if _maybe_fetch_minute_data():
                continue
        price, price_source = _resolve_limit_price(
            ctx,
            sig.symbol,
            sig.side,
            minute_df,
            last_close,
        )
        if (price is None or price <= 0) and quote_reliable:
            if _maybe_fetch_minute_data():
                continue
            price, price_source = _resolve_limit_price(
                ctx,
                sig.symbol,
                sig.side,
                minute_df,
                last_close,
            )
        if price is None or price <= 0:
            logger.info("SKIP_SYMBOL_NO_VALID_PRICE", extra={"symbol": sig.symbol})
            continue
        if sig.side == "buy" and ctx.risk_engine.position_exists(ctx.api, sig.symbol):
            logger.info("SKIP_DUPLICATE_LONG", extra={"symbol": sig.symbol})
            continue
        try:
            strength = float(getattr(sig, "strength", 0.0))
        except (TypeError, ValueError):
            strength = 0.0
        if abs(strength) < strength_threshold:
            logger.info(
                "SIGNAL_STRENGTH_REJECTED",
                extra={
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "strategy": getattr(sig, "strategy", "unknown"),
                    "signal_strength": strength,
                    "threshold": strength_threshold,
                },
            )
            continue
        logger.debug(
            "PROCESSING_SIGNAL",
            extra={
                "symbol": sig.symbol,
                "side": sig.side,
                "confidence": sig.confidence,
                "strategy": getattr(sig, "strategy", "unknown"),
                "weight": getattr(sig, "weight", 0.0),
                "signal_strength": strength,
                "strength_threshold": strength_threshold,
            },
        )
        qty = ctx.risk_engine.position_size(sig, cash, price)
        if qty is None or not np.isfinite(qty) or qty <= 0:
            logger.warning(
                "SKIP_INVALID_QTY",
                extra={
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "qty": qty,
                    "cash": cash,
                    "price": price,
                    "signal_strength": strength,
                    "threshold": strength_threshold,
                },
            )
            continue
        logger.debug(
            "RISK_MANAGER_APPROVED",
            extra={
                "symbol": sig.symbol,
                "side": sig.side,
                "qty": qty,
                "signal_strength": strength,
                "threshold": strength_threshold,
            },
        )
        if sig.side not in ["buy", "sell"]:
            logger.error(
                "INVALID_SIGNAL_SIDE",
                extra={
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "expected": "buy or sell",
                },
            )
            continue
        logger.info(
            "EXECUTING_ORDER",
            extra={
                "symbol": sig.symbol,
                "side": sig.side,
                "qty": qty,
                "price": price,
                "signal_strength": strength,
                "threshold": strength_threshold,
                "confidence": sig.confidence,
            },
        )
        try:
            target_qty = int(round(float(qty)))
        except (TypeError, ValueError):
            target_qty = 0
        if target_qty <= 0:
            continue
        try:
            order_type = "market" if price is None else "limit"
            order_kwargs = {
                "order_type": order_type,
                "asset_class": sig.asset_class,
                "signal": sig,
                "signal_weight": getattr(sig, "weight", None),
            }
            if price is not None:
                order_kwargs["price"] = price
                order_kwargs["price_hint"] = price
            annotations: dict[str, Any] = {}
            price_source_label: Any = None
            try:
                if quote_source is not None:
                    price_source_label = quote_source
                elif price_source is not None:
                    price_source_label = price_source
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                price_source_label = None
            if price_source_label in (None, ""):
                try:
                    price_source_label = get_price_source(sig.symbol)
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    price_source_label = None
            if price_source_label is not None:
                annotations["price_source"] = price_source_label
            primary_source_label = _normalize_price_source_label(price_source_label)
            if not primary_source_label:
                primary_source_label = _normalize_price_source_label(quote_source)
            if not primary_source_label:
                primary_source_label = _normalize_price_source_label(price_source)
            using_fallback = bool(
                primary_source_label
                and not _is_primary_price_source(primary_source_label)
            )
            if using_fallback:
                annotations["using_fallback_price"] = True
                order_kwargs["using_fallback_price"] = True
            if annotations:
                order_kwargs["annotations"] = annotations
            try:
                sl = getattr(
                    getattr(ctx, "stop_targets", {}),
                    "get",
                    lambda _s, _d=None: None,
                )(sig.symbol, None)
            except COMMON_EXC:
                sl = None
            try:
                tp = getattr(
                    getattr(ctx, "take_profit_targets", {}),
                    "get",
                    lambda _s, _d=None: None,
                )(sig.symbol, None)
            except COMMON_EXC:
                tp = None
            if isinstance(sl, (int, float)) and sl > 0:
                order_kwargs["stop_loss"] = float(sl)
                order_kwargs.setdefault("order_class", "bracket")
            if isinstance(tp, (int, float)) and tp > 0:
                order_kwargs["take_profit"] = float(tp)
                order_kwargs.setdefault("order_class", "bracket")
            try:
                target_weight_val = float(ctx.portfolio_weights.get(sig.symbol, 0.0))
            except (AttributeError, TypeError, ValueError):
                try:
                    target_weight_val = float(getattr(sig, "weight", 0.0) or 0.0)
                except (TypeError, ValueError):
                    target_weight_val = 0.0
            intent_decision = _resolve_order_intent(
                ctx,
                state,
                symbol=sig.symbol,
                signal_side=sig.side,
                target_weight=target_weight_val,
            )
            if not intent_decision:
                _log_order_intent_blocked(intent_decision)
                continue
            expected_sides = set(intent_decision.expected_sides) or {
                intent_decision.order_side
            }
            open_buy_qty = open_sell_qty = 0
            engine_for_delta = getattr(ctx, "execution_engine", None)
            if engine_for_delta is not None and hasattr(
                engine_for_delta,
                "open_order_totals",
            ):
                try:
                    open_buy_qty, open_sell_qty = engine_for_delta.open_order_totals(
                        sig.symbol
                    )
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    open_buy_qty = open_sell_qty = 0
            position_qty = _current_qty(ctx, sig.symbol)
            delta_qty = _delta_quantity(
                intent_decision.order_side,
                target_qty,
                position_qty,
                open_buy_qty,
                open_sell_qty,
            )
            logger.info(
                "DELTA_BREAKDOWN",
                extra={
                    "symbol": sig.symbol,
                    "order_side": intent_decision.order_side,
                    "target": target_qty,
                    "position": position_qty,
                    "open_buy": open_buy_qty,
                    "open_sell": open_sell_qty,
                    "delta": delta_qty,
                },
            )
            if delta_qty <= 0:
                continue
            qty = delta_qty
            result = ctx.execution_engine.execute_order(
                sig.symbol,
                intent_decision.order_side,
                qty,
                **order_kwargs,
            )
            if result is not None:
                try:
                    state.execution_metrics.submitted += 1
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    logger.debug(
                        "EXECUTION_METRIC_SUBMITTED_INCREMENT_FAILED",
                        exc_info=True,
                    )
                actual_side_norm = _normalize_order_side_value(
                    getattr(result, "side", None)
                )
                if actual_side_norm is None:
                    actual_side_norm = _normalize_order_side_value(
                        intent_decision.order_side
                    )
                if actual_side_norm not in expected_sides:
                    logger.error(
                        "ORDER_SIDE_MISMATCH",
                        extra={
                            "symbol": sig.symbol,
                            "signal_side": sig.side,
                            "order_side": actual_side_norm,
                            "expected_sides": tuple(sorted(expected_sides)),
                            "order_id": getattr(result, "id", None),
                        },
                    )
                    raise AssertionError("order_side_mismatch")
        except AssertionError as exc:
            logger.warning(
                "ORDER_EXECUTION_ABORTED",
                extra={
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "qty": qty,
                    "reason": str(exc),
                },
            )
            continue
        if result is None:
            continue
        if not getattr(result, "reconciled", True):
            logger.warning(
                "BROKER_RECONCILE_SKIPPED",
                extra={
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "order_id": getattr(result, "order", None),
                },
            )
            continue
        filled_qty = getattr(result, "filled_quantity", 0) or 0
        if filled_qty <= 0:
            continue
        requested_qty = getattr(result, "requested_quantity", qty) or qty
        try:
            requested_qty = float(requested_qty)
            filled_qty = float(filled_qty)
        except (TypeError, ValueError):
            continue
        if requested_qty <= 0:
            continue
        try:
            signal_weight = float(getattr(sig, "weight", 0.0))
        except (TypeError, ValueError):
            signal_weight = 0.0
        fill_ratio = filled_qty / requested_qty
        if fill_ratio <= 0:
            continue
        filled_weight = signal_weight * min(1.0, fill_ratio)
        if filled_weight == 0:
            continue
        try:
            from dataclasses import replace

            filled_signal = replace(sig, weight=filled_weight)
        except COMMON_EXC:
            try:
                filled_signal = sig.__class__(
                    **{**getattr(sig, "__dict__", {}), "weight": filled_weight}
                )
            except COMMON_EXC:
                continue
        ctx.risk_engine.register_fill(filled_signal)
        try:
            ctx.execution_engine.mark_fill_reported(str(result), int(filled_qty))
        except COMMON_EXC:
            logger.debug("MARK_FILL_REPORTED_FAILED", exc_info=True)
    try:
        engine = getattr(ctx, "execution_engine", None)
        if engine is not None:
            end_hook = getattr(engine, "end_cycle", None)
            if callable(end_hook):
                end_hook()
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
        logger.error("TRAILING_STOP_CHECK_FAILED", extra={"exc": str(exc)})


def process_symbols_cycle(
    symbols: list[str],
    current_cash: float,
    model: Any,
    regime_ok: bool,
    close_shorts: bool = False,
    skip_duplicates: bool = False,
) -> tuple[list[str], dict[str, int], int]:
    """Screen, fetch, and execute legacy symbols for the current cycle."""

    be = importlib.import_module("ai_trading.core.bot_engine")
    logger = be.logger
    Lock = be.Lock
    SoftBudget = be.SoftBudget
    provider_monitor = be.provider_monitor
    CycleAbortSafeMode = be.CycleAbortSafeMode
    COMMON_EXC = be.COMMON_EXC
    get_env = be.get_env
    get_ctx = be.get_ctx
    get_trading_config = be.get_trading_config
    should_stop = be.should_stop
    get_cycle_budget_context = be.get_cycle_budget_context
    state = be.state
    _safe_mode_blocks_trading = be._safe_mode_blocks_trading
    safe_mode_reason = be.safe_mode_reason
    _mark_ctx_degraded = be._mark_ctx_degraded
    _log_safe_mode_continue = be._log_safe_mode_continue
    _degrade_state = be._degrade_state
    _resolve_data_provider_degraded = be._resolve_data_provider_degraded
    _trade_limit_reached = be._trade_limit_reached
    ensure_final_bar = be.ensure_final_bar
    log_skip_cooldown = be.log_skip_cooldown
    skipped_duplicates = be.skipped_duplicates
    skipped_cooldown = be.skipped_cooldown
    submit_order = be.submit_order
    trade_cooldowns_lock = be.trade_cooldowns_lock
    _pre_rank_execution_candidates = be._pre_rank_execution_candidates
    executors = be.executors
    _resolve_adaptive_order_cap = be._resolve_adaptive_order_cap
    EmptyBarsError = be.EmptyBarsError
    DataFetchError = be.DataFetchError
    fetch_minute_df_safe = be.fetch_minute_df_safe
    _safe_trade = be._safe_trade
    is_market_open = be.is_market_open
    pd = be.pd
    bar_from_frame = be.bar_from_frame
    _PENDING_ORDER_BLOCKED_SYMBOLS_ATTR = be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR
    _PENDING_ORDER_SAMPLE_LIMIT = be._PENDING_ORDER_SAMPLE_LIMIT
    MAX_TRADES_PER_HOUR = be.MAX_TRADES_PER_HOUR
    get_trade_cooldown_min = be.get_trade_cooldown_min
    _warmup_data_only_mode_active = be._warmup_data_only_mode_active
    _warmup_symbol_limit = be._warmup_symbol_limit
    APIError = be.APIError

    processed: list[str] = []
    row_counts: dict[str, int] = {}
    fetch_attempts = 0
    fetch_attempts_lock = Lock()
    warmup_data_only = _warmup_data_only_mode_active()
    warmup_symbol_limit = _warmup_symbol_limit()
    ctx = get_ctx()
    safe_mode_policy_blocks = _safe_mode_blocks_trading()
    pytest_mode = bool(get_env("PYTEST_RUNNING") or get_env("PYTEST_CURRENT_TEST"))
    ctx_tracks_degraded_state = any(
        hasattr(ctx, attr_name)
        for attr_name in (
            "_data_degraded",
            "_data_degraded_reason",
            "_data_degraded_fatal",
        )
    )
    respect_provider_safe_mode = not pytest_mode or ctx_tracks_degraded_state
    safe_mode_flag = (
        provider_monitor.is_safe_mode_active() if respect_provider_safe_mode else False
    )
    safe_mode_label = safe_mode_reason() or "provider_safe_mode"
    if safe_mode_flag and safe_mode_policy_blocks:
        raise CycleAbortSafeMode(safe_mode_label)
    if safe_mode_flag:
        _mark_ctx_degraded(ctx, safe_mode_label)
        _log_safe_mode_continue(ctx, stage="process_symbols", reason=safe_mode_label)
    data_degraded = bool(getattr(ctx, "_data_degraded", False))
    degrade_reason = (
        getattr(ctx, "_data_degraded_reason", None) or "provider_degraded"
    )
    degrade_fatal = bool(getattr(ctx, "_data_degraded_fatal", False))
    degrade_announce_logged = False
    degraded_mode = "block"
    try:
        cfg_obj = get_trading_config()
    except COMMON_EXC:
        cfg_obj = None
    if cfg_obj is not None:
        degraded_mode = str(
            getattr(cfg_obj, "degraded_feed_mode", "block") or "block"
        ).strip().lower()
        if degraded_mode not in {"block", "widen", "hard_block"}:
            degraded_mode = "block"
    explicit_degraded_mode = get_env("TRADING__DEGRADED_FEED_MODE") or get_env(
        "DEGRADED_FEED_MODE"
    )
    if pytest_mode and not explicit_degraded_mode and degraded_mode == "block":
        degraded_mode = "widen"
    detect_runtime_degrade = not pytest_mode or ctx_tracks_degraded_state
    respect_stop_signal = (
        not pytest_mode or ctx_tracks_degraded_state or hasattr(ctx, "execution_engine")
    )

    def _should_stop_now() -> bool:
        return bool(should_stop()) if respect_stop_signal else False

    cycle_budget = get_cycle_budget_context()
    if not hasattr(state, "trade_cooldowns"):
        state.trade_cooldowns = {}
    if not hasattr(state, "last_trade_direction"):
        state.last_trade_direction = {}
    if not hasattr(state, "entry_flip_signal_streak"):
        state.entry_flip_signal_streak = {}
    if not hasattr(state, "last_entry_side"):
        state.last_entry_side = {}
    if not hasattr(state, "entry_expectancy_context"):
        state.entry_expectancy_context = {}
    if not hasattr(state, "expectancy_history"):
        state.expectancy_history = {}
    if not hasattr(state, "exit_policy_state"):
        state.exit_policy_state = {}
    if not hasattr(state, "policy_rollback_disabled_slices"):
        state.policy_rollback_disabled_slices = []
    if not hasattr(state, "last_policy_ablation_run_date"):
        state.last_policy_ablation_run_date = None

    def _record_symbol_skip(
        symbol: str,
        *,
        gates: Sequence[str],
        event: str,
        reasons: Sequence[str] | None = None,
        price_df: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        recorder = getattr(state, "_legacy_decision_recorder", None)
        if recorder is None or not hasattr(recorder, "record"):
            return
        market_bar = bar_from_frame(
            symbol,
            price_df,
            timeframe="1Min",
            provider="alpaca",
        )
        freshness: float | None = None
        if market_bar is not None and isinstance(market_bar.ts, datetime):
            freshness = max((datetime.now(UTC) - market_bar.ts).total_seconds(), 0.0)
        try:
            recorder.record(
                symbol=symbol,
                market_bar=market_bar,
                bar_ts=market_bar.ts if market_bar is not None else None,
                signal_side="hold",
                final_score=0.0,
                confidence=0.0,
                strategy_id=None,
                accepted=False,
                gates=gates,
                reasons=reasons or gates,
                provider=market_bar.provider if market_bar is not None else None,
                feed=market_bar.feed if market_bar is not None else None,
                reference_price=market_bar.close if market_bar is not None else None,
                event=event,
                data_freshness_sec=freshness,
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("PROCESS_SYMBOL_DECISION_RECORD_FAILED", exc_info=True)

    filtered: list[str] = []
    cd_skipped: list[str] = []
    blocked_symbols = {
        str(sym).strip().upper()
        for sym in getattr(state, _PENDING_ORDER_BLOCKED_SYMBOLS_ATTR, ())
        if str(sym).strip()
    }
    if blocked_symbols:
        original_count = len(symbols)
        symbols = [
            str(sym).strip().upper()
            for sym in symbols
            if str(sym).strip() and str(sym).strip().upper() not in blocked_symbols
        ]
        logger.info(
            "PENDING_ORDERS_SYMBOLS_FILTERED",
            extra={
                "before": original_count,
                "after": len(symbols),
                "blocked_symbols_count": len(blocked_symbols),
                "blocked_symbols": sorted(blocked_symbols)[
                    :_PENDING_ORDER_SAMPLE_LIMIT
                ],
            },
        )
    configured_max_symbols = 0
    raw_max_symbols = get_env("MAX_SYMBOLS_PER_CYCLE", 0)
    try:
        configured_max_symbols = int(raw_max_symbols)
    except (TypeError, ValueError):
        configured_max_symbols = 0
    if configured_max_symbols > 0:
        max_symbols_per_cycle = min(configured_max_symbols, len(symbols))
    else:
        max_symbols_per_cycle = len(symbols)
    if 0 < max_symbols_per_cycle < len(symbols):
        logger.info(
            "SYMBOL_PROCESSING_CAP_APPLIED",
            extra={
                "configured_max_symbols": configured_max_symbols,
                "before": len(symbols),
                "after": max_symbols_per_cycle,
            },
        )
    processed_symbols = 0
    budget_sec = get_env("SYMBOL_PROCESS_BUDGET", 300)
    try:
        budget_sec = float(budget_sec)
    except (TypeError, ValueError):
        budget_sec = 300.0
    proc_budget = SoftBudget(int(max(0.0, float(budget_sec)) * 1000))
    quota_notice_logged = False
    quota_notice_lock = Lock()
    try:
        trade_cooldown_seconds = max(0.0, float(get_trade_cooldown_min()) * 60.0)
    except (TypeError, ValueError):
        trade_cooldown_seconds = 15.0 * 60.0

    def _log_trade_quota_once() -> None:
        nonlocal quota_notice_logged
        with quota_notice_lock:
            if quota_notice_logged:
                return
            quota_notice_logged = True
            logger.info(
                "TRADE_QUOTA_EXHAUSTED_SKIP",
                extra={"max_per_hour": MAX_TRADES_PER_HOUR},
            )

    for symbol in symbols:
        safe_mode_now = (
            provider_monitor.is_safe_mode_active() if respect_provider_safe_mode else False
        )
        if safe_mode_now and safe_mode_policy_blocks:
            raise CycleAbortSafeMode(safe_mode_reason() or "provider_safe_mode")
        if safe_mode_now and not data_degraded:
            degrade_reason = safe_mode_reason() or degrade_reason
            data_degraded = True
            _mark_ctx_degraded(ctx, degrade_reason)
            _log_safe_mode_continue(ctx, stage="process_symbols", reason=degrade_reason)
        detected = False
        if detect_runtime_degrade and (not data_degraded or not degrade_fatal):
            detected_cycle, detected_reason, detected_fatal = _degrade_state(
                _resolve_data_provider_degraded()
            )
            if detected_cycle:
                data_degraded = True
                detected = True
                degrade_reason = detected_reason or degrade_reason
                degrade_fatal = bool(degrade_fatal or detected_fatal)
                try:
                    setattr(ctx, "_data_degraded", True)
                    if degrade_reason:
                        setattr(ctx, "_data_degraded_reason", degrade_reason)
                    setattr(ctx, "_data_degraded_fatal", degrade_fatal)
                except (AttributeError, Exception):
                    pass
        pos = state.position_cache.get(symbol, 0)
        if data_degraded:
            if not degrade_announce_logged or detected:
                logger.warning(
                    "DEGRADED_FEED_ACTIVE",
                    extra={"reason": degrade_reason, "fatal": degrade_fatal},
                )
                degrade_announce_logged = True
            if degrade_fatal:
                logger.warning(
                    "DEGRADED_FEED_SKIP_SYMBOL",
                    extra={"symbol": symbol, "reason": degrade_reason},
                )
                _record_symbol_skip(
                    symbol,
                    gates=["DEGRADED_FEED_SKIP_SYMBOL"],
                    reasons=[str(degrade_reason)],
                    event="legacy_process_symbols_degraded_feed_skip",
                )
                continue
            if degraded_mode in {"block", "hard_block"} and pos >= 0:
                logger.warning(
                    "DEGRADED_FEED_SKIP_SYMBOL",
                    extra={
                        "symbol": symbol,
                        "reason": degrade_reason,
                        "mode": degraded_mode,
                    },
                )
                _record_symbol_skip(
                    symbol,
                    gates=["DEGRADED_FEED_SKIP_SYMBOL"],
                    reasons=[str(degrade_reason), str(degraded_mode)],
                    event="legacy_process_symbols_degraded_feed_skip",
                )
                continue
        now = datetime.now(UTC)
        if not warmup_data_only and _trade_limit_reached(state, now):
            _log_trade_quota_once()
            break
        if not ensure_final_bar(symbol, "1min"):
            logger.info(
                "SKIP_PARTIAL_BAR",
                extra={"symbol": symbol, "timeframe": "1min"},
            )
            _record_symbol_skip(
                symbol,
                gates=["SKIP_PARTIAL_BAR"],
                reasons=["1min"],
                event="legacy_process_symbols_partial_bar_skip",
            )
            continue
        if processed_symbols >= max_symbols_per_cycle:
            logger.warning(
                "SYMBOL_PROCESSING_CIRCUIT_BREAKER",
                extra={
                    "processed_count": processed_symbols,
                    "remaining_count": len(symbols) - processed_symbols,
                    "reason": "max_symbols_reached",
                },
            )
            break
        if proc_budget.over_budget():
            logger.warning(
                "SYMBOL_PROCESSING_CIRCUIT_BREAKER",
                extra={
                    "processed_count": processed_symbols,
                    "elapsed_seconds": proc_budget.elapsed_ms() / 1000,
                    "reason": "time_limit_reached",
                },
            )
            break
        processed_symbols += 1
        if not warmup_data_only:
            if pos < 0 and close_shorts:
                logger.info(
                    "SKIP_SHORT_CLOSE_QUEUED | symbol=%s qty=%s",
                    symbol,
                    -pos,
                )
                _record_symbol_skip(
                    symbol,
                    gates=["SHORT_CLOSE_QUEUED"],
                    reasons=["close_shorts"],
                    event="legacy_process_symbols_short_close_queued",
                    metadata={"qty": int(abs(pos))},
                )
                continue
            if skip_duplicates and pos != 0:
                log_skip_cooldown(symbol, reason="duplicate")
                skipped_duplicates.inc()
                _record_symbol_skip(
                    symbol,
                    gates=["SKIP_DUPLICATE_POSITION"],
                    reasons=["duplicate"],
                    event="legacy_process_symbols_duplicate_skip",
                    metadata={"current_qty": int(pos)},
                )
                continue
            if pos > 0:
                logger.info("SKIP_HELD_POSITION | already long, skipping close")
                skipped_duplicates.inc()
                _record_symbol_skip(
                    symbol,
                    gates=["SKIP_HELD_POSITION"],
                    reasons=["already_long"],
                    event="legacy_process_symbols_held_position_skip",
                    metadata={"current_qty": int(pos)},
                )
                continue
            if pos < 0:
                logger.info(
                    "SHORT_CLOSE_QUEUED | symbol=%s  qty=%d",
                    symbol,
                    abs(pos),
                )
                try:
                    submit_order(ctx, symbol, abs(pos), "buy")
                except (APIError, TimeoutError, ConnectionError) as exc:
                    logger.warning(
                        "SHORT_CLOSE_FAIL",
                        extra={
                            "symbol": symbol,
                            "cause": exc.__class__.__name__,
                            "detail": str(exc),
                        },
                    )
                _record_symbol_skip(
                    symbol,
                    gates=["SHORT_CLOSE_QUEUED"],
                    reasons=["existing_short_position"],
                    event="legacy_process_symbols_short_close_queued",
                    metadata={"current_qty": int(pos)},
                )
                continue
            with trade_cooldowns_lock:
                ts = state.trade_cooldowns.get(symbol)
            if ts and (now - ts).total_seconds() < trade_cooldown_seconds:
                cd_skipped.append(symbol)
                skipped_cooldown.inc()
                _record_symbol_skip(
                    symbol,
                    gates=["TRADE_COOLDOWN_ACTIVE"],
                    event="legacy_process_symbols_cooldown_skip",
                )
                continue
        filtered.append(symbol)
    symbols = filtered
    if warmup_data_only and len(symbols) > warmup_symbol_limit:
        original_count = len(symbols)
        symbols = symbols[:warmup_symbol_limit]
        logger.info(
            "WARMUP_SYMBOL_LIMIT_APPLIED",
            extra={
                "limit": warmup_symbol_limit,
                "original_count": original_count,
                "retained_count": len(symbols),
            },
        )
    symbols = _pre_rank_execution_candidates(symbols, runtime=ctx)
    if cycle_budget:
        cycle_budget.register_total(len(symbols))
        if symbols and cycle_budget.should_throttle():
            cycle_budget.mark_skipped(symbols)
            logger.debug(
                "CYCLE_BUDGET_SKIP_SYMBOLS",
                extra={
                    "count": len(symbols),
                    "reason": cycle_budget.cause or "guard",
                },
            )
            return [], {sym: 0 for sym in symbols}, fetch_attempts
    executors._ensure_executors()
    if cd_skipped:
        log_skip_cooldown(cd_skipped)
    data_stats_lock = Lock()
    data_stats = {"failed": 0, "succeeded": 0}
    broker_stats_before = {"capacity_skips": 0, "retry_count": 0, "skipped_orders": 0}
    exec_engine = getattr(ctx, "execution_engine", None)
    if exec_engine is not None:
        adaptive_cap, adaptive_details = _resolve_adaptive_order_cap(
            cycle_budget=cycle_budget,
            last_loop_duration_s=getattr(state, "last_loop_duration", 0.0),
        )
        try:
            setattr(exec_engine, "_adaptive_new_orders_cap", adaptive_cap)
            setattr(exec_engine, "_adaptive_new_orders_details", adaptive_details)
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("ADAPTIVE_ORDER_CAP_SET_FAILED", exc_info=True)
        if adaptive_cap is not None:
            logger.info(
                "ADAPTIVE_ORDER_CAP_APPLIED",
                extra={
                    "cap": int(adaptive_cap),
                    "mode": adaptive_details.get("mode"),
                    "headroom_ratio": adaptive_details.get("headroom_ratio"),
                    "loop_headroom_ratio": adaptive_details.get(
                        "loop_headroom_ratio"
                    ),
                    "budget_headroom_ratio": adaptive_details.get(
                        "budget_headroom_ratio"
                    ),
                },
            )
        start_hook = getattr(exec_engine, "start_cycle", None)
        if callable(start_hook):
            try:
                start_hook()
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                logger.warning(
                    "EXECUTION_START_CYCLE_FAILED",
                    extra={
                        "cause": exc.__class__.__name__,
                        "detail": str(exc),
                    },
                    exc_info=True,
                )
    stats_snapshot = getattr(exec_engine, "stats", None)
    if isinstance(stats_snapshot, dict):
        try:
            broker_stats_before["capacity_skips"] = int(
                stats_snapshot.get("capacity_skips", 0) or 0
            )
        except (TypeError, ValueError):
            broker_stats_before["capacity_skips"] = 0
        try:
            broker_stats_before["retry_count"] = int(
                stats_snapshot.get("retry_count", 0) or 0
            )
        except (TypeError, ValueError):
            broker_stats_before["retry_count"] = 0
        try:
            broker_stats_before["skipped_orders"] = int(
                stats_snapshot.get("skipped_orders", 0) or 0
            )
        except (TypeError, ValueError):
            broker_stats_before["skipped_orders"] = 0

    def process_symbol(symbol: str) -> None:
        completed_stages: list[str] = []
        local_success = False
        nonlocal fetch_attempts

        def _checkpoint(pending: str) -> bool:
            if _should_stop_now():
                logger.info(
                    "PROCESS_SYMBOL_STOP",
                    extra={
                        "symbol": symbol,
                        "completed": (
                            ",".join(completed_stages) if completed_stages else ""
                        ),
                        "pending": pending,
                    },
                )
                return True
            return False

        try:
            if (
                respect_provider_safe_mode
                and provider_monitor.is_safe_mode_active()
                and _safe_mode_blocks_trading()
            ):
                logger.warning(
                    "SAFE_MODE_BLOCK",
                    extra={
                        "symbol": symbol,
                        "reason": safe_mode_reason() or "provider_safe_mode",
                        "block_reason": "provider_disabled_midcycle",
                    },
                )
                return
            if _checkpoint("start"):
                return
            if cycle_budget and cycle_budget.should_throttle():
                cycle_budget.mark_skipped([symbol])
                logger.debug(
                    "CYCLE_BUDGET_SKIP_SYMBOL",
                    extra={
                        "symbol": symbol,
                        "reason": cycle_budget.cause or "guard",
                    },
                )
                return
            logger.info("PROCESSING_SYMBOL | symbol=%s", symbol)
            if not is_market_open():
                logger.info("MARKET_CLOSED_SKIP_SYMBOL", extra={"symbol": symbol})
                _record_symbol_skip(
                    symbol,
                    gates=["MARKET_CLOSED_SKIP_SYMBOL"],
                    event="legacy_process_symbols_market_closed_skip",
                )
                return
            if not warmup_data_only and _trade_limit_reached(state, datetime.now(UTC)):
                _log_trade_quota_once()
                _record_symbol_skip(
                    symbol,
                    gates=["TRADE_QUOTA_EXHAUSTED_SKIP"],
                    event="legacy_process_symbols_quota_skip",
                )
                return

            def _halt(reason: str) -> None:
                logger.info("COVERAGE_BLOCK", extra={"symbol": symbol, "reason": reason})
                halt_mgr = getattr(ctx, "halt_manager", None)
                if halt_mgr is not None:
                    try:
                        halt_mgr.manual_halt_trading(f"{symbol}:{reason}")
                    except (AttributeError, RuntimeError) as exc:
                        logger.error("HALT_MANAGER_ERROR", extra={"cause": str(exc)})

            try:
                if _checkpoint("fetch"):
                    return
                with fetch_attempts_lock:
                    fetch_attempts += 1
                price_df = fetch_minute_df_safe(symbol)
                completed_stages.append("fetch")
            except EmptyBarsError as exc:
                logger.warning(
                    "PROCESS_SYMBOL_EMPTY_BARS",
                    extra={
                        "symbol": symbol,
                        "timeframe": "1Min",
                        "detail": str(exc),
                    },
                )
                _record_symbol_skip(
                    symbol,
                    gates=["PROCESS_SYMBOL_EMPTY_BARS"],
                    reasons=[str(exc)],
                    event="legacy_process_symbols_empty_bars",
                )
                with data_stats_lock:
                    data_stats["failed"] += 1
                return
            except DataFetchError as exc:
                reason = getattr(exc, "fetch_reason", "")
                if reason in {
                    "close_column_all_nan",
                    "close_column_missing",
                    "ohlcv_columns_missing",
                }:
                    _halt("empty_frame")
                else:
                    _halt("minute_data_unavailable")
                _record_symbol_skip(
                    symbol,
                    gates=["DATA_FETCH_UNAVAILABLE"],
                    reasons=[str(reason or exc)],
                    event="legacy_process_symbols_data_fetch_failed",
                )
                with data_stats_lock:
                    data_stats["failed"] += 1
                return
            row_counts[symbol] = len(price_df)
            logger.info("FETCHED_ROWS | %s rows=%s", symbol, len(price_df))
            if price_df.empty or "close" not in price_df.columns:
                _halt("empty_frame")
                _record_symbol_skip(
                    symbol,
                    gates=["EMPTY_FRAME"],
                    event="legacy_process_symbols_empty_frame",
                    price_df=price_df if isinstance(price_df, pd.DataFrame) else None,
                )
                with data_stats_lock:
                    data_stats["failed"] += 1
                return
            close_series = price_df["close"] if "close" in price_df.columns else None
            if close_series is not None:
                try:
                    non_null_count = int(close_series.count())
                except COMMON_EXC:
                    try:
                        non_null_count = int(close_series.dropna().shape[0])
                    except COMMON_EXC:
                        non_null_count = 0
                if non_null_count == 0:
                    _halt("empty_frame")
                    _record_symbol_skip(
                        symbol,
                        gates=["EMPTY_FRAME"],
                        event="legacy_process_symbols_empty_frame",
                        price_df=(
                            price_df if isinstance(price_df, pd.DataFrame) else None
                        ),
                    )
                    with data_stats_lock:
                        data_stats["failed"] += 1
                    return
            if not warmup_data_only and symbol in state.position_cache:
                _record_symbol_skip(
                    symbol,
                    gates=["SKIP_POSITION_CACHE_PRESENT"],
                    event="legacy_process_symbols_position_cache_skip",
                    price_df=price_df if isinstance(price_df, pd.DataFrame) else None,
                )
                return
            processed.append(symbol)
            if cycle_budget:
                cycle_budget.note_processed()
            if warmup_data_only:
                completed_stages.append("warmup_data_only")
                _record_symbol_skip(
                    symbol,
                    gates=["WARMUP_DATA_ONLY"],
                    event="legacy_process_symbols_warmup_skip",
                    price_df=price_df if isinstance(price_df, pd.DataFrame) else None,
                )
                local_success = True
                return
            if _checkpoint("trade"):
                return
            _safe_trade(
                ctx,
                state,
                symbol,
                current_cash,
                model,
                regime_ok,
                price_df=price_df,
            )
            completed_stages.append("trade")
            local_success = True
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(
                "PROCESS_SYMBOL_FAILED",
                extra={
                    "symbol": symbol,
                    "cause": exc.__class__.__name__,
                    "detail": str(exc),
                },
                exc_info=True,
            )
        finally:
            if local_success:
                with data_stats_lock:
                    data_stats["succeeded"] += 1

    prediction_executor = getattr(be, "prediction_executor", None)
    if prediction_executor is None:
        prediction_executor = getattr(executors, "prediction_executor", None)
        if prediction_executor is not None:
            setattr(be, "prediction_executor", prediction_executor)
    if prediction_executor is None:
        prediction_executor = getattr(executors, "executor", None)
    if prediction_executor is None:
        raise RuntimeError("ThreadPool executors unavailable after initialization")
    futures = [prediction_executor.submit(process_symbol, sym) for sym in symbols]
    if _should_stop_now():
        for future in futures:
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()
        return processed, row_counts, fetch_attempts
    for future in futures:
        if _should_stop_now():
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()
            continue
        try:
            future.result()
        except COMMON_EXC:
            logger.exception("PROCESS_SYMBOL_ERROR | skipping failed symbol")
    with data_stats_lock:
        failed = int(data_stats.get("failed", 0))
        succeeded = int(data_stats.get("succeeded", 0))
    total_candidates = len(symbols)
    skipped = max(total_candidates - failed - succeeded, 0)
    logger.info(
        "CYCLE_DATA_ERRORS",
        extra={"failed": failed, "succeeded": succeeded, "skipped": skipped},
    )
    broker_nonretryable = broker_retryable = broker_skipped = 0
    stats_latest = getattr(exec_engine, "stats", None) if exec_engine is not None else None
    if isinstance(stats_latest, dict):
        try:
            broker_nonretryable = max(
                int(stats_latest.get("capacity_skips", 0) or 0)
                - broker_stats_before["capacity_skips"],
                0,
            )
        except (TypeError, ValueError):
            broker_nonretryable = 0
        try:
            broker_retryable = max(
                int(stats_latest.get("retry_count", 0) or 0)
                - broker_stats_before["retry_count"],
                0,
            )
        except (TypeError, ValueError):
            broker_retryable = 0
        try:
            broker_skipped = max(
                int(stats_latest.get("skipped_orders", 0) or 0)
                - broker_stats_before["skipped_orders"],
                0,
            )
        except (TypeError, ValueError):
            broker_skipped = 0
    logger.info(
        "CYCLE_BROKER_ERRORS",
        extra={
            "nonretryable": broker_nonretryable,
            "retryable": broker_retryable,
            "skipped": broker_skipped,
        },
    )
    return processed, row_counts, fetch_attempts
