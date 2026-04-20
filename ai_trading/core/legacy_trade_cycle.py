"""Legacy trade decision flow extracted from ``bot_engine.py``."""
from __future__ import annotations

import importlib
from datetime import UTC, datetime
from typing import Any, Callable, Mapping, Sequence

from ai_trading.contracts import Bar, bar_from_frame


def execute_legacy_trade_logic(
    ctx: Any,
    state: Any,
    symbol: str,
    balance: float,
    model: Any,
    regime_ok: bool,
    *,
    price_df: Any = None,
    now_provider: Callable[[], datetime] | None = None,
) -> bool:
    """Run the legacy non-netting per-symbol trade flow."""

    be = importlib.import_module("ai_trading.core.bot_engine")
    feat_df: Any = None

    def _decision_bar() -> Bar | None:
        for frame in (feat_df, price_df):
            bar = bar_from_frame(
                symbol,
                frame,
                timeframe="1Min",
                provider="alpaca",
            )
            if bar is not None:
                return bar
        return None

    def _record_legacy_decision(
        *,
        accepted: bool,
        gates: Sequence[str],
        reasons: Sequence[str] | None = None,
        event: str,
        signal_side: str = "hold",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        recorder = getattr(state, "_legacy_decision_recorder", None)
        if recorder is None or not hasattr(recorder, "record"):
            return
        market_bar = _decision_bar()
        freshness: float | None = None
        if market_bar is not None and isinstance(market_bar.ts, datetime):
            freshness = max((datetime.now(UTC) - market_bar.ts).total_seconds(), 0.0)
        try:
            recorder.record(
                symbol=symbol,
                market_bar=market_bar,
                bar_ts=market_bar.ts if market_bar is not None else None,
                signal_side=signal_side,
                final_score=0.0,
                confidence=0.0,
                strategy_id=None,
                accepted=accepted,
                gates=gates,
                reasons=reasons or gates,
                provider=market_bar.provider if market_bar is not None else None,
                feed=market_bar.feed if market_bar is not None else None,
                reference_price=market_bar.close if market_bar is not None else None,
                event=event,
                data_freshness_sec=freshness,
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
            )
        except Exception:
            be.logger.debug("LEGACY_DECISION_RECORD_FAILED", exc_info=True)

    if not be.pre_trade_checks(ctx, state, symbol, balance, regime_ok):
        be.logger.debug("SKIP_PRE_TRADE_CHECKS", extra={"symbol": symbol})
        _record_legacy_decision(
            accepted=False,
            gates=["PRE_TRADE_CHECKS_FAILED"],
            event="legacy_pretrade_block",
        )
        return False

    provider_enabled = True
    primary_provider_fn = getattr(be.data_fetcher_module, "is_primary_provider_enabled", None)
    if callable(primary_provider_fn):
        try:
            provider_enabled = bool(primary_provider_fn())
        except be.COMMON_EXC as exc:  # pragma: no cover - defensive logging
            be.logger.warning(
                "PRIMARY_PROVIDER_STATUS_ERROR",
                extra={"symbol": symbol, "detail": str(exc)},
            )
            provider_enabled = True
    if not provider_enabled:
        be._mark_primary_provider_fallback(
            state, symbol, reason="primary_provider_disabled"
        )
        be.logger.warning(
            "PRIMARY_PROVIDER_DEGRADED",
            extra={"symbol": symbol, "provider": "alpaca"},
        )
    else:
        be._clear_primary_provider_fallback(state, symbol, provider="alpaca")

    if be.is_safe_mode_active():
        reason = be.safe_mode_reason() or "provider_safe_mode"
        if be._safe_mode_blocks_trading():
            be.logger.warning(
                "SAFE_MODE_BLOCK",
                extra={
                    "symbol": symbol,
                    "reason": reason,
                    "block_reason": "provider_disabled",
                },
            )
            _record_legacy_decision(
                accepted=False,
                gates=["SAFE_MODE_BLOCK"],
                reasons=[reason],
                event="legacy_safe_mode_block",
            )
            return False
        setattr(state, "prefer_backup_quotes", True)
        be._mark_ctx_degraded(ctx, reason)
        be._log_safe_mode_continue(ctx, stage="trade_logic", reason=reason, symbol=symbol)

    with be.StageTimer(be.logger, "DATA_FETCH_TOTAL_MS", symbol=symbol):
        raw_df, feat_df, skip_flag = be._fetch_feature_data(
            ctx, state, symbol, price_df=price_df
        )
    if be.is_safe_mode_active() and be._safe_mode_blocks_trading():
        reason = be.safe_mode_reason() or "provider_safe_mode"
        be.logger.warning(
            "SAFE_MODE_BLOCK",
            extra={
                "symbol": symbol,
                "reason": reason,
                "block_reason": "provider_disabled_midcycle",
            },
        )
        return False
    if feat_df is None:
        _record_legacy_decision(
            accepted=False,
            gates=["FEATURE_DATA_UNAVAILABLE" if skip_flag else "DATA_FETCH_UNAVAILABLE"],
            event="legacy_feature_block",
        )
        return bool(skip_flag) if skip_flag is not None else False

    default_feature_values = {
        "macd": 0.0,
        "atr": 0.0,
        "vwap": 0.0,
        "macds": 0.0,
        "sma_50": 0.0,
        "sma_200": 0.0,
        "rsi": 50.0,
        "ichimoku_conv": 0.0,
        "ichimoku_base": 0.0,
        "stochrsi": 0.5,
    }
    for col, neutral_value in default_feature_values.items():
        if col not in feat_df.columns:
            feat_df[col] = neutral_value

    feature_names = be._model_feature_names(model)
    missing = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        be.logger.debug(
            f"Feature snapshot for {symbol}: macd={feat_df['macd'].iloc[-1]}, atr={feat_df['atr'].iloc[-1]}, vwap={feat_df['vwap'].iloc[-1]}, macds={feat_df['macds'].iloc[-1]}, sma_50={feat_df['sma_50'].iloc[-1]}, sma_200={feat_df['sma_200'].iloc[-1]}"
        )
        be.logger.info("SKIP_MISSING_FEATURES | symbol=%s  missing=%s", symbol, missing)
        _record_legacy_decision(
            accepted=False,
            gates=["MISSING_FEATURES"],
            reasons=missing,
            event="legacy_missing_features",
        )
        return True

    try:
        final_score, conf, strat = be._evaluate_trade_signal(
            ctx, state, feat_df, symbol, model
        )
    except ValueError as exc:
        be.logger.info(
            "SKIP_SIGNAL_INVALID",
            extra={"symbol": symbol, "reason": str(exc)},
        )
        _record_legacy_decision(
            accepted=False,
            gates=["SIGNAL_INVALID"],
            reasons=[str(exc)],
            event="legacy_signal_invalid",
        )
        return True
    if be.pd.isna(final_score) or be.pd.isna(conf):
        be.logger.warning(f"Skipping {symbol}: model returned NaN prediction")
        _record_legacy_decision(
            accepted=False,
            gates=["SIGNAL_NAN"],
            event="legacy_signal_nan",
        )
        return True

    current_qty = be._current_qty(ctx, symbol)
    if current_qty == 0:
        entry_times = getattr(state, "position_entry_times", None)
        if isinstance(entry_times, dict):
            entry_times.pop(symbol, None)
        be._reset_reversal_signal_streak(state, symbol)

    now_fn = now_provider or (lambda: datetime.now(UTC))
    now = now_fn()
    signal = "buy" if final_score > 0 else "sell" if final_score < 0 else "hold"

    if be._exit_positions_if_needed(
        ctx, state, symbol, feat_df, final_score, conf, current_qty
    ):
        _record_legacy_decision(
            accepted=False,
            gates=["POSITION_EXIT_MANAGED"],
            reasons=["position_exit_managed"],
            event="legacy_position_exit",
            signal_side=signal,
            metadata={
                "final_score": float(final_score),
                "confidence": float(conf),
            },
        )
        return True

    with be.trade_cooldowns_lock:
        cd_ts = state.trade_cooldowns.get(symbol)
    if cd_ts and (now - cd_ts).total_seconds() < be.get_trade_cooldown_min() * 60:
        prev = state.last_trade_direction.get(symbol)
        if prev and (
            (prev == "buy" and signal == "sell") or (prev == "sell" and signal == "buy")
        ):
            be.logger.info("SKIP_REVERSED_SIGNAL", extra={"symbol": symbol})
            _record_legacy_decision(
                accepted=False,
                gates=["REVERSED_SIGNAL_COOLDOWN"],
                event="legacy_reversed_signal_block",
                signal_side=signal,
                metadata={
                    "final_score": float(final_score),
                    "confidence": float(conf),
                },
            )
            return True
        be.logger.debug("SKIP_COOLDOWN", extra={"symbol": symbol})
        _record_legacy_decision(
            accepted=False,
            gates=["TRADE_COOLDOWN_ACTIVE"],
            event="legacy_trade_cooldown",
            signal_side=signal,
            metadata={
                "final_score": float(final_score),
                "confidence": float(conf),
            },
        )
        return True

    if be._check_trade_frequency_limits(state, symbol, now):
        be.logger.info("SKIP_FREQUENCY_LIMIT", extra={"symbol": symbol})
        _record_legacy_decision(
            accepted=False,
            gates=["TRADE_FREQUENCY_LIMIT"],
            event="legacy_frequency_limit",
            signal_side=signal,
            metadata={
                "final_score": float(final_score),
                "confidence": float(conf),
            },
        )
        return True

    alpha_decay_guard: dict[str, Any] | None = None
    if current_qty == 0:
        alpha_decay_guard = be._alpha_decay_entry_guard(state, symbol, now)
        if alpha_decay_guard.get("blocked"):
            be.logger.info(
                "ENTRY_BLOCKED_ALPHA_DECAY",
                extra={
                    "symbol": symbol,
                    "trades_in_window": alpha_decay_guard.get("trades_in_window", 0),
                    "window_minutes": alpha_decay_guard.get("window_minutes", 0),
                    "max_trades_window": alpha_decay_guard.get("max_trades_window", 0),
                },
            )
            _record_legacy_decision(
                accepted=False,
                gates=["ENTRY_BLOCKED_ALPHA_DECAY"],
                event="legacy_alpha_decay_block",
                signal_side=signal,
                metadata={
                    "final_score": float(final_score),
                    "confidence": float(conf),
                },
            )
            return True

    local_threshold = max(be.get_buy_threshold(), be.get_conf_threshold())
    meta_capped = bool(getattr(ctx.signal_manager, "meta_confidence_capped", False))
    if meta_capped:
        cap_limit = be._metafallback_confidence_cap()
        try:
            local_threshold = min(float(local_threshold), float(cap_limit))
        except Exception:
            local_threshold = min(local_threshold, cap_limit)
        local_threshold = max(local_threshold, be.get_conf_threshold())
    fallback_confidence_bonus = be.get_fallback_entry_confidence_bonus()
    if fallback_confidence_bonus > 0:
        quality = be._ensure_data_quality_bucket(state).get(symbol, {})
        if isinstance(quality, be.MappingABC):
            using_fallback_provider = bool(quality.get("using_fallback_provider"))
            stale_data = bool(quality.get("stale_data"))
            missing_ohlcv = bool(quality.get("missing_ohlcv"))
            if using_fallback_provider or stale_data or missing_ohlcv:
                threshold_before = local_threshold
                local_threshold = min(1.0, local_threshold + fallback_confidence_bonus)
                be.logger.info(
                    "ENTRY_THRESHOLD_RAISED_DEGRADED_DATA",
                    extra={
                        "symbol": symbol,
                        "threshold_before": threshold_before,
                        "threshold_after": local_threshold,
                        "confidence_bonus": fallback_confidence_bonus,
                        "using_fallback_provider": using_fallback_provider,
                        "stale_data": stale_data,
                        "missing_ohlcv": missing_ohlcv,
                    },
                )

    if current_qty == 0 and alpha_decay_guard is not None:
        threshold_bump = float(alpha_decay_guard.get("threshold_bump", 0.0) or 0.0)
        if threshold_bump > 0:
            threshold_before = local_threshold
            local_threshold = min(1.0, local_threshold + threshold_bump)
            be.logger.info(
                "ENTRY_THRESHOLD_RAISED_ALPHA_DECAY",
                extra={
                    "symbol": symbol,
                    "threshold_before": threshold_before,
                    "threshold_after": local_threshold,
                    "threshold_bump": threshold_bump,
                    "trades_in_window": alpha_decay_guard.get("trades_in_window", 0),
                    "window_minutes": alpha_decay_guard.get("window_minutes", 0),
                    "start_trades": alpha_decay_guard.get("start_trades", 0),
                },
            )

    feed_reliability = be._get_symbol_feed_reliability(symbol)
    if current_qty == 0 and bool(feed_reliability.get("active")):
        reliability_threshold_bonus = be._safe_float(feed_reliability.get("threshold_bonus"))
        if reliability_threshold_bonus is not None and reliability_threshold_bonus > 0.0:
            threshold_before = local_threshold
            local_threshold = min(1.0, local_threshold + reliability_threshold_bonus)
            be.logger.info(
                "ENTRY_THRESHOLD_RAISED_FEED_RELIABILITY",
                extra={
                    "symbol": symbol,
                    "threshold_before": threshold_before,
                    "threshold_after": local_threshold,
                    "reliability_score": be._safe_float(feed_reliability.get("score")),
                    "sample_count": int(feed_reliability.get("sample_count", 0) or 0),
                    "threshold_bonus": reliability_threshold_bonus,
                },
            )

    long_entry_candidate = final_score > 0 and conf >= local_threshold and current_qty == 0
    short_entry_candidate = final_score < 0 and conf >= local_threshold and current_qty == 0
    if (
        current_qty == 0
        and (long_entry_candidate or short_entry_candidate)
        and bool(feed_reliability.get("blocked"))
    ):
        entry_side = "buy" if long_entry_candidate else "sell_short"
        be.logger.info(
            "ENTRY_BLOCKED_FEED_RELIABILITY",
            extra={
                "symbol": symbol,
                "side": entry_side,
                "final_score": final_score,
                "confidence": conf,
                "reliability_score": be._safe_float(feed_reliability.get("score")),
                "sample_count": int(feed_reliability.get("sample_count", 0) or 0),
                "min_score": be._safe_float(feed_reliability.get("min_score")),
            },
        )
        be._reset_entry_flip_signal_streak(state, symbol)
        _record_legacy_decision(
            accepted=False,
            gates=["ENTRY_BLOCKED_FEED_RELIABILITY"],
            event="legacy_feed_reliability_block",
            signal_side=signal,
            metadata={
                "final_score": float(final_score),
                "confidence": float(conf),
            },
        )
        return True
    if current_qty == 0 and not (long_entry_candidate or short_entry_candidate):
        be._reset_entry_flip_signal_streak(state, symbol)

    if long_entry_candidate:
        if not be._entry_flip_confirmation_ready(
            state,
            symbol=symbol,
            candidate_side="long",
            final_score=float(final_score),
            confidence=float(conf),
        ):
            return True
        if not be._entry_expectancy_allowed(
            state,
            symbol=symbol,
            regime=getattr(state, "current_regime", "sideways"),
            side="long",
        ):
            return True
        if not be._profitability_governor_allows_entry(
            state,
            symbol=symbol,
            regime=getattr(state, "current_regime", "sideways"),
            side="long",
        ):
            return True
        blocked, degraded_extra = be._entry_data_degraded(state, symbol)
        if blocked:
            be.logger.warning(
                "ENTRY_BLOCKED_DEGRADED_MINUTE_DATA",
                extra={
                    "symbol": symbol,
                    "side": "buy",
                    "final_score": final_score,
                    "confidence": conf,
                    **degraded_extra,
                },
            )
            _record_legacy_decision(
                accepted=False,
                gates=["ENTRY_BLOCKED_DEGRADED_MINUTE_DATA"],
                event="legacy_degraded_data_block",
                signal_side="buy",
                metadata={
                    "final_score": float(final_score),
                    "confidence": float(conf),
                },
            )
            return True
        if symbol in state.long_positions:
            held = state.position_cache.get(symbol, 0)
            be.logger.info(
                f"Skipping BUY for {symbol} — position already LONG {held} shares"
            )
            _record_legacy_decision(
                accepted=False,
                gates=["POSITION_ALREADY_LONG"],
                event="legacy_position_already_held",
                signal_side="buy",
                metadata={"held_qty": held},
            )
            return True
        return bool(
            be._enter_long(
            ctx, state, symbol, balance, feat_df, final_score, conf, strat
            )
        )

    if short_entry_candidate:
        if not be._entry_flip_confirmation_ready(
            state,
            symbol=symbol,
            candidate_side="short",
            final_score=float(final_score),
            confidence=float(conf),
        ):
            return True
        if not be._entry_expectancy_allowed(
            state,
            symbol=symbol,
            regime=getattr(state, "current_regime", "sideways"),
            side="short",
        ):
            return True
        if not be._profitability_governor_allows_entry(
            state,
            symbol=symbol,
            regime=getattr(state, "current_regime", "sideways"),
            side="short",
        ):
            return True
        blocked, degraded_extra = be._entry_data_degraded(state, symbol)
        if blocked:
            be.logger.warning(
                "ENTRY_BLOCKED_DEGRADED_MINUTE_DATA",
                extra={
                    "symbol": symbol,
                    "side": "sell",
                    "final_score": final_score,
                    "confidence": conf,
                    **degraded_extra,
                },
            )
            _record_legacy_decision(
                accepted=False,
                gates=["ENTRY_BLOCKED_DEGRADED_MINUTE_DATA"],
                event="legacy_degraded_data_block",
                signal_side="sell",
                metadata={
                    "final_score": float(final_score),
                    "confidence": float(conf),
                },
            )
            return True
        if symbol in state.short_positions:
            held = abs(state.position_cache.get(symbol, 0))
            be.logger.info(
                f"Skipping SELL for {symbol} — position already SHORT {held} shares"
            )
            _record_legacy_decision(
                accepted=False,
                gates=["POSITION_ALREADY_SHORT"],
                event="legacy_position_already_held",
                signal_side="sell",
                metadata={"held_qty": held},
            )
            return True
        return bool(
            be._enter_short(ctx, state, symbol, feat_df, final_score, conf, strat)
        )

    if current_qty != 0:
        atr = feat_df["atr"].iloc[-1]
        _record_legacy_decision(
            accepted=False,
            gates=["MANAGE_EXISTING_POSITION"],
            event="legacy_manage_existing_position",
            signal_side=signal,
            metadata={
                "final_score": float(final_score),
                "confidence": float(conf),
                "current_qty": int(current_qty),
                "atr": float(atr),
            },
        )
        return bool(
            be._manage_existing_position(
                ctx, state, symbol, feat_df, conf, atr, current_qty
            )
        )

    be.logger.info(
        f"SKIP_LOW_OR_NO_SIGNAL | symbol={symbol}  "
        f"final_score={final_score:.4f}  confidence={conf:.4f}  threshold={local_threshold:.4f}"
    )
    _record_legacy_decision(
        accepted=False,
        gates=["LOW_OR_NO_SIGNAL"],
        event="legacy_low_signal_hold",
        signal_side=signal,
        metadata={
            "final_score": float(final_score),
            "confidence": float(conf),
            "threshold": float(local_threshold),
        },
    )
    return True


__all__ = ["execute_legacy_trade_logic"]
