"""Strategy allocation with signal confirmation and hold protection."""

from __future__ import annotations

import copy
import logging
import re
from typing import Any

try:  # Local import to avoid cycles during docs builds
    from ai_trading.strategies.base import StrategySignal
except Exception:  # pragma: no cover - optional at import time
    StrategySignal = tuple()  # type: ignore[assignment]

from ai_trading.config.management import TradingConfig, get_env, get_trading_config
from ai_trading.core.enums import OrderSide

logger = logging.getLogger(__name__)
_missing_attr_warned: set[str] = set()
_invalid_value_warned: set[str] = set()
_GAP_RATIO_RE = re.compile(r"gap_ratio=([0-9.]+)%")
_LIMIT_RE = re.compile(r"limit=([0-9.]+)%")
def _resolve_allocator_eps() -> float:
    try:
        value = get_env("ALLOCATOR_EPS", None, cast=float)
    except Exception:
        value = None
    try:
        if value is None:
            return 1e-6
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 1e-6


EPS = _resolve_allocator_eps()


def _resolve_conf_threshold(cfg) -> float:
    """Return a confidence threshold from various config attribute names."""
    for name in ("score_confidence_min", "min_confidence", "conf_threshold"):
        v = getattr(cfg, name, None)
        if isinstance(v, (int, float)) and 0 <= float(v) <= 1:
            return float(v)
    return 0.6


class StrategyAllocator:
    """Allocator implementing signal confirmation and confidence gating."""

    def __init__(self, config: Any | None = None) -> None:
        base_config = config if config is not None else get_trading_config()
        self.config = copy.deepcopy(base_config)
        self._ensure_config_attributes()
        self.signal_history: dict[str, list[float]] = {}
        self.last_direction: dict[str, str] = {}
        self.last_confidence: dict[str, float] = {}
        self.hold_protect: dict[str, int] = {}

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_gap_ratio_limit(self) -> float:
        candidates = (
            getattr(self.config, "gap_ratio_limit", None),
            getattr(self.config, "data_max_gap_ratio_intraday", None),
        )
        for candidate in candidates:
            value = self._coerce_float(candidate)
            if value is not None and value >= 0.0:
                return value
        try:
            env_value = get_env("AI_TRADING_GAP_RATIO_LIMIT", None, cast=float)
        except Exception:
            env_value = None
        value = self._coerce_float(env_value)
        if value is not None and value >= 0.0:
            return value
        return 0.005

    def _fallback_gap_limit(self, primary_limit: float) -> float:
        fallback_candidate = self._coerce_float(
            getattr(self.config, "data_max_gap_ratio_intraday", None)
        )
        candidates = [primary_limit * 1.5, primary_limit + 0.005]
        if fallback_candidate is not None:
            candidates.append(fallback_candidate)
        return max(candidates)

    def _extract_gap_ratio(
        self, metadata: dict[str, Any] | None, reason: str | None
    ) -> float | None:
        if metadata is not None:
            direct = self._coerce_float(metadata.get("gap_ratio"))
            if direct is not None:
                return direct
            coverage = metadata.get("_coverage_meta") or metadata.get("coverage_meta")
            if isinstance(coverage, dict):
                coverage_ratio = self._coerce_float(coverage.get("gap_ratio"))
                if coverage_ratio is not None:
                    return coverage_ratio
        if reason:
            match = _GAP_RATIO_RE.search(reason)
            if match:
                try:
                    return float(match.group(1)) / 100.0
                except (TypeError, ValueError):
                    return None
        return None

    def _extract_limit_from_reason(self, reason: str | None) -> float | None:
        if not reason:
            return None
        match = _LIMIT_RE.search(reason)
        if match:
            try:
                return float(match.group(1)) / 100.0
            except (TypeError, ValueError):
                return None
        return None

    def _normalize_signal_reliability(self, signal: Any) -> None:
        metadata_raw = getattr(signal, "metadata", None)
        metadata = metadata_raw if isinstance(metadata_raw, dict) else None
        if metadata is None:
            return

        reliable = metadata.get("price_reliable", getattr(signal, "price_reliable", True))
        reason = metadata.get(
            "price_reliable_reason", getattr(signal, "price_reliable_reason", None)
        )
        fallback_provider = metadata.get("fallback_provider") or metadata.get(
            "data_provider"
        )
        fallback_label = None
        if isinstance(fallback_provider, str):
            fallback_label = fallback_provider.strip().lower() or None

        gap_ratio = self._extract_gap_ratio(metadata, reason)
        limit_from_reason = self._extract_limit_from_reason(reason)
        primary_limit = (
            limit_from_reason
            if limit_from_reason is not None
            else self._resolve_gap_ratio_limit()
        )
        fallback_limit = self._fallback_gap_limit(primary_limit)

        fallback_gap_relaxed = bool(metadata.get("fallback_gap_relaxed"))
        allow_fallback = fallback_gap_relaxed or (
            fallback_label is not None and not fallback_label.startswith("alpaca")
        )

        if not reliable and allow_fallback:
            if gap_ratio is None or gap_ratio <= fallback_limit:
                metadata["price_reliable"] = True
                metadata.pop("price_reliable_reason", None)
                metadata.setdefault("price_reliable_override", True)
                if gap_ratio is not None:
                    metadata["gap_ratio"] = gap_ratio
                logger.info(
                    "FALLBACK_PRICE_ACCEPTED",
                    extra={
                        "symbol": getattr(signal, "symbol", "?"),
                        "gap_ratio": gap_ratio,
                        "limit": fallback_limit,
                        "fallback_provider": fallback_provider,
                    },
                )

    def replace_config(self, **changes: Any) -> TradingConfig:
        """Return new TradingConfig with ``changes`` applied and set it."""
        if isinstance(self.config, TradingConfig):
            updated = self.config.to_dict()
            updated.update(changes)
            new_cfg = TradingConfig(**updated)
        else:
            new_cfg = copy.deepcopy(self.config)
            for key, value in changes.items():
                try:
                    setattr(new_cfg, key, value)
                except Exception:
                    logger.warning("Failed to set config attribute %s", key, exc_info=True)
        self.config = copy.deepcopy(new_cfg)
        self._ensure_config_attributes()
        return self.config

    def _ensure_config_attributes(self) -> None:
        """Ensure config has required attributes with sensible defaults."""
        required = {
            "signal_confirmation_bars": 2,
            "delta_threshold": 0.02,
            "min_confidence": 0.6,
        }
        for attr, default in required.items():
            if not hasattr(self.config, attr) or getattr(self.config, attr, None) is None:
                if attr not in _missing_attr_warned:
                    logger.warning("Config missing attribute %s, setting default: %s", attr, default)
                    _missing_attr_warned.add(attr)
                if isinstance(self.config, TradingConfig):
                    try:
                        self.config._values[attr] = default  # type: ignore[attr-defined]
                    except Exception:
                        logger.debug("Failed to set TradingConfig value for %s", attr, exc_info=True)
                else:
                    # Fallback for SimpleNamespace or other objects
                    try:
                        object.__setattr__(self.config, attr, default)
                    except Exception:
                        logger.debug("Failed to set config attribute %s", attr, exc_info=True)

    def select_signals(self, signals_by_strategy: dict[str, list[Any]]) -> list[Any]:
        """Compatibility wrapper for allocate()."""
        return self.allocate(signals_by_strategy)

    def allocate(self, signals_by_strategy: dict[str, list[Any]]) -> list[Any]:
        if not signals_by_strategy or not isinstance(signals_by_strategy, dict):
            logger.debug("Allocate called with invalid signals_by_strategy: %s", type(signals_by_strategy))
            return []

        total_signals = sum(len(signals) for signals in signals_by_strategy.values())
        logger.debug(
            "Allocate called with %s strategies, %s total signals",
            len(signals_by_strategy),
            total_signals,
        )
        logger.debug(
            "Current config: signal_confirmation_bars=%s, min_confidence=%s, delta_threshold=%s",
            self.config.signal_confirmation_bars,
            getattr(self.config, "min_confidence", None),
            self.config.delta_threshold,
        )
        logger.debug("Current signal_history state: %s", dict(self.signal_history))

        confirmed = self._confirm_signals(signals_by_strategy)
        confirmed_count = sum(len(sigs) for sigs in confirmed.values())
        logger.debug(
            "Signal confirmation produced %s confirmed signals from %s input signals",
            confirmed_count,
            total_signals,
        )

        threshold = _resolve_conf_threshold(self.config)
        gated: dict[str, list[Any]] = {}
        for strat, sigs in (confirmed or {}).items():
            kept: list[Any] = []
            for s in sigs or []:
                try:
                    c = float(getattr(s, "confidence", 0.0))
                except (TypeError, ValueError):
                    c = 0.0
                if c >= threshold:
                    kept.append(s)
                else:
                    logger.info(
                        "CONFIDENCE_DROP",
                        extra={
                            "strategy": strat,
                            "symbol": getattr(s, "symbol", "?"),
                            "confidence": c,
                            "threshold": threshold,
                        },
                    )
            if kept:
                gated[strat] = kept

        result = self._allocate_confirmed(gated)
        logger.debug("Final allocation returning %s signals", len(result))
        if result:
            logger.debug("Returned signal symbols: %s", [s.symbol for s in result])
        return result

    def _confirm_signals(self, signals_by_strategy: dict[str, list[Any]]) -> dict[str, list[Any]]:
        confirmed: dict[str, list[Any]] = {}
        for strategy, signals in signals_by_strategy.items():
            if not isinstance(signals, list):
                logger.warning("Strategy %s has non-list signals: %s", strategy, type(signals))
                confirmed[strategy] = []
                continue

            logger.debug("Processing strategy %s with %s signals", strategy, len(signals))
            confirmed[strategy] = []

            for s in signals:
                if not hasattr(s, "symbol") or not hasattr(s, "side") or not hasattr(s, "confidence"):
                    logger.warning("Invalid signal object missing required attributes: %s", s)
                    continue
                if not s.symbol or not isinstance(s.symbol, str):
                    logger.warning("Invalid signal symbol: %s", s.symbol)
                    continue
                # Normalize side to canonical strings and accept common aliases
                side_value = getattr(s, "side", "")
                side_is_enum = isinstance(side_value, OrderSide)
                side_is_enum = bool(getattr(s, "_side_is_enum", side_is_enum))
                if side_is_enum:
                    s.side = "buy" if side_value is OrderSide.BUY else "sell"
                else:
                    try:
                        side_norm = str(side_value).strip().lower()
                    except Exception:
                        side_norm = ""
                    # Map aliases
                    if side_norm in ("long", "buy", "enter_long"):
                        s.side = "buy"
                    elif side_norm in ("short", "sell", "sell_short", "enter_short"):
                        s.side = "sell"
                    else:
                        logger.warning("Invalid signal side: %s", getattr(s, "side", side_norm))
                        continue

                self._normalize_signal_reliability(s)

                try:
                    original = float(s.confidence)
                    conf = original
                    if conf < 0 or conf > 1:
                        if conf > 1:
                            import math

                            normalized = (math.tanh(conf - 1) + 1) / 2
                            normalized = 0.5 + normalized * 0.5
                            conf = normalized
                        else:
                            conf = max(0.01, conf)
                        conf = max(0.01, min(1.0, conf))
                        s.confidence = conf
                        logger.info(
                            "CONFIDENCE_NORMALIZED | symbol=%s original=%.4f normalized=%.4f",
                            s.symbol,
                            original,
                            conf,
                        )
                except (TypeError, ValueError):
                    logger.warning("Invalid signal confidence (not numeric): %s", getattr(s, "confidence", None))
                    continue

                logger.debug("Signal: %s, confidence: %s", s.symbol, s.confidence)
                key = f"{s.symbol}_{s.side}"
                self.signal_history.setdefault(key, []).append(s.confidence)

                bars = getattr(self.config, "signal_confirmation_bars", 2)
                if bars is None or not isinstance(bars, int) or bars < 1:
                    if "signal_confirmation_bars" not in _invalid_value_warned:
                        logger.warning("Invalid signal_confirmation_bars: %s, using default 2", bars)
                        _invalid_value_warned.add("signal_confirmation_bars")
                    bars = 2
                self.signal_history[key] = self.signal_history[key][-bars:]

                history = self.signal_history[key]

                threshold = getattr(self.config, "min_confidence", 0.6)
                if threshold is None or not isinstance(threshold, (int, float)):
                    if "min_confidence" not in _invalid_value_warned:
                        logger.warning(
                            "Invalid min_confidence threshold: %s, using default 0.6",
                            threshold,
                        )
                        _invalid_value_warned.add("min_confidence")
                    threshold = 0.6

                if len(history) >= bars:
                    if not history:
                        logger.warning("Empty signal history for %s, skipping confirmation", key)
                        continue
                    try:
                        avg_conf = sum(history) / len(history)
                        if not isinstance(avg_conf, (int, float)) or avg_conf < 0:
                            logger.warning("Invalid average confidence %s for %s", avg_conf, s.symbol)
                            continue
                    except Exception as e:  # noqa: BLE001 - broad to log
                        logger.warning("Error calculating average confidence for %s: %s", s.symbol, e)
                        continue

                    logger.debug(
                        "Signal confirmation check: %s, history=%s, avg_conf=%.4f, threshold=%.4f, bars_required=%s, bars_available=%s",
                        s.symbol,
                        history,
                        avg_conf,
                        threshold,
                        bars,
                        len(history),
                    )

                    if avg_conf >= threshold:
                        confirmed_signal = copy.deepcopy(s)
                        confirmed_signal.confidence = avg_conf
                        confirmed[strategy].append(confirmed_signal)
                        logger.debug(
                            "Signal CONFIRMED: %s with avg_conf: %.4f >= threshold: %.4f",
                            s.symbol,
                            avg_conf,
                            threshold,
                        )
                    else:
                        logger.debug(
                            "Signal REJECTED: %s, avg_conf: %.4f < threshold: %.4f",
                            s.symbol,
                            avg_conf,
                            threshold,
                        )
                else:
                    logger.debug(
                        "Signal NOT READY: %s, history length: %s/%s",
                        s.symbol,
                        len(history),
                        bars,
                    )
                    logger.debug("  Current history for %s: %s", key, history)
                    if isinstance(StrategySignal, type) and isinstance(s, StrategySignal) and side_is_enum:
                        if s.confidence >= threshold:
                            confirmed_signal = copy.deepcopy(s)
                            confirmed_signal.confidence = float(s.confidence)
                            confirmed[strategy].append(confirmed_signal)
                            logger.debug(
                                "StrategySignal pre-confirmed: %s confidence=%.4f threshold=%.4f",
                                s.symbol,
                                confirmed_signal.confidence,
                                threshold,
                            )
                        else:
                            logger.debug(
                                "StrategySignal below threshold for pre-confirmation: %s (%.4f < %.4f)",
                                s.symbol,
                                float(s.confidence),
                                threshold,
                            )
        return confirmed

    def _allocate_confirmed(self, confirmed_signals: dict[str, list[Any]]) -> list[Any]:
        final: list[Any] = []
        all_signals: list[Any] = []
        for strategy, signals in confirmed_signals.items():
            for s in signals:
                s.strategy = strategy
                all_signals.append(s)

        prev_conf = self.last_confidence.copy()
        pending_conf: dict[str, float] = {}

        for s in sorted(all_signals, key=lambda x: (x.symbol, -x.confidence)):
            last_dir = self.last_direction.get(s.symbol)
            last_conf = prev_conf.get(s.symbol, 0.0)
            if last_dir and last_dir != s.side:
                if s.side == "sell" and self.hold_protect.get(s.symbol, 0) > 0:
                    remaining = self.hold_protect.get(s.symbol, 0)
                    logger.info(
                        "HOLD_PROTECT_ACTIVE",
                        extra={"symbol": s.symbol, "remaining_cycles": remaining},
                    )
                    self.hold_protect[s.symbol] = max(0, remaining - 1)
                    continue

            delta = abs(s.confidence - last_conf) if last_conf else s.confidence
            threshold_raw = getattr(self.config, "delta_threshold", 0.0)
            try:
                threshold = float(threshold_raw) if threshold_raw is not None else 0.0
            except (TypeError, ValueError):
                threshold = 0.0

            if delta < threshold and last_dir == s.side:
                logger.info(
                    "SIGNAL_SKIPPED_DELTA",
                    extra={
                        "symbol": s.symbol,
                        "delta": delta,
                        "threshold": threshold,
                    },
                )
                continue

            self.last_direction[s.symbol] = s.side
            pending_conf[s.symbol] = s.confidence
            if s.side == "buy":
                self.hold_protect[s.symbol] = 4
            final.append(s)

        if pending_conf:
            self.last_confidence.update(pending_conf)

        self._assign_portfolio_weights(final)
        return final

    def _assign_portfolio_weights(self, signals: list[Any]) -> None:
        """Assign portfolio weights to signals based on confidence."""
        if not signals:
            return

        max_total = getattr(self.config, "exposure_cap_aggressive", 0.88)
        buys = [s for s in signals if s.side == "buy"]
        sells = [s for s in signals if s.side == "sell"]

        if buys:
            total_conf = sum(s.confidence for s in buys)
            if total_conf > 0:
                available = max_total
                for signal in buys:
                    weight = signal.confidence / total_conf * available
                    max_individual = min(0.25, available / len(buys) * 1.5)
                    signal.weight = min(weight, max_individual)
                    logger.debug(
                        "Assigned weight %.3f to %s (confidence=%.3f)",
                        signal.weight,
                        signal.symbol,
                        signal.confidence,
                    )
            else:
                equal = min(max_total / len(buys), 0.15)
                for signal in buys:
                    signal.weight = equal
                    logger.debug("Assigned equal weight %.3f to %s", equal, signal.symbol)

        for signal in sells:
            max_exit = min(0.25, max_total / max(len(sells), 1))
            signal.weight = max_exit
            logger.debug("Assigned exit weight %.3f to %s", signal.weight, signal.symbol)

        self._scale_buy_weights(buys, max_total)

        # When provider safe-mode is active, clarify that allocation is theoretical only.
        total_buy_weight = sum(s.weight for s in buys)
        try:
            from ai_trading.data.provider_monitor import is_safe_mode_active
            degraded = bool(is_safe_mode_active())
        except Exception:
            degraded = False
        message = (
            "Portfolio allocation (theoretical, degraded feed): %s buys (total weight: %.3f), %s sells"
            if degraded
            else "Portfolio allocation: %s buys (total weight: %.3f), %s sells"
        )
        logger.info(
            message,
            len(buys),
            total_buy_weight,
            len(sells),
        )

    def _scale_buy_weights(self, buys: list[Any], cap: float) -> bool:
        """Scale buy weights proportionally when exceeding ``cap``."""

        if not buys:
            return False
        try:
            total_before = sum(max(float(getattr(sig, "weight", 0.0)), 0.0) for sig in buys)
        except (TypeError, ValueError):
            total_before = sum(max(float(sig.weight), 0.0) for sig in buys if hasattr(sig, "weight"))
        if total_before <= 0 or total_before <= cap + EPS:
            return False
        scale = cap / total_before
        scaled_weights: list[tuple[Any, float]] = []
        for sig in buys:
            try:
                weight = float(getattr(sig, "weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            scaled = max(weight, 0.0) * scale
            scaled_weights.append((sig, scaled))
        total_scaled = sum(weight for _, weight in scaled_weights)
        cap_target = min(cap, total_scaled) if total_scaled > 0 else 0.0
        normalised: list[tuple[Any, float]] = []
        if total_scaled > 0:
            ratio = cap_target / total_scaled
            for sig, scaled in scaled_weights:
                normalised.append((sig, scaled * ratio))
        else:
            normalised = [(sig, 0.0) for sig, _ in scaled_weights]

        for sig, scaled in normalised:
            sig.weight = round(max(scaled, 0.0), 7)

        total_after = sum(getattr(sig, "weight", 0.0) for sig in buys)
        if total_after > cap + EPS:
            excess = total_after - cap
            for sig, _ in sorted(normalised, key=lambda item: getattr(item[0], "weight", 0.0), reverse=True):
                if excess <= EPS:
                    break
                current_weight = getattr(sig, "weight", 0.0)
                reduction = min(current_weight, excess)
                sig.weight = round(max(current_weight - reduction, 0.0), 7)
                excess -= reduction
            total_after = sum(getattr(sig, "weight", 0.0) for sig in buys)
        if total_after > cap + EPS:
            largest = max(buys, key=lambda s: getattr(s, "weight", 0.0))
            adjustment = total_after - cap
            largest.weight = round(max(getattr(largest, "weight", 0.0) - adjustment, 0.0), 7)
            total_after = sum(getattr(sig, "weight", 0.0) for sig in buys)

        total_after = min(total_after, cap + EPS)
        logger.info(
            "ALLOCATION_SCALED | buys=%d total_weight_before=%.6f total_weight_after=%.6f cap=%.6f method=proportional",
            len(buys),
            total_before,
            total_after,
            cap,
        )
        return True

    def update_reward(self, strategy: str, reward: float) -> None:
        """Update reward for a strategy (compatibility hook for tests)."""
        logger.info("Strategy %s reward updated: %s", strategy, reward)


__all__ = ["StrategyAllocator"]
