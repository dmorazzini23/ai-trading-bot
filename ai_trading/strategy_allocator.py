"""Strategy allocation with signal confirmation and hold protection."""

from __future__ import annotations

import copy
from dataclasses import replace
import logging
from typing import Any

from ai_trading.config.management import TradingConfig

CONFIG = TradingConfig()
logger = logging.getLogger(__name__)


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
        self.config = copy.deepcopy(config or CONFIG)
        self._ensure_config_attributes()
        self.signal_history: dict[str, list[float]] = {}
        self.last_direction: dict[str, str] = {}
        self.last_confidence: dict[str, float] = {}
        self.hold_protect: dict[str, int] = {}

    def replace_config(self, **changes: Any) -> TradingConfig:
        """Return new TradingConfig with ``changes`` applied and set it."""
        new_cfg = replace(self.config, **changes) if changes else replace(self.config)
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
                logger.warning("Config missing attribute %s, setting default: %s", attr, default)
                # TradingConfig is frozen; bypass immutability for defaults
                object.__setattr__(self.config, attr, default)

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
                if s.side not in ["buy", "sell"]:
                    logger.warning("Invalid signal side: %s", s.side)
                    continue

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
                    logger.warning("Invalid signal_confirmation_bars: %s, using default 2", bars)
                    bars = 2
                self.signal_history[key] = self.signal_history[key][-bars:]

                if len(self.signal_history[key]) >= bars:
                    history = self.signal_history[key]
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

                    threshold = getattr(self.config, "min_confidence", 0.6)
                    if threshold is None or not isinstance(threshold, (int, float)):
                        logger.warning(
                            "Invalid min_confidence threshold: %s, using default 0.6", threshold
                        )
                        threshold = 0.6

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
                        len(self.signal_history[key]),
                        bars,
                    )
                    logger.debug("  Current history for %s: %s", key, self.signal_history[key])
        return confirmed

    def _allocate_confirmed(self, confirmed_signals: dict[str, list[Any]]) -> list[Any]:
        final: list[Any] = []
        all_signals: list[Any] = []
        for strategy, signals in confirmed_signals.items():
            for s in signals:
                s.strategy = strategy
                all_signals.append(s)

        for s in sorted(all_signals, key=lambda x: (x.symbol, -x.confidence)):
            last_dir = self.last_direction.get(s.symbol)
            last_conf = self.last_confidence.get(s.symbol, 0.0)
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
            if delta < self.config.delta_threshold and last_dir == s.side:
                logger.info(
                    "SIGNAL_SKIPPED_DELTA",
                    extra={
                        "symbol": s.symbol,
                        "delta": delta,
                        "threshold": self.config.delta_threshold,
                    },
                )
                continue

            self.last_direction[s.symbol] = s.side
            self.last_confidence[s.symbol] = s.confidence
            if s.side == "buy":
                self.hold_protect[s.symbol] = 4
            final.append(s)

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
                    logger.debug(
                        "Assigned equal weight %.3f to %s", equal, signal.symbol
                    )

        for signal in sells:
            max_exit = min(0.25, max_total / max(len(sells), 1))
            signal.weight = max_exit
            logger.debug(
                "Assigned exit weight %.3f to %s", signal.weight, signal.symbol
            )

        total_buy = sum(s.weight for s in buys)
        if total_buy > max_total:
            logger.warning(
                "Total buy weight %.3f exceeds cap %.3f, scaling down", total_buy, max_total
            )
            scale = max_total / total_buy
            for signal in buys:
                signal.weight *= scale
                logger.debug("Scaled weight to %.3f for %s", signal.weight, signal.symbol)

        logger.info(
            "Portfolio allocation: %s buys (total weight: %.3f), %s sells",
            len(buys),
            sum(s.weight for s in buys),
            len(sells),
        )

    def update_reward(self, strategy: str, reward: float) -> None:
        """Update reward for a strategy (compatibility hook for tests)."""
        logger.info("Strategy %s reward updated: %s", strategy, reward)


__all__ = ["StrategyAllocator"]

