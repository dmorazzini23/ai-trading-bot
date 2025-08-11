"""
Strategy allocation with proper signal confirmation and hold protection.
"""
import logging
from typing import Any

from ai_trading.config.management import TradingConfig
CONFIG = TradingConfig()

import copy

logger = logging.getLogger(__name__)


class StrategyAllocator:
    def __init__(self, config=None):
        # AI-AGENT-REF: Create a copy of config to prevent shared state issues between instances
        self.config = copy.deepcopy(config or CONFIG)

        # AI-AGENT-REF: Defensive initialization to ensure required attributes exist
        self._ensure_config_attributes()

        self.signal_history: dict[str, list[float]] = {}
        self.last_direction: dict[str, str] = {}
        self.last_confidence: dict[str, float] = {}
        self.hold_protect: dict[str, int] = {}

    def _ensure_config_attributes(self):
        """Ensure all required config attributes exist with proper defaults."""
        required_attrs = {
            'signal_confirmation_bars': 2,
            'delta_threshold': 0.02,
            'min_confidence': 0.6
        }

        for attr, default_value in required_attrs.items():
            if not hasattr(self.config, attr):
                logger.warning(f"Config missing attribute {attr}, setting default: {default_value}")
                setattr(self.config, attr, default_value)
            elif getattr(self.config, attr, None) is None:
                logger.warning(f"Config attribute {attr} is None, setting default: {default_value}")
                setattr(self.config, attr, default_value)

    def allocate(self, signals_by_strategy: dict[str, list[Any]]) -> list[Any]:
        # Add debug logging and input validation
        if not signals_by_strategy:
            logger.debug("Allocate called with empty signals_by_strategy")
            return []

        if not isinstance(signals_by_strategy, dict):
            logger.warning("Allocate called with non-dict signals_by_strategy: %s", type(signals_by_strategy))
            return []

        # Count total signals for logging
        total_signals = sum(len(signals) for signals in signals_by_strategy.values())
        logger.debug(f"Allocate called with {len(signals_by_strategy)} strategies, {total_signals} total signals")

        # AI-AGENT-REF: Enhanced debugging for signal confirmation troubleshooting
        logger.debug(f"Current config: signal_confirmation_bars={self.config.signal_confirmation_bars}, "
                    f"min_confidence={self.config.min_confidence}, delta_threshold={self.config.delta_threshold}")
        logger.debug(f"Current signal_history state: {dict(self.signal_history)}")

        confirmed_signals = self._confirm_signals(signals_by_strategy)

        # AI-AGENT-REF: Log confirmed signals for debugging
        confirmed_count = sum(len(signals) for signals in confirmed_signals.values())
        logger.debug(f"Signal confirmation produced {confirmed_count} confirmed signals from {total_signals} input signals")

        result = self._allocate_confirmed(confirmed_signals)

        logger.debug(f"Final allocation returning {len(result)} signals")
        if result:
            symbols = [s.symbol for s in result]
            logger.debug(f"Returned signal symbols: {symbols}")

        return result

    def _confirm_signals(self, signals_by_strategy: dict[str, list[Any]]) -> dict[str, list[Any]]:
        confirmed: dict[str, list[Any]] = {}

        for strategy, signals in signals_by_strategy.items():
            if not isinstance(signals, list):
                logger.warning(f"Strategy {strategy} has non-list signals: {type(signals)}")
                confirmed[strategy] = []
                continue

            logger.debug(f"Processing strategy {strategy} with {len(signals)} signals")
            confirmed[strategy] = []

            for s in signals:
                # Validate signal object has required attributes
                if not hasattr(s, 'symbol') or not hasattr(s, 'side') or not hasattr(s, 'confidence'):
                    logger.warning(f"Invalid signal object missing required attributes: {s}")
                    continue

                # Validate signal values
                if not s.symbol or not isinstance(s.symbol, str):
                    logger.warning(f"Invalid signal symbol: {s.symbol}")
                    continue

                if s.side not in ['buy', 'sell']:
                    logger.warning(f"Invalid signal side: {s.side}")
                    continue

                try:
                    original_confidence = float(s.confidence)
                    confidence = original_confidence

                    # AI-AGENT-REF: Enhanced confidence normalization with better validation
                    if confidence < 0 or confidence > 1:
                        if confidence > 1:
                            # For values > 1, use improved tanh-based normalization
                            import math
                            normalized_confidence = (math.tanh(confidence - 1) + 1) / 2
                            # Ensure it stays in [0.5, 1] range for values > 1
                            normalized_confidence = 0.5 + (normalized_confidence * 0.5)
                            confidence = normalized_confidence
                        else:
                            # For negative values, clamp to small positive value
                            confidence = max(0.01, confidence)

                        # Final safety clamp to ensure [0,1] range
                        confidence = max(0.01, min(1.0, confidence))
                        s.confidence = confidence

                        # Log normalization with original value
                        logger.info(f"CONFIDENCE_NORMALIZED | symbol={s.symbol} original={original_confidence:.4f} normalized={confidence:.4f}")

                except (TypeError, ValueError):
                    logger.warning(f"Invalid signal confidence (not numeric): {s.confidence}")
                    continue

                logger.debug(f"Signal: {s.symbol}, confidence: {s.confidence}")
                key = f"{s.symbol}_{s.side}"
                if key not in self.signal_history:
                    self.signal_history[key] = []

                self.signal_history[key].append(s.confidence)
                # AI-AGENT-REF: Ensure signal_confirmation_bars is valid before using it
                confirmation_bars = getattr(self.config, 'signal_confirmation_bars', 2)
                if confirmation_bars is None or not isinstance(confirmation_bars, int) or confirmation_bars < 1:
                    logger.warning(f"Invalid signal_confirmation_bars: {confirmation_bars}, using default 2")
                    confirmation_bars = 2

                self.signal_history[key] = self.signal_history[key][-confirmation_bars:]

                if len(self.signal_history[key]) >= confirmation_bars:
                    # AI-AGENT-REF: Enhanced signal confirmation with additional validation
                    history_values = self.signal_history[key]
                    if not history_values:  # Defensive check for empty history
                        logger.warning(f"Empty signal history for {key}, skipping confirmation")
                        continue

                    # Calculate average confidence with additional validation
                    try:
                        avg_conf = sum(history_values) / len(history_values)
                        if not isinstance(avg_conf, (int, float)) or avg_conf < 0:
                            logger.warning(f"Invalid average confidence {avg_conf} for {s.symbol}, skipping")
                            continue
                    except (ZeroDivisionError, TypeError, ValueError) as e:
                        logger.warning(f"Error calculating average confidence for {s.symbol}: {e}")
                        continue

                    # AI-AGENT-REF: use configurable min_confidence with robust fallback
                    min_conf_threshold = getattr(self.config, 'min_confidence', 0.6)

                    # AI-AGENT-REF: Additional defensive check for None or invalid threshold
                    if min_conf_threshold is None or not isinstance(min_conf_threshold, (int, float)):
                        logger.warning(f"Invalid min_confidence threshold: {min_conf_threshold}, using default 0.6")
                        min_conf_threshold = 0.6

                    # AI-AGENT-REF: Enhanced debugging for confirmation decisions
                    logger.debug(f"Signal confirmation check: {s.symbol}, history={history_values}, "
                                f"avg_conf={avg_conf:.4f}, threshold={min_conf_threshold:.4f}, "
                                f"bars_required={confirmation_bars}, bars_available={len(history_values)}")

                    if avg_conf >= min_conf_threshold:
                        # AI-AGENT-REF: Create a copy of the signal to avoid mutation issues
                        import copy
                        confirmed_signal = copy.deepcopy(s)
                        confirmed_signal.confidence = avg_conf
                        confirmed[strategy].append(confirmed_signal)
                        logger.debug(f"Signal CONFIRMED: {s.symbol} with avg_conf: {avg_conf:.4f} >= threshold: {min_conf_threshold:.4f}")
                    else:
                        logger.debug(f"Signal REJECTED: {s.symbol}, avg_conf: {avg_conf:.4f} < threshold: {min_conf_threshold:.4f}")
                else:
                    logger.debug(f"Signal NOT READY: {s.symbol}, history length: {len(self.signal_history[key])}/{confirmation_bars} (need {confirmation_bars - len(self.signal_history[key])} more)")
                    # AI-AGENT-REF: Additional debug info about what's in history
                    logger.debug(f"  Current history for {key}: {self.signal_history[key]}")
        return confirmed

    def _allocate_confirmed(self, confirmed_signals: dict[str, list[Any]]) -> list[Any]:
        final_signals: list[Any] = []
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
                    extra={"symbol": s.symbol, "delta": delta, "threshold": self.config.delta_threshold},
                )
                continue

            self.last_direction[s.symbol] = s.side
            self.last_confidence[s.symbol] = s.confidence

            if s.side == "buy":
                self.hold_protect[s.symbol] = 4  # Changed from 3 to 4

            final_signals.append(s)

        # AI-AGENT-REF: Assign portfolio weights to final signals
        self._assign_portfolio_weights(final_signals)
        return final_signals

    def _assign_portfolio_weights(self, signals: list[Any]) -> None:
        """Assign portfolio weights to signals based on confidence and risk management."""
        if not signals:
            return

        # Get exposure cap from config (default 88%)
        max_total_exposure = getattr(self.config, 'exposure_cap_aggressive', 0.88)

        # Group signals by asset class for exposure management
        buy_signals = [s for s in signals if s.side == "buy"]
        sell_signals = [s for s in signals if s.side == "sell"]

        # Handle buy signals with portfolio allocation
        if buy_signals:
            # Calculate total confidence for normalization
            total_confidence = sum(s.confidence for s in buy_signals)

            if total_confidence > 0:
                # Allocate weights based on confidence, but cap total exposure
                available_exposure = max_total_exposure

                for signal in buy_signals:
                    # Base weight from confidence proportion
                    confidence_weight = (signal.confidence / total_confidence) * available_exposure

                    # Cap individual position at reasonable maximum (e.g., 25% for diversification)
                    max_individual_weight = min(0.25, available_exposure / len(buy_signals) * 1.5)
                    signal.weight = min(confidence_weight, max_individual_weight)

                    logger.debug(f"Assigned weight {signal.weight:.3f} to {signal.symbol} (confidence={signal.confidence:.3f})")
            else:
                # Fallback: equal weight distribution
                equal_weight = min(max_total_exposure / len(buy_signals), 0.15)  # Cap at 15% per position
                for signal in buy_signals:
                    signal.weight = equal_weight
                    logger.debug(f"Assigned equal weight {signal.weight:.3f} to {signal.symbol}")

        # Handle sell signals (exit positions, so weight represents position to close)
        for signal in sell_signals:
            # For sell signals, weight should represent the fraction of the position to close
            # AI-AGENT-REF: Fix sell signal weight assignment to respect exposure caps

            # For position exits, we need to size based on current position, not total capital
            # The weight should represent the portfolio allocation of the position being closed
            # rather than trying to allocate 100% of total capital

            # Use a reasonable default that respects exposure caps
            # This should be set based on actual position size, but for safety, cap at max exposure
            max_exit_weight = min(0.25, max_total_exposure / max(len(sell_signals), 1))  # Cap individual exits
            signal.weight = max_exit_weight

            logger.debug(f"Assigned exit weight {signal.weight:.3f} to {signal.symbol} (was defaulting to 1.0)")

        # Validate total exposure doesn't exceed cap
        total_buy_weight = sum(s.weight for s in buy_signals)
        if total_buy_weight > max_total_exposure:
            logger.warning(f"Total buy weight {total_buy_weight:.3f} exceeds cap {max_total_exposure:.3f}, scaling down")
            scale_factor = max_total_exposure / total_buy_weight
            for signal in buy_signals:
                signal.weight *= scale_factor
                logger.debug(f"Scaled weight to {signal.weight:.3f} for {signal.symbol}")

        logger.info(f"Portfolio allocation: {len(buy_signals)} buys (total weight: {sum(s.weight for s in buy_signals):.3f}), {len(sell_signals)} sells")

    def update_reward(self, strategy: str, reward: float) -> None:
        """Update reward for a strategy (placeholder for test compatibility)."""
        logger.info(f"Strategy {strategy} reward updated: {reward}")


__all__ = ["StrategyAllocator"]

