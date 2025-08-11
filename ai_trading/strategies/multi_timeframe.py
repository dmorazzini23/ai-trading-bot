"""
Multi-timeframe analysis framework for comprehensive market analysis.

Provides hierarchical signal generation across multiple timeframes,
signal combination logic, and timeframe conflict resolution for
institutional-grade trading strategies.
"""

# AI-AGENT-REF: use standard imports for hard dependencies
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pandas as pd

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger

from ..core.enums import TimeFrame


class SignalStrength(Enum):
    """Signal strength enumeration."""

    VERY_WEAK = 1
    WEAK = 2
    NEUTRAL = 3
    STRONG = 4
    VERY_STRONG = 5


class SignalDirection(Enum):
    """Signal direction enumeration."""

    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


class MultiTimeframeSignal:
    """Container for multi-timeframe trading signals."""

    def __init__(
        self,
        timeframe: TimeFrame,
        direction: SignalDirection,
        strength: SignalStrength,
        confidence: float,
        indicator: str = "",
        metadata: dict = None,
    ):
        """Initialize multi-timeframe signal."""
        self.timeframe = timeframe
        self.direction = direction
        self.strength = strength
        self.confidence = confidence  # 0.0 to 1.0
        self.indicator = indicator
        self.metadata = metadata or {}
        self.timestamp = datetime.now(UTC)

        # Calculate composite score
        self.score = self._calculate_score()

    def _calculate_score(self) -> float:
        """Calculate composite signal score."""
        # Base score from direction and strength
        base_score = self.direction.value * self.strength.value

        # Apply confidence weighting
        weighted_score = base_score * self.confidence

        # Normalize to -5.0 to +5.0 range
        return weighted_score

    def __repr__(self) -> str:
        return (
            f"Signal({self.timeframe.value}, {self.direction.name}, "
            f"{self.strength.name}, conf={self.confidence:.2f})"
        )


class TimeframeHierarchy:
    """
    Manages timeframe hierarchy and relationships.

    Defines the relative importance and alignment requirements
    between different timeframes for signal generation.
    """

    def __init__(self):
        """Initialize timeframe hierarchy."""
        # AI-AGENT-REF: Multi-timeframe hierarchy management
        self.hierarchy = {
            TimeFrame.WEEK_1: {"weight": 1.0, "rank": 1},
            TimeFrame.DAY_1: {"weight": 0.8, "rank": 2},
            TimeFrame.HOUR_4: {"weight": 0.6, "rank": 3},
            TimeFrame.HOUR_1: {"weight": 0.4, "rank": 4},
            TimeFrame.MINUTE_30: {"weight": 0.3, "rank": 5},
            TimeFrame.MINUTE_15: {"weight": 0.2, "rank": 6},
            TimeFrame.MINUTE_5: {"weight": 0.1, "rank": 7},
            TimeFrame.MINUTE_1: {"weight": 0.05, "rank": 8},
        }

        # Timeframe relationships for alignment checking
        self.alignment_groups = [
            [TimeFrame.WEEK_1, TimeFrame.DAY_1],  # Long-term
            [TimeFrame.DAY_1, TimeFrame.HOUR_4, TimeFrame.HOUR_1],  # Medium-term
            [TimeFrame.HOUR_1, TimeFrame.MINUTE_30, TimeFrame.MINUTE_15],  # Short-term
            [TimeFrame.MINUTE_15, TimeFrame.MINUTE_5, TimeFrame.MINUTE_1],  # Intraday
        ]

        logger.info("TimeframeHierarchy initialized with 8 timeframes")

    def get_weight(self, timeframe: TimeFrame) -> float:
        """Get weight for a timeframe."""
        return self.hierarchy.get(timeframe, {}).get("weight", 0.0)

    def get_rank(self, timeframe: TimeFrame) -> int:
        """Get rank for a timeframe (1=highest, 8=lowest)."""
        return self.hierarchy.get(timeframe, {}).get("rank", 999)

    def is_higher_timeframe(self, tf1: TimeFrame, tf2: TimeFrame) -> bool:
        """Check if tf1 is a higher timeframe than tf2."""
        return self.get_rank(tf1) < self.get_rank(tf2)

    def get_aligned_timeframes(self, timeframe: TimeFrame) -> list[TimeFrame]:
        """Get timeframes that should align with the given timeframe."""
        for group in self.alignment_groups:
            if timeframe in group:
                return group
        return [timeframe]

    def calculate_timeframe_score(
        self, signals: dict[TimeFrame, MultiTimeframeSignal]
    ) -> float:
        """Calculate weighted score across all timeframes."""
        try:
            total_weighted_score = 0.0
            total_weight = 0.0

            for timeframe, signal in signals.items():
                weight = self.get_weight(timeframe)
                weighted_score = signal.score * weight

                total_weighted_score += weighted_score
                total_weight += weight

            # Normalize by total weight
            if total_weight > 0:
                return total_weighted_score / total_weight
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating timeframe score: {e}")
            return 0.0


class MultiTimeframeAnalyzer:
    """
    Comprehensive multi-timeframe market analysis engine.

    Analyzes market conditions across multiple timeframes,
    generates unified signals, and provides conflict resolution.
    """

    def __init__(self, primary_timeframes: list[TimeFrame] = None):
        """Initialize multi-timeframe analyzer."""
        # AI-AGENT-REF: Multi-timeframe analysis engine
        self.primary_timeframes = primary_timeframes or [
            TimeFrame.DAY_1,
            TimeFrame.HOUR_4,
            TimeFrame.HOUR_1,
            TimeFrame.MINUTE_15,
        ]

        self.hierarchy = TimeframeHierarchy()
        self.current_signals = {}
        self.signal_history = {}

        # Analysis parameters
        self.alignment_threshold = 0.7  # Require 70% alignment for strong signals
        self.confidence_threshold = 0.6  # Minimum confidence for signal consideration

        logger.info(
            f"MultiTimeframeAnalyzer initialized with timeframes: {[tf.value for tf in self.primary_timeframes]}"
        )

    def analyze_symbol(
        self, symbol: str, market_data: dict[TimeFrame, pd.DataFrame]
    ) -> dict[str, Any]:
        """
        Perform comprehensive multi-timeframe analysis for a symbol.

        Args:
            symbol: Trading symbol to analyze
            market_data: Dictionary mapping timeframes to OHLCV data

        Returns:
            Comprehensive analysis results
        """
        try:
            analysis_start = datetime.now(UTC)

            # Generate signals for each timeframe
            timeframe_signals = {}
            for timeframe in self.primary_timeframes:
                if timeframe in market_data:
                    signals = self._analyze_timeframe(
                        symbol, timeframe, market_data[timeframe]
                    )
                    timeframe_signals[timeframe] = signals

            # Combine signals across timeframes
            combined_analysis = self._combine_timeframe_signals(
                symbol, timeframe_signals
            )

            # Perform alignment analysis
            alignment_analysis = self._analyze_signal_alignment(timeframe_signals)

            # Generate trading recommendation
            recommendation = self._generate_trading_recommendation(
                combined_analysis, alignment_analysis
            )

            # Store signals for history
            self.current_signals[symbol] = timeframe_signals
            self._update_signal_history(symbol, timeframe_signals)

            analysis_time = (datetime.now(UTC) - analysis_start).total_seconds()

            return {
                "symbol": symbol,
                "timestamp": datetime.now(UTC),
                "analysis_time_seconds": analysis_time,
                "timeframe_signals": timeframe_signals,
                "combined_analysis": combined_analysis,
                "alignment_analysis": alignment_analysis,
                "recommendation": recommendation,
                "signal_count": len(timeframe_signals),
            }

        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    def _analyze_timeframe(
        self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame
    ) -> list[MultiTimeframeSignal]:
        """Analyze a single timeframe and generate signals."""
        try:
            signals = []

            if len(data) < 20:
                logger.warning(
                    f"Insufficient data for {symbol} {timeframe.value}: {len(data)} bars"
                )
                return signals

            # Moving Average signals
            ma_signals = self._generate_ma_signals(timeframe, data)
            signals.extend(ma_signals)

            # RSI signals
            rsi_signals = self._generate_rsi_signals(timeframe, data)
            signals.extend(rsi_signals)

            # MACD signals
            macd_signals = self._generate_macd_signals(timeframe, data)
            signals.extend(macd_signals)

            # Bollinger Band signals
            bb_signals = self._generate_bollinger_signals(timeframe, data)
            signals.extend(bb_signals)

            # Volume signals
            volume_signals = self._generate_volume_signals(timeframe, data)
            signals.extend(volume_signals)

            logger.debug(
                f"Generated {len(signals)} signals for {symbol} {timeframe.value}"
            )
            return signals

        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe.value}: {e}")
            return []

    def _generate_ma_signals(
        self, timeframe: TimeFrame, data: pd.DataFrame
    ) -> list[MultiTimeframeSignal]:
        """Generate moving average signals."""
        try:
            signals = []

            # Calculate moving averages
            data["SMA_20"] = data["close"].rolling(20).mean()
            data["SMA_50"] = data["close"].rolling(50).mean()
            data["EMA_12"] = data["close"].ewm(span=12).mean()
            data["EMA_26"] = data["close"].ewm(span=26).mean()

            latest_price = data["close"].iloc[-1]
            sma_20 = data["SMA_20"].iloc[-1]
            sma_50 = data["SMA_50"].iloc[-1]
            ema_12 = data["EMA_12"].iloc[-1]
            ema_26 = data["EMA_26"].iloc[-1]

            # Price vs SMA signals
            if latest_price > sma_20 > sma_50:
                direction = SignalDirection.BULLISH
                strength = (
                    SignalStrength.STRONG
                    if latest_price > sma_20 * 1.02
                    else SignalStrength.WEAK
                )
            elif latest_price < sma_20 < sma_50:
                direction = SignalDirection.BEARISH
                strength = (
                    SignalStrength.STRONG
                    if latest_price < sma_20 * 0.98
                    else SignalStrength.WEAK
                )
            else:
                direction = SignalDirection.NEUTRAL
                strength = SignalStrength.NEUTRAL

            signals.append(
                MultiTimeframeSignal(
                    timeframe,
                    direction,
                    strength,
                    0.7,
                    "MA_Trend",
                    {"sma_20": sma_20, "sma_50": sma_50},
                )
            )

            # EMA crossover signals
            if ema_12 > ema_26:
                ema_direction = SignalDirection.BULLISH
                ema_strength = (
                    SignalStrength.STRONG
                    if (ema_12 - ema_26) / ema_26 > 0.01
                    else SignalStrength.WEAK
                )
            elif ema_12 < ema_26:
                ema_direction = SignalDirection.BEARISH
                ema_strength = (
                    SignalStrength.STRONG
                    if (ema_26 - ema_12) / ema_26 > 0.01
                    else SignalStrength.WEAK
                )
            else:
                ema_direction = SignalDirection.NEUTRAL
                ema_strength = SignalStrength.NEUTRAL

            signals.append(
                MultiTimeframeSignal(
                    timeframe,
                    ema_direction,
                    ema_strength,
                    0.6,
                    "EMA_Crossover",
                    {"ema_12": ema_12, "ema_26": ema_26},
                )
            )

            return signals

        except Exception as e:
            logger.error(f"Error generating MA signals: {e}")
            return []

    def _generate_rsi_signals(
        self, timeframe: TimeFrame, data: pd.DataFrame
    ) -> list[MultiTimeframeSignal]:
        """Generate RSI-based signals."""
        try:
            signals = []

            # Calculate RSI
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))

            latest_rsi = data["RSI"].iloc[-1]

            if pd.isna(latest_rsi):
                return signals

            # RSI signals
            if latest_rsi > 70:
                direction = SignalDirection.BEARISH
                strength = (
                    SignalStrength.VERY_STRONG
                    if latest_rsi > 80
                    else SignalStrength.STRONG
                )
                confidence = min(0.9, (latest_rsi - 70) / 20 + 0.5)
            elif latest_rsi < 30:
                direction = SignalDirection.BULLISH
                strength = (
                    SignalStrength.VERY_STRONG
                    if latest_rsi < 20
                    else SignalStrength.STRONG
                )
                confidence = min(0.9, (30 - latest_rsi) / 20 + 0.5)
            else:
                direction = SignalDirection.NEUTRAL
                strength = SignalStrength.NEUTRAL
                confidence = 0.3

            signals.append(
                MultiTimeframeSignal(
                    timeframe,
                    direction,
                    strength,
                    confidence,
                    "RSI",
                    {"rsi": latest_rsi},
                )
            )

            return signals

        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
            return []

    def _generate_macd_signals(
        self, timeframe: TimeFrame, data: pd.DataFrame
    ) -> list[MultiTimeframeSignal]:
        """Generate MACD signals."""
        try:
            signals = []

            # Calculate MACD
            ema_12 = data["close"].ewm(span=12).mean()
            ema_26 = data["close"].ewm(span=26).mean()
            data["MACD"] = ema_12 - ema_26
            data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()
            data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]

            latest_macd = data["MACD"].iloc[-1]
            latest_signal = data["MACD_Signal"].iloc[-1]
            latest_histogram = data["MACD_Histogram"].iloc[-1]

            if pd.isna(latest_macd) or pd.isna(latest_signal):
                return signals

            # MACD signals
            if latest_macd > latest_signal and latest_histogram > 0:
                direction = SignalDirection.BULLISH
                strength = (
                    SignalStrength.STRONG
                    if abs(latest_histogram) > abs(latest_macd) * 0.1
                    else SignalStrength.WEAK
                )
            elif latest_macd < latest_signal and latest_histogram < 0:
                direction = SignalDirection.BEARISH
                strength = (
                    SignalStrength.STRONG
                    if abs(latest_histogram) > abs(latest_macd) * 0.1
                    else SignalStrength.WEAK
                )
            else:
                direction = SignalDirection.NEUTRAL
                strength = SignalStrength.NEUTRAL

            signals.append(
                MultiTimeframeSignal(
                    timeframe,
                    direction,
                    strength,
                    0.65,
                    "MACD",
                    {
                        "macd": latest_macd,
                        "signal": latest_signal,
                        "histogram": latest_histogram,
                    },
                )
            )

            return signals

        except Exception as e:
            logger.error(f"Error generating MACD signals: {e}")
            return []

    def _generate_bollinger_signals(
        self, timeframe: TimeFrame, data: pd.DataFrame
    ) -> list[MultiTimeframeSignal]:
        """Generate Bollinger Band signals."""
        try:
            signals = []

            # Calculate Bollinger Bands
            data["BB_Middle"] = data["close"].rolling(20).mean()
            data["BB_Std"] = data["close"].rolling(20).std()
            data["BB_Upper"] = data["BB_Middle"] + (data["BB_Std"] * 2)
            data["BB_Lower"] = data["BB_Middle"] - (data["BB_Std"] * 2)

            latest_price = data["close"].iloc[-1]
            bb_upper = data["BB_Upper"].iloc[-1]
            bb_lower = data["BB_Lower"].iloc[-1]
            data["BB_Middle"].iloc[-1]

            if pd.isna(bb_upper) or pd.isna(bb_lower):
                return signals

            # Bollinger Band signals
            bb_position = (latest_price - bb_lower) / (bb_upper - bb_lower)

            if bb_position > 0.8:
                direction = SignalDirection.BEARISH
                strength = (
                    SignalStrength.STRONG if bb_position > 0.95 else SignalStrength.WEAK
                )
                confidence = min(0.8, bb_position)
            elif bb_position < 0.2:
                direction = SignalDirection.BULLISH
                strength = (
                    SignalStrength.STRONG if bb_position < 0.05 else SignalStrength.WEAK
                )
                confidence = min(0.8, 1 - bb_position)
            else:
                direction = SignalDirection.NEUTRAL
                strength = SignalStrength.NEUTRAL
                confidence = 0.3

            signals.append(
                MultiTimeframeSignal(
                    timeframe,
                    direction,
                    strength,
                    confidence,
                    "Bollinger_Bands",
                    {
                        "bb_position": bb_position,
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower,
                    },
                )
            )

            return signals

        except Exception as e:
            logger.error(f"Error generating Bollinger signals: {e}")
            return []

    def _generate_volume_signals(
        self, timeframe: TimeFrame, data: pd.DataFrame
    ) -> list[MultiTimeframeSignal]:
        """Generate volume-based signals."""
        try:
            signals = []

            if "volume" not in data.columns:
                return signals

            # Calculate volume metrics
            data["Volume_SMA"] = data["volume"].rolling(20).mean()

            latest_volume = data["volume"].iloc[-1]
            avg_volume = data["Volume_SMA"].iloc[-1]

            if pd.isna(avg_volume) or avg_volume == 0:
                return signals

            volume_ratio = latest_volume / avg_volume

            # Volume confirmation signals
            if volume_ratio > 1.5:
                # High volume - confirms price movement
                price_change = (data["close"].iloc[-1] - data["close"].iloc[-2]) / data[
                    "close"
                ].iloc[-2]
                if price_change > 0.01:
                    direction = SignalDirection.BULLISH
                elif price_change < -0.01:
                    direction = SignalDirection.BEARISH
                else:
                    direction = SignalDirection.NEUTRAL

                strength = (
                    SignalStrength.STRONG if volume_ratio > 2.0 else SignalStrength.WEAK
                )
                confidence = min(0.7, volume_ratio / 3.0)

            else:
                direction = SignalDirection.NEUTRAL
                strength = SignalStrength.NEUTRAL
                confidence = 0.2

            signals.append(
                MultiTimeframeSignal(
                    timeframe,
                    direction,
                    strength,
                    confidence,
                    "Volume",
                    {"volume_ratio": volume_ratio},
                )
            )

            return signals

        except Exception as e:
            logger.error(f"Error generating volume signals: {e}")
            return []

    def _combine_timeframe_signals(
        self,
        symbol: str,
        timeframe_signals: dict[TimeFrame, list[MultiTimeframeSignal]],
    ) -> dict[str, Any]:
        """Combine signals across all timeframes."""
        try:
            # Aggregate signals by indicator type
            indicator_scores = {}
            all_signals = []

            for timeframe, signals in timeframe_signals.items():
                weight = self.hierarchy.get_weight(timeframe)

                for signal in signals:
                    all_signals.append(signal)

                    if signal.indicator not in indicator_scores:
                        indicator_scores[signal.indicator] = []

                    weighted_score = signal.score * weight
                    indicator_scores[signal.indicator].append(weighted_score)

            # Calculate combined scores
            combined_scores = {}
            for indicator, scores in indicator_scores.items():
                combined_scores[indicator] = (
                    sum(scores) / len(scores) if scores else 0.0
                )

            # Overall signal score
            overall_score = (
                sum(combined_scores.values()) / len(combined_scores)
                if combined_scores
                else 0.0
            )

            # Signal statistics
            bullish_signals = len(
                [s for s in all_signals if s.direction == SignalDirection.BULLISH]
            )
            bearish_signals = len(
                [s for s in all_signals if s.direction == SignalDirection.BEARISH]
            )
            neutral_signals = len(
                [s for s in all_signals if s.direction == SignalDirection.NEUTRAL]
            )

            # Average confidence
            avg_confidence = (
                sum(s.confidence for s in all_signals) / len(all_signals)
                if all_signals
                else 0.0
            )

            return {
                "overall_score": overall_score,
                "indicator_scores": combined_scores,
                "signal_counts": {
                    "bullish": bullish_signals,
                    "bearish": bearish_signals,
                    "neutral": neutral_signals,
                    "total": len(all_signals),
                },
                "average_confidence": avg_confidence,
                "timeframe_weights": {
                    tf.value: self.hierarchy.get_weight(tf) for tf in timeframe_signals
                },
            }

        except Exception as e:
            logger.error(f"Error combining timeframe signals: {e}")
            return {"error": str(e)}

    def _analyze_signal_alignment(
        self, timeframe_signals: dict[TimeFrame, list[MultiTimeframeSignal]]
    ) -> dict[str, Any]:
        """Analyze signal alignment across timeframes."""
        try:
            alignment_analysis = {
                "overall_alignment": 0.0,
                "alignment_by_indicator": {},
                "conflicting_timeframes": [],
                "consensus_direction": SignalDirection.NEUTRAL,
                "alignment_strength": "weak",
            }

            # Group signals by indicator
            indicator_groups = {}
            for timeframe, signals in timeframe_signals.items():
                for signal in signals:
                    if signal.indicator not in indicator_groups:
                        indicator_groups[signal.indicator] = {}
                    indicator_groups[signal.indicator][timeframe] = signal

            # Analyze alignment for each indicator
            indicator_alignments = {}
            for indicator, tf_signals in indicator_groups.items():
                alignment_score = self._calculate_indicator_alignment(tf_signals)
                indicator_alignments[indicator] = alignment_score

            # Overall alignment
            overall_alignment = (
                sum(indicator_alignments.values()) / len(indicator_alignments)
                if indicator_alignments
                else 0.0
            )

            # Determine consensus direction
            all_scores = []
            for signals in timeframe_signals.values():
                for signal in signals:
                    weight = self.hierarchy.get_weight(signal.timeframe)
                    all_scores.append(signal.score * weight)

            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                if avg_score > 1.0:
                    consensus_direction = SignalDirection.BULLISH
                elif avg_score < -1.0:
                    consensus_direction = SignalDirection.BEARISH
                else:
                    consensus_direction = SignalDirection.NEUTRAL

            # Alignment strength
            if overall_alignment > 0.8:
                alignment_strength = "very_strong"
            elif overall_alignment > 0.6:
                alignment_strength = "strong"
            elif overall_alignment > 0.4:
                alignment_strength = "moderate"
            else:
                alignment_strength = "weak"

            alignment_analysis.update(
                {
                    "overall_alignment": overall_alignment,
                    "alignment_by_indicator": indicator_alignments,
                    "consensus_direction": consensus_direction,
                    "alignment_strength": alignment_strength,
                }
            )

            return alignment_analysis

        except Exception as e:
            logger.error(f"Error analyzing signal alignment: {e}")
            return {"error": str(e)}

    def _calculate_indicator_alignment(
        self, tf_signals: dict[TimeFrame, MultiTimeframeSignal]
    ) -> float:
        """Calculate alignment score for a specific indicator across timeframes."""
        try:
            if len(tf_signals) < 2:
                return 1.0  # Perfect alignment if only one timeframe

            # Get all signal directions
            directions = [signal.direction.value for signal in tf_signals.values()]

            # Calculate alignment as consistency of directions
            if len(set(directions)) == 1:
                return 1.0  # Perfect alignment

            # Partial alignment based on dominant direction
            direction_counts = {}
            for direction in directions:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1

            max_count = max(direction_counts.values())
            alignment_ratio = max_count / len(directions)

            return alignment_ratio

        except Exception as e:
            logger.error(f"Error calculating indicator alignment: {e}")
            return 0.0

    def _generate_trading_recommendation(
        self, combined_analysis: dict, alignment_analysis: dict
    ) -> dict[str, Any]:
        """Generate final trading recommendation."""
        try:
            recommendation = {
                "action": "HOLD",
                "confidence": 0.0,
                "risk_level": "medium",
                "position_size_multiplier": 1.0,
                "reasoning": [],
                "warnings": [],
            }

            overall_score = combined_analysis.get("overall_score", 0.0)
            alignment = alignment_analysis.get("overall_alignment", 0.0)
            avg_confidence = combined_analysis.get("average_confidence", 0.0)

            # Action determination
            if overall_score > 2.0 and alignment > 0.6:
                recommendation["action"] = "BUY"
                recommendation["confidence"] = min(
                    0.95, (overall_score / 5.0 + alignment) / 2
                )
                recommendation["reasoning"].append(
                    "Strong bullish signals with good alignment"
                )

            elif overall_score < -2.0 and alignment > 0.6:
                recommendation["action"] = "SELL"
                recommendation["confidence"] = min(
                    0.95, (abs(overall_score) / 5.0 + alignment) / 2
                )
                recommendation["reasoning"].append(
                    "Strong bearish signals with good alignment"
                )

            elif abs(overall_score) > 1.0:
                if overall_score > 0:
                    recommendation["action"] = "WEAK_BUY"
                else:
                    recommendation["action"] = "WEAK_SELL"
                recommendation["confidence"] = min(0.7, avg_confidence)
                recommendation["reasoning"].append(
                    "Moderate signals but limited alignment"
                )

            else:
                recommendation["action"] = "HOLD"
                recommendation["confidence"] = 0.3
                recommendation["reasoning"].append("Mixed or weak signals")

            # Risk level assessment
            if alignment < 0.4:
                recommendation["risk_level"] = "high"
                recommendation["warnings"].append("Low signal alignment increases risk")
                recommendation["position_size_multiplier"] = 0.5

            elif avg_confidence < 0.5:
                recommendation["risk_level"] = "medium_high"
                recommendation["warnings"].append("Low average confidence")
                recommendation["position_size_multiplier"] = 0.75

            else:
                recommendation["risk_level"] = "medium"
                recommendation["position_size_multiplier"] = 1.0

            # Additional reasoning
            signal_counts = combined_analysis.get("signal_counts", {})
            if signal_counts.get("neutral", 0) > signal_counts.get(
                "bullish", 0
            ) + signal_counts.get("bearish", 0):
                recommendation["warnings"].append(
                    "Many neutral signals indicate uncertainty"
                )

            return recommendation

        except Exception as e:
            logger.error(f"Error generating trading recommendation: {e}")
            return {"action": "HOLD", "confidence": 0.0, "error": str(e)}

    def _update_signal_history(
        self, symbol: str, signals: dict[TimeFrame, list[MultiTimeframeSignal]]
    ):
        """Update signal history for trend analysis."""
        try:
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []

            # Store current signals with timestamp
            history_entry = {"timestamp": datetime.now(UTC), "signals": signals}

            self.signal_history[symbol].append(history_entry)

            # Keep only last 100 entries
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]

        except Exception as e:
            logger.error(f"Error updating signal history: {e}")

    def get_signal_trend(
        self, symbol: str, lookback_periods: int = 10
    ) -> dict[str, Any]:
        """Analyze signal trends over time."""
        try:
            if (
                symbol not in self.signal_history
                or len(self.signal_history[symbol]) < 2
            ):
                return {"error": "Insufficient history"}

            recent_history = self.signal_history[symbol][-lookback_periods:]

            # Track score trends
            scores = []
            for entry in recent_history:
                total_score = 0.0
                signal_count = 0

                for timeframe, signals in entry["signals"].items():
                    weight = self.hierarchy.get_weight(timeframe)
                    for signal in signals:
                        total_score += signal.score * weight
                        signal_count += 1

                if signal_count > 0:
                    avg_score = total_score / signal_count
                    scores.append(avg_score)

            if len(scores) < 2:
                return {"error": "Insufficient score data"}

            # Calculate trend
            trend_direction = "neutral"
            if scores[-1] > scores[0]:
                trend_direction = "improving" if scores[-1] > 0 else "recovering"
            elif scores[-1] < scores[0]:
                trend_direction = "declining" if scores[-1] < 0 else "weakening"

            # Calculate trend strength
            score_change = abs(scores[-1] - scores[0])
            trend_strength = min(1.0, score_change / 5.0)

            return {
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "current_score": scores[-1],
                "score_change": scores[-1] - scores[0],
                "periods_analyzed": len(scores),
            }

        except Exception as e:
            logger.error(f"Error analyzing signal trend: {e}")
            return {"error": str(e)}

    def get_current_signals(
        self, symbol: str
    ) -> dict[TimeFrame, list[MultiTimeframeSignal]]:
        """Get current signals for a symbol."""
        return self.current_signals.get(symbol, {})
