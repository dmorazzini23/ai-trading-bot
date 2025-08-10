"""
Market regime detection and analysis system.

Identifies and classifies market regimes (bull, bear, sideways, crisis)
using multiple indicators and statistical models for adaptive trading strategies.
"""

# AI-AGENT-REF: use centralized import management
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from .imports import np, pd

# Use the centralized logger as per AGENTS.md
try:
    from ai_trading.logging import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    DISTRIBUTION = "distribution"
    ACCUMULATION = "accumulation"


class VolatilityRegime(Enum):
    """Volatility regime classifications."""

    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"


class TrendStrength(Enum):
    """Trend strength classifications."""

    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class RegimeDetector:
    """
    Market regime detection engine.

    Analyzes market conditions to identify current regime
    and predict regime transitions using multiple indicators.
    """

    def __init__(self, lookback_periods: int = 252):
        """Initialize regime detector."""
        # AI-AGENT-REF: Market regime detection system
        self.lookback_periods = lookback_periods  # 1 year of trading days

        # Regime thresholds
        self.volatility_thresholds = {
            "low": 0.10,  # 10% annualized volatility
            "normal": 0.20,  # 20% annualized volatility
            "high": 0.35,  # 35% annualized volatility
            "extreme": 0.50,  # 50% annualized volatility
        }

        self.trend_thresholds = {
            "bull_threshold": 0.15,  # 15% return over period
            "bear_threshold": -0.20,  # -20% return over period
            "sideways_threshold": 0.05,  # +/- 5% for sideways
        }

        # VIX-equivalent thresholds (if available)
        self.fear_thresholds = {
            "low_fear": 15,
            "normal_fear": 25,
            "high_fear": 35,
            "extreme_fear": 50,
        }

        # Regime history
        self.regime_history = []
        self.current_regime = None
        self.regime_confidence = 0.0

        logger.info(f"RegimeDetector initialized with {lookback_periods} day lookback")

    def detect_regime(
        self, market_data: pd.DataFrame, supplementary_data: dict = None
    ) -> dict[str, Any]:
        """
        Detect current market regime using comprehensive analysis.

        Args:
            market_data: OHLCV data with price and volume information
            supplementary_data: Additional data (VIX, economic indicators, etc.)

        Returns:
            Comprehensive regime analysis
        """
        try:
            if len(market_data) < 50:
                logger.warning(
                    f"Insufficient data for regime detection: {len(market_data)} bars"
                )
                return {"error": "Insufficient data"}

            analysis_start = datetime.now(UTC)

            # Calculate key metrics
            trend_analysis = self._analyze_trend(market_data)
            volatility_analysis = self._analyze_volatility(market_data)
            momentum_analysis = self._analyze_momentum(market_data)
            volume_analysis = self._analyze_volume(market_data)

            # Incorporate supplementary data if available
            sentiment_analysis = self._analyze_sentiment(supplementary_data)

            # Detect primary regime
            primary_regime = self._determine_primary_regime(
                trend_analysis, volatility_analysis, momentum_analysis
            )

            # Detect secondary characteristics
            secondary_characteristics = self._detect_secondary_characteristics(
                market_data, volume_analysis, sentiment_analysis
            )

            # Calculate regime confidence
            confidence_score = self._calculate_regime_confidence(
                trend_analysis, volatility_analysis, momentum_analysis
            )

            # Predict regime transition probability
            transition_analysis = self._analyze_regime_transitions(market_data)

            # Update regime history
            regime_result = {
                "timestamp": datetime.now(UTC),
                "primary_regime": primary_regime,
                "secondary_characteristics": secondary_characteristics,
                "confidence_score": confidence_score,
                "trend_analysis": trend_analysis,
                "volatility_analysis": volatility_analysis,
                "momentum_analysis": momentum_analysis,
                "volume_analysis": volume_analysis,
                "sentiment_analysis": sentiment_analysis,
                "transition_analysis": transition_analysis,
                "analysis_time_seconds": (
                    datetime.now(UTC) - analysis_start
                ).total_seconds(),
            }

            self._update_regime_history(regime_result)

            return regime_result

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {"error": str(e)}

    def _analyze_trend(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze trend characteristics."""
        try:
            # Calculate returns over different periods
            returns_1m = (
                (data["close"].iloc[-1] / data["close"].iloc[-21] - 1)
                if len(data) >= 21
                else 0
            )
            returns_3m = (
                (data["close"].iloc[-1] / data["close"].iloc[-63] - 1)
                if len(data) >= 63
                else 0
            )
            returns_6m = (
                (data["close"].iloc[-1] / data["close"].iloc[-126] - 1)
                if len(data) >= 126
                else 0
            )
            returns_12m = (
                (data["close"].iloc[-1] / data["close"].iloc[-252] - 1)
                if len(data) >= 252
                else 0
            )

            # Calculate moving averages
            data["SMA_50"] = data["close"].rolling(50).mean()
            data["SMA_200"] = data["close"].rolling(200).mean()

            current_price = data["close"].iloc[-1]
            sma_50 = (
                data["SMA_50"].iloc[-1]
                if not pd.isna(data["SMA_50"].iloc[-1])
                else current_price
            )
            sma_200 = (
                data["SMA_200"].iloc[-1]
                if not pd.isna(data["SMA_200"].iloc[-1])
                else current_price
            )

            # Trend direction
            price_vs_sma50 = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
            price_vs_sma200 = (current_price - sma_200) / sma_200 if sma_200 > 0 else 0
            sma_trend = (sma_50 - sma_200) / sma_200 if sma_200 > 0 else 0

            # Trend strength calculation
            trend_strength = self._calculate_trend_strength(
                returns_1m, returns_3m, returns_6m, price_vs_sma50, price_vs_sma200
            )

            # Determine trend direction
            if returns_6m > self.trend_thresholds["bull_threshold"]:
                trend_direction = "bullish"
            elif returns_6m < self.trend_thresholds["bear_threshold"]:
                trend_direction = "bearish"
            else:
                trend_direction = "sideways"

            return {
                "direction": trend_direction,
                "strength": trend_strength,
                "returns_1m": returns_1m,
                "returns_3m": returns_3m,
                "returns_6m": returns_6m,
                "returns_12m": returns_12m,
                "price_vs_sma50": price_vs_sma50,
                "price_vs_sma200": price_vs_sma200,
                "sma_trend": sma_trend,
                "above_sma200": current_price > sma_200,
            }

        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {"direction": "unknown", "strength": TrendStrength.WEAK}

    def _analyze_volatility(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze volatility characteristics."""
        try:
            # Calculate returns
            data["returns"] = data["close"].pct_change()

            # Different volatility measures
            vol_10d = (
                data["returns"].tail(10).std() * np.sqrt(252) if len(data) >= 10 else 0
            )
            vol_30d = (
                data["returns"].tail(30).std() * np.sqrt(252) if len(data) >= 30 else 0
            )
            vol_60d = (
                data["returns"].tail(60).std() * np.sqrt(252) if len(data) >= 60 else 0
            )
            vol_252d = (
                data["returns"].tail(252).std() * np.sqrt(252)
                if len(data) >= 252
                else 0
            )

            # Average True Range (volatility measure)
            data["high_low"] = data["high"] - data["low"]
            data["high_close"] = abs(data["high"] - data["close"].shift(1))
            data["low_close"] = abs(data["low"] - data["close"].shift(1))
            data["true_range"] = data[["high_low", "high_close", "low_close"]].max(
                axis=1
            )
            atr = (
                data["true_range"].rolling(14).mean().iloc[-1] if len(data) >= 14 else 0
            )

            # Volatility regime classification
            current_vol = vol_30d
            if current_vol < self.volatility_thresholds["low"]:
                vol_regime = VolatilityRegime.LOW_VOL
            elif current_vol < self.volatility_thresholds["normal"]:
                vol_regime = VolatilityRegime.NORMAL_VOL
            elif current_vol < self.volatility_thresholds["high"]:
                vol_regime = VolatilityRegime.HIGH_VOL
            else:
                vol_regime = VolatilityRegime.EXTREME_VOL

            # Volatility trend (increasing or decreasing)
            vol_trend = "stable"
            if vol_10d > vol_30d * 1.2:
                vol_trend = "increasing"
            elif vol_10d < vol_30d * 0.8:
                vol_trend = "decreasing"

            # GARCH-like volatility clustering
            returns_squared = data["returns"] ** 2
            vol_clustering = (
                returns_squared.tail(20).autocorr(lag=1) if len(data) >= 21 else 0
            )

            return {
                "regime": vol_regime,
                "trend": vol_trend,
                "current_vol": current_vol,
                "vol_10d": vol_10d,
                "vol_30d": vol_30d,
                "vol_60d": vol_60d,
                "vol_252d": vol_252d,
                "atr": atr,
                "volatility_clustering": vol_clustering,
                "vol_percentile": self._calculate_volatility_percentile(
                    current_vol, vol_252d
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {"regime": VolatilityRegime.NORMAL_VOL, "current_vol": 0.2}

    def _analyze_momentum(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze momentum characteristics."""
        try:
            # RSI calculation
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

            # MACD calculation
            ema_12 = data["close"].ewm(span=12).mean()
            ema_26 = data["close"].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal

            current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
            current_signal = (
                macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0
            )
            current_histogram = (
                macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0
            )

            # Momentum classification
            if current_rsi > 70 and current_macd > current_signal:
                momentum_state = "overbought_bullish"
            elif current_rsi < 30 and current_macd < current_signal:
                momentum_state = "oversold_bearish"
            elif current_rsi > 50 and current_macd > current_signal:
                momentum_state = "bullish"
            elif current_rsi < 50 and current_macd < current_signal:
                momentum_state = "bearish"
            else:
                momentum_state = "neutral"

            # Price momentum (rate of change)
            roc_10 = (
                (data["close"].iloc[-1] / data["close"].iloc[-11] - 1)
                if len(data) >= 11
                else 0
            )
            roc_20 = (
                (data["close"].iloc[-1] / data["close"].iloc[-21] - 1)
                if len(data) >= 21
                else 0
            )

            return {
                "state": momentum_state,
                "rsi": current_rsi,
                "macd": current_macd,
                "macd_signal": current_signal,
                "macd_histogram": current_histogram,
                "roc_10d": roc_10,
                "roc_20d": roc_20,
                "momentum_strength": self._calculate_momentum_strength(
                    current_rsi, current_macd, roc_20
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {"state": "neutral", "rsi": 50, "momentum_strength": "weak"}

    def _analyze_volume(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze volume characteristics."""
        try:
            if "volume" not in data.columns:
                return {"trend": "unknown", "strength": "weak"}

            # Volume moving averages
            volume_sma_20 = data["volume"].rolling(20).mean()
            volume_sma_50 = data["volume"].rolling(50).mean()

            current_volume = data["volume"].iloc[-1]
            avg_volume_20 = (
                volume_sma_20.iloc[-1]
                if not pd.isna(volume_sma_20.iloc[-1])
                else current_volume
            )
            avg_volume_50 = (
                volume_sma_50.iloc[-1]
                if not pd.isna(volume_sma_50.iloc[-1])
                else current_volume
            )

            # Volume ratios
            volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            volume_ratio_50 = current_volume / avg_volume_50 if avg_volume_50 > 0 else 1

            # Volume trend
            if avg_volume_20 > avg_volume_50 * 1.1:
                volume_trend = "increasing"
            elif avg_volume_20 < avg_volume_50 * 0.9:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"

            # On-Balance Volume (OBV)
            data["obv"] = 0
            for i in range(1, len(data)):
                if data["close"].iloc[i] > data["close"].iloc[i - 1]:
                    data["obv"].iloc[i] = (
                        data["obv"].iloc[i - 1] + data["volume"].iloc[i]
                    )
                elif data["close"].iloc[i] < data["close"].iloc[i - 1]:
                    data["obv"].iloc[i] = (
                        data["obv"].iloc[i - 1] - data["volume"].iloc[i]
                    )
                else:
                    data["obv"].iloc[i] = data["obv"].iloc[i - 1]

            obv_trend = "neutral"
            if len(data) >= 20:
                obv_20_ago = data["obv"].iloc[-21]
                obv_current = data["obv"].iloc[-1]
                if obv_current > obv_20_ago * 1.05:
                    obv_trend = "bullish"
                elif obv_current < obv_20_ago * 0.95:
                    obv_trend = "bearish"

            return {
                "trend": volume_trend,
                "current_volume": current_volume,
                "volume_ratio_20": volume_ratio_20,
                "volume_ratio_50": volume_ratio_50,
                "obv_trend": obv_trend,
                "volume_strength": (
                    "high"
                    if volume_ratio_20 > 1.5
                    else "normal" if volume_ratio_20 > 0.8 else "low"
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {"trend": "unknown", "strength": "weak"}

    def _analyze_sentiment(self, supplementary_data: dict = None) -> dict[str, Any]:
        """Analyze market sentiment indicators."""
        try:
            sentiment_analysis = {
                "fear_greed_index": 50,  # Neutral default
                "put_call_ratio": 1.0,  # Neutral default
                "sentiment_score": "neutral",
            }

            if not supplementary_data:
                return sentiment_analysis

            # VIX or volatility index analysis
            vix_level = supplementary_data.get("vix", 20)
            if vix_level < self.fear_thresholds["low_fear"]:
                fear_level = "low_fear"
            elif vix_level < self.fear_thresholds["normal_fear"]:
                fear_level = "normal"
            elif vix_level < self.fear_thresholds["high_fear"]:
                fear_level = "high_fear"
            else:
                fear_level = "extreme_fear"

            sentiment_analysis["vix_level"] = vix_level
            sentiment_analysis["fear_level"] = fear_level

            # Put/Call ratio if available
            if "put_call_ratio" in supplementary_data:
                pcr = supplementary_data["put_call_ratio"]
                sentiment_analysis["put_call_ratio"] = pcr

                if pcr > 1.2:
                    sentiment_analysis["sentiment_score"] = "bearish"
                elif pcr < 0.8:
                    sentiment_analysis["sentiment_score"] = "bullish"
                else:
                    sentiment_analysis["sentiment_score"] = "neutral"

            # Economic indicators
            if "economic_indicators" in supplementary_data:
                econ_data = supplementary_data["economic_indicators"]
                sentiment_analysis["economic_indicators"] = econ_data

            return sentiment_analysis

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"sentiment_score": "neutral"}

    def _determine_primary_regime(
        self, trend_analysis: dict, volatility_analysis: dict, momentum_analysis: dict
    ) -> MarketRegime:
        """Determine the primary market regime."""
        try:
            trend_direction = trend_analysis.get("direction", "sideways")
            vol_regime = volatility_analysis.get("regime", VolatilityRegime.NORMAL_VOL)
            momentum_state = momentum_analysis.get("state", "neutral")
            returns_6m = trend_analysis.get("returns_6m", 0)
            volatility_analysis.get("current_vol", 0.2)

            # Crisis detection (extreme volatility + significant losses)
            if vol_regime == VolatilityRegime.EXTREME_VOL and returns_6m < -0.15:
                return MarketRegime.CRISIS

            # High volatility regime
            if vol_regime in [VolatilityRegime.HIGH_VOL, VolatilityRegime.EXTREME_VOL]:
                return MarketRegime.HIGH_VOLATILITY

            # Bull market detection
            if (
                trend_direction == "bullish"
                and returns_6m > self.trend_thresholds["bull_threshold"]
                and momentum_state in ["bullish", "overbought_bullish"]
            ):
                return MarketRegime.BULL_MARKET

            # Bear market detection
            if (
                trend_direction == "bearish"
                and returns_6m < self.trend_thresholds["bear_threshold"]
                and momentum_state in ["bearish", "oversold_bearish"]
            ):
                return MarketRegime.BEAR_MARKET

            # Recovery detection (coming out of bear market)
            if (
                trend_direction == "bullish"
                and momentum_state == "bullish"
                and returns_6m > -0.05
                and returns_6m < 0.10
            ):
                return MarketRegime.RECOVERY

            # Sideways/range-bound market
            if (
                trend_direction == "sideways"
                and abs(returns_6m) < self.trend_thresholds["sideways_threshold"]
            ):
                return MarketRegime.SIDEWAYS

            # Default to sideways if unclear
            return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.error(f"Error determining primary regime: {e}")
            return MarketRegime.SIDEWAYS

    def _detect_secondary_characteristics(
        self, data: pd.DataFrame, volume_analysis: dict, sentiment_analysis: dict
    ) -> list[str]:
        """Detect secondary market characteristics."""
        try:
            characteristics = []

            # Volume characteristics
            volume_strength = volume_analysis.get("volume_strength", "normal")
            if volume_strength == "high":
                characteristics.append("high_volume")
            elif volume_strength == "low":
                characteristics.append("low_volume")

            # Sentiment characteristics
            fear_level = sentiment_analysis.get("fear_level", "normal")
            if fear_level == "extreme_fear":
                characteristics.append("extreme_fear")
            elif fear_level == "low_fear":
                characteristics.append("complacency")

            # Distribution/Accumulation patterns
            obv_trend = volume_analysis.get("obv_trend", "neutral")
            price_trend = (
                "up" if data["close"].iloc[-1] > data["close"].iloc[-21] else "down"
            )

            if obv_trend == "bullish" and price_trend == "down":
                characteristics.append("accumulation")
            elif obv_trend == "bearish" and price_trend == "up":
                characteristics.append("distribution")

            # Seasonal patterns (simplified)
            current_month = datetime.now(UTC).month
            if current_month in [11, 12, 1]:  # Winter rally season
                characteristics.append("seasonal_strength")
            elif current_month in [5]:  # Sell in May
                characteristics.append("seasonal_weakness")

            return characteristics

        except Exception as e:
            logger.error(f"Error detecting secondary characteristics: {e}")
            return []

    def _calculate_regime_confidence(
        self, trend_analysis: dict, volatility_analysis: dict, momentum_analysis: dict
    ) -> float:
        """Calculate confidence in regime identification."""
        try:
            confidence_factors = []

            # Trend confidence
            trend_strength = trend_analysis.get("strength", TrendStrength.WEAK)
            if isinstance(trend_strength, TrendStrength):
                trend_conf = trend_strength.value / 5.0
            else:
                trend_conf = 0.4
            confidence_factors.append(trend_conf)

            # Volatility confidence (lower volatility = higher confidence for trends)
            vol_regime = volatility_analysis.get("regime", VolatilityRegime.NORMAL_VOL)
            if vol_regime == VolatilityRegime.LOW_VOL:
                vol_conf = 0.9
            elif vol_regime == VolatilityRegime.NORMAL_VOL:
                vol_conf = 0.7
            elif vol_regime == VolatilityRegime.HIGH_VOL:
                vol_conf = 0.5
            else:  # EXTREME_VOL
                vol_conf = 0.3
            confidence_factors.append(vol_conf)

            # Momentum confirmation
            momentum_state = momentum_analysis.get("state", "neutral")
            if momentum_state in ["bullish", "bearish"]:
                momentum_conf = 0.8
            elif momentum_state in ["overbought_bullish", "oversold_bearish"]:
                momentum_conf = 0.9
            else:
                momentum_conf = 0.4
            confidence_factors.append(momentum_conf)

            # Calculate weighted average
            overall_confidence = sum(confidence_factors) / len(confidence_factors)

            return min(0.95, max(0.05, overall_confidence))

        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5

    def _analyze_regime_transitions(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze probability of regime transitions."""
        try:
            if len(self.regime_history) < 10:
                return {"transition_probability": 0.1, "next_likely_regime": "unknown"}

            # Simple transition analysis based on recent regime stability
            recent_regimes = [
                entry["primary_regime"] for entry in self.regime_history[-10:]
            ]
            unique_regimes = set(recent_regimes)

            # If many regime changes recently, higher transition probability
            regime_changes = len(unique_regimes)
            if regime_changes > 3:
                transition_prob = 0.4  # High probability of change
            elif regime_changes > 2:
                transition_prob = 0.2  # Medium probability
            else:
                transition_prob = 0.1  # Low probability

            # Predict next likely regime based on common transitions
            current_regime = self.current_regime
            next_likely_regime = current_regime  # Default to same

            # Simple transition logic
            if current_regime == MarketRegime.BULL_MARKET:
                next_likely_regime = MarketRegime.DISTRIBUTION
            elif current_regime == MarketRegime.BEAR_MARKET:
                next_likely_regime = MarketRegime.ACCUMULATION
            elif current_regime == MarketRegime.CRISIS:
                next_likely_regime = MarketRegime.RECOVERY
            elif current_regime == MarketRegime.SIDEWAYS:
                # Could go either way - check momentum
                if len(data) >= 20:
                    recent_returns = (
                        data["close"].iloc[-1] / data["close"].iloc[-21] - 1
                    )
                    if recent_returns > 0.05:
                        next_likely_regime = MarketRegime.BULL_MARKET
                    elif recent_returns < -0.05:
                        next_likely_regime = MarketRegime.BEAR_MARKET

            return {
                "transition_probability": transition_prob,
                "next_likely_regime": next_likely_regime,
                "regime_stability": 1.0 - transition_prob,
                "recent_regime_changes": regime_changes,
            }

        except Exception as e:
            logger.error(f"Error analyzing regime transitions: {e}")
            return {"transition_probability": 0.1}

    def _calculate_trend_strength(
        self,
        returns_1m: float,
        returns_3m: float,
        returns_6m: float,
        price_vs_sma50: float,
        price_vs_sma200: float,
    ) -> TrendStrength:
        """Calculate trend strength based on multiple factors."""
        try:
            # Score based on return consistency
            return_scores = []

            # 1-month return score
            if abs(returns_1m) > 0.05:
                return_scores.append(2)
            elif abs(returns_1m) > 0.02:
                return_scores.append(1)
            else:
                return_scores.append(0)

            # 3-month return score
            if abs(returns_3m) > 0.10:
                return_scores.append(2)
            elif abs(returns_3m) > 0.05:
                return_scores.append(1)
            else:
                return_scores.append(0)

            # 6-month return score
            if abs(returns_6m) > 0.15:
                return_scores.append(2)
            elif abs(returns_6m) > 0.08:
                return_scores.append(1)
            else:
                return_scores.append(0)

            # Moving average position scores
            if abs(price_vs_sma50) > 0.05:
                return_scores.append(1)
            if abs(price_vs_sma200) > 0.10:
                return_scores.append(1)

            # Calculate total score
            total_score = sum(return_scores)

            if total_score >= 6:
                return TrendStrength.VERY_STRONG
            elif total_score >= 4:
                return TrendStrength.STRONG
            elif total_score >= 2:
                return TrendStrength.MODERATE
            elif total_score >= 1:
                return TrendStrength.WEAK
            else:
                return TrendStrength.VERY_WEAK

        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return TrendStrength.WEAK

    def _calculate_volatility_percentile(
        self, current_vol: float, historical_vol: float
    ) -> float:
        """Calculate volatility percentile rank."""
        try:
            if historical_vol == 0:
                return 0.5

            # Simple percentile estimation
            vol_ratio = current_vol / historical_vol

            if vol_ratio > 2.0:
                return 0.95
            elif vol_ratio > 1.5:
                return 0.85
            elif vol_ratio > 1.2:
                return 0.70
            elif vol_ratio > 0.8:
                return 0.50
            elif vol_ratio > 0.6:
                return 0.30
            else:
                return 0.15

        except Exception as e:
            logger.error(f"Error calculating volatility percentile: {e}")
            return 0.5

    def _calculate_momentum_strength(self, rsi: float, macd: float, roc: float) -> str:
        """Calculate momentum strength classification."""
        try:
            strength_score = 0

            # RSI contribution
            if rsi > 70 or rsi < 30:
                strength_score += 2
            elif rsi > 60 or rsi < 40:
                strength_score += 1

            # MACD contribution
            if abs(macd) > 0.05:  # Arbitrary threshold
                strength_score += 2
            elif abs(macd) > 0.02:
                strength_score += 1

            # Rate of change contribution
            if abs(roc) > 0.05:
                strength_score += 2
            elif abs(roc) > 0.02:
                strength_score += 1

            if strength_score >= 5:
                return "very_strong"
            elif strength_score >= 3:
                return "strong"
            elif strength_score >= 2:
                return "moderate"
            else:
                return "weak"

        except Exception as e:
            logger.error(f"Error calculating momentum strength: {e}")
            return "weak"

    def _update_regime_history(self, regime_result: dict):
        """Update regime history for trend analysis."""
        try:
            self.regime_history.append(regime_result)

            # Keep only last 100 entries
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]

            # Update current regime
            self.current_regime = regime_result["primary_regime"]
            self.regime_confidence = regime_result["confidence_score"]

        except Exception as e:
            logger.error(f"Error updating regime history: {e}")

    def get_regime_summary(self) -> dict[str, Any]:
        """Get current regime summary."""
        try:
            if not self.regime_history:
                return {"error": "No regime data available"}

            latest = self.regime_history[-1]

            # Calculate regime stability
            recent_regimes = [
                entry["primary_regime"] for entry in self.regime_history[-10:]
            ]
            regime_changes = len(set(recent_regimes))
            stability = 1.0 - (regime_changes / 10.0)

            return {
                "current_regime": (
                    latest["primary_regime"].value
                    if hasattr(latest["primary_regime"], "value")
                    else str(latest["primary_regime"])
                ),
                "confidence": latest["confidence_score"],
                "stability": stability,
                "secondary_characteristics": latest.get(
                    "secondary_characteristics", []
                ),
                "volatility_regime": (
                    latest["volatility_analysis"]["regime"].value
                    if hasattr(latest["volatility_analysis"]["regime"], "value")
                    else str(latest["volatility_analysis"]["regime"])
                ),
                "trend_direction": latest["trend_analysis"]["direction"],
                "momentum_state": latest["momentum_analysis"]["state"],
                "last_updated": latest["timestamp"],
                "regime_history_length": len(self.regime_history),
            }

        except Exception as e:
            logger.error(f"Error getting regime summary: {e}")
            return {"error": str(e)}

    def get_regime_recommendations(self) -> dict[str, Any]:
        """Get trading recommendations based on current regime."""
        try:
            if not self.current_regime:
                return {"error": "No current regime identified"}

            recommendations = {
                "strategy_type": "conservative",
                "position_size_multiplier": 1.0,
                "risk_level": "medium",
                "recommended_timeframes": ["1h", "4h", "1d"],
                "avoid_strategies": [],
                "preferred_strategies": [],
            }

            # Regime-specific recommendations
            if self.current_regime == MarketRegime.BULL_MARKET:
                recommendations.update(
                    {
                        "strategy_type": "aggressive_long",
                        "position_size_multiplier": 1.2,
                        "risk_level": "medium_low",
                        "preferred_strategies": [
                            "momentum",
                            "breakout",
                            "trend_following",
                        ],
                        "avoid_strategies": ["short_selling", "contrarian"],
                    }
                )

            elif self.current_regime == MarketRegime.BEAR_MARKET:
                recommendations.update(
                    {
                        "strategy_type": "defensive",
                        "position_size_multiplier": 0.7,
                        "risk_level": "high",
                        "preferred_strategies": [
                            "short_selling",
                            "mean_reversion",
                            "defensive",
                        ],
                        "avoid_strategies": ["momentum_long", "growth"],
                    }
                )

            elif self.current_regime == MarketRegime.CRISIS:
                recommendations.update(
                    {
                        "strategy_type": "crisis_mode",
                        "position_size_multiplier": 0.3,
                        "risk_level": "very_high",
                        "preferred_strategies": ["cash", "defensive", "volatility"],
                        "avoid_strategies": ["momentum", "leverage", "growth"],
                    }
                )

            elif self.current_regime == MarketRegime.HIGH_VOLATILITY:
                recommendations.update(
                    {
                        "strategy_type": "volatility_adjusted",
                        "position_size_multiplier": 0.6,
                        "risk_level": "high",
                        "preferred_strategies": [
                            "mean_reversion",
                            "volatility_trading",
                        ],
                        "avoid_strategies": ["trend_following", "momentum"],
                    }
                )

            elif self.current_regime == MarketRegime.SIDEWAYS:
                recommendations.update(
                    {
                        "strategy_type": "range_bound",
                        "position_size_multiplier": 0.9,
                        "risk_level": "medium",
                        "preferred_strategies": [
                            "mean_reversion",
                            "range_trading",
                            "pairs",
                        ],
                        "avoid_strategies": ["trend_following", "breakout"],
                    }
                )

            # Adjust based on confidence
            if self.regime_confidence < 0.6:
                recommendations["position_size_multiplier"] *= 0.8
                recommendations["risk_level"] = "high"

            return recommendations

        except Exception as e:
            logger.error(f"Error getting regime recommendations: {e}")
            return {"error": str(e)}
