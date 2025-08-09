"""
Technical Signal Analyzer for advanced position management.

Analyzes technical indicators to inform position holding decisions:
- Momentum divergence detection (price vs momentum)
- Volume analysis for position strength validation
- Relative strength analysis vs sector/market
- Support/resistance level identification

AI-AGENT-REF: Technical signal analysis for intelligent position exits
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# AI-AGENT-REF: graceful imports with fallbacks
try:
    import numpy as np
    import pandas as pd
except ImportError:
    # Use fallback implementations
    class MockNumpy:
        nan = float('nan')
        def array(self, data): return list(data) if data else []
        def mean(self, arr): return sum(arr) / len(arr) if arr else 0
        def std(self, arr):
            if not arr: return 0
            mean_val = sum(arr) / len(arr)
            return (sum((x - mean_val)**2 for x in arr) / len(arr))**0.5
        def corrcoef(self, x, y): return [[1.0, 0.0], [0.0, 1.0]]  # Mock correlation
    np = MockNumpy()
    
    class MockPandas:
        def DataFrame(self, data=None): return data or {}
        def Series(self, data=None): return data or []
        def isna(self, val): return val is None or (isinstance(val, float) and val != val)
    pd = MockPandas()

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Technical signal strength levels."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    NEUTRAL = "neutral"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class DivergenceType(Enum):
    """Types of momentum divergence."""
    BULLISH = "bullish"      # Price down, momentum up
    BEARISH = "bearish"      # Price up, momentum down
    NONE = "none"


@dataclass
class TechnicalSignals:
    """Technical analysis signals for position management."""
    symbol: str
    timestamp: datetime
    
    # Momentum analysis
    momentum_score: float  # 0-1 momentum strength
    divergence_type: DivergenceType
    divergence_strength: float  # 0-1 strength of divergence
    
    # Volume analysis
    volume_strength: float  # 0-1 volume confirmation
    volume_trend: str  # "increasing", "decreasing", "neutral"
    
    # Relative strength
    relative_strength_score: float  # 0-1 vs market/sector
    outperformance_rank: float  # 0-1 percentile rank
    
    # Support/Resistance
    distance_to_support: float  # % distance to nearest support
    distance_to_resistance: float  # % distance to nearest resistance
    sr_confidence: float  # 0-1 confidence in S/R levels
    
    # Overall assessment
    hold_recommendation: SignalStrength
    exit_urgency: float  # 0-1 urgency to exit position


class TechnicalSignalAnalyzer:
    """
    Analyze technical indicators for position management decisions.
    
    Provides sophisticated technical analysis to inform:
    - Position holding vs exit decisions
    - Momentum divergence detection
    - Volume confirmation analysis
    - Relative strength assessment
    - Support/resistance proximity
    """
    
    def __init__(self, ctx=None):
        self.ctx = ctx
        self.logger = logging.getLogger(__name__ + ".TechnicalSignalAnalyzer")
        
        # Analysis parameters
        self.momentum_period = 14
        self.volume_period = 20
        self.divergence_lookback = 10
        self.sr_periods = [20, 50, 100]  # Multiple timeframes for S/R
        
        # Signal thresholds
        self.strong_momentum_threshold = 0.7
        self.divergence_threshold = 0.6
        self.volume_confirmation_threshold = 0.6
        
    def analyze_signals(self, symbol: str, position_data: Any = None) -> TechnicalSignals:
        """
        Perform comprehensive technical analysis for position management.
        
        Args:
            symbol: Symbol to analyze
            position_data: Current position information
            
        Returns:
            TechnicalSignals with comprehensive analysis
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if market_data is None or len(market_data) < 50:
                return self._get_default_signals(symbol)
            
            # Perform individual analyses
            momentum_analysis = self._analyze_momentum(market_data)
            divergence_analysis = self._analyze_divergence(market_data)
            volume_analysis = self._analyze_volume(market_data)
            relative_strength = self._analyze_relative_strength(symbol, market_data)
            support_resistance = self._analyze_support_resistance(market_data)
            
            # Generate overall assessment
            hold_recommendation = self._calculate_hold_recommendation(
                momentum_analysis, divergence_analysis, volume_analysis,
                relative_strength, support_resistance
            )
            
            exit_urgency = self._calculate_exit_urgency(
                momentum_analysis, divergence_analysis, volume_analysis
            )
            
            # Create comprehensive signal object
            signals = TechnicalSignals(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                momentum_score=momentum_analysis.get('score', 0.5),
                divergence_type=divergence_analysis.get('type', DivergenceType.NONE),
                divergence_strength=divergence_analysis.get('strength', 0.0),
                volume_strength=volume_analysis.get('strength', 0.5),
                volume_trend=volume_analysis.get('trend', 'neutral'),
                relative_strength_score=relative_strength.get('score', 0.5),
                outperformance_rank=relative_strength.get('rank', 0.5),
                distance_to_support=support_resistance.get('support_distance', 10.0),
                distance_to_resistance=support_resistance.get('resistance_distance', 10.0),
                sr_confidence=support_resistance.get('confidence', 0.5),
                hold_recommendation=hold_recommendation,
                exit_urgency=exit_urgency
            )
            
            self.logger.info(
                "TECHNICAL_SIGNALS | %s momentum=%.2f divergence=%s volume=%.2f hold=%s",
                symbol, momentum_analysis.get('score', 0.5),
                divergence_analysis.get('type', DivergenceType.NONE).value,
                volume_analysis.get('strength', 0.5), hold_recommendation.value
            )
            
            return signals
            
        except Exception as exc:
            self.logger.warning("analyze_signals failed for %s: %s", symbol, exc)
            return self._get_default_signals(symbol)
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum indicators (RSI, MACD, price momentum)."""
        try:
            if 'close' not in data.columns or len(data) < self.momentum_period + 10:
                return {'score': 0.5, 'rsi': 50.0, 'macd': 0.0}
            
            closes = data['close']
            
            # Calculate RSI
            rsi = self._calculate_rsi(closes, self.momentum_period)
            
            # Calculate MACD
            macd_line, macd_signal = self._calculate_macd(closes)
            macd_histogram = macd_line - macd_signal if macd_line and macd_signal else 0.0
            
            # Calculate price rate of change
            if len(closes) >= 10:
                roc = (closes.iloc[-1] / closes.iloc[-10] - 1.0) * 100
            else:
                roc = 0.0
            
            # Combine momentum indicators into single score
            # RSI contribution (0-1 scale)
            rsi_normalized = rsi / 100.0 if not pd.isna(rsi) else 0.5
            
            # MACD contribution (-1 to 1, normalized to 0-1)
            macd_normalized = 0.5
            if macd_histogram != 0:
                # Simple normalization - positive MACD = bullish momentum
                macd_normalized = 0.5 + (macd_histogram / abs(macd_histogram)) * 0.2
                macd_normalized = max(0.0, min(1.0, macd_normalized))
            
            # ROC contribution (-50% to +50% capped, normalized to 0-1)
            roc_capped = max(-50.0, min(50.0, roc))
            roc_normalized = (roc_capped + 50.0) / 100.0
            
            # Weighted combination
            momentum_score = (rsi_normalized * 0.5) + (macd_normalized * 0.3) + (roc_normalized * 0.2)
            
            return {
                'score': momentum_score,
                'rsi': rsi if not pd.isna(rsi) else 50.0,
                'macd': macd_histogram,
                'roc': roc
            }
            
        except Exception:
            return {'score': 0.5, 'rsi': 50.0, 'macd': 0.0}
    
    def _analyze_divergence(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect momentum divergences between price and indicators."""
        try:
            if len(data) < self.divergence_lookback + self.momentum_period:
                return {'type': DivergenceType.NONE, 'strength': 0.0}
            
            closes = data['close'].tail(self.divergence_lookback + 5)
            
            # Calculate RSI for divergence analysis
            rsi_values = []
            for i in range(len(closes) - self.momentum_period + 1):
                window = closes.iloc[i:i + self.momentum_period]
                rsi = self._calculate_rsi(window, self.momentum_period)
                rsi_values.append(rsi if not pd.isna(rsi) else 50.0)
            
            if len(rsi_values) < self.divergence_lookback:
                return {'type': DivergenceType.NONE, 'strength': 0.0}
            
            # Get recent price and RSI trends
            recent_prices = closes.tail(self.divergence_lookback).tolist()
            recent_rsi = rsi_values[-self.divergence_lookback:]
            
            # Calculate linear trends (simple slope)
            price_trend = self._calculate_trend(recent_prices)
            rsi_trend = self._calculate_trend(recent_rsi)
            
            # Detect divergence
            divergence_type = DivergenceType.NONE
            divergence_strength = 0.0
            
            # Bearish divergence: price up, momentum down
            if price_trend > 0.1 and rsi_trend < -0.1:
                divergence_type = DivergenceType.BEARISH
                divergence_strength = abs(price_trend) + abs(rsi_trend)
            
            # Bullish divergence: price down, momentum up  
            elif price_trend < -0.1 and rsi_trend > 0.1:
                divergence_type = DivergenceType.BULLISH
                divergence_strength = abs(price_trend) + abs(rsi_trend)
            
            # Normalize strength to 0-1
            divergence_strength = min(1.0, divergence_strength / 2.0)
            
            return {
                'type': divergence_type,
                'strength': divergence_strength,
                'price_trend': price_trend,
                'rsi_trend': rsi_trend
            }
            
        except Exception:
            return {'type': DivergenceType.NONE, 'strength': 0.0}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns for trend confirmation."""
        try:
            if 'volume' not in data.columns or len(data) < self.volume_period:
                return {'strength': 0.5, 'trend': 'neutral'}
            
            volumes = data['volume']
            closes = data['close'] if 'close' in data.columns else None
            
            # Calculate volume moving average
            vol_sma = volumes.rolling(window=self.volume_period).mean()
            current_vol = volumes.iloc[-1]
            avg_vol = vol_sma.iloc[-1] if not pd.isna(vol_sma.iloc[-1]) else current_vol
            
            # Volume strength relative to average
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                vol_strength = min(1.0, vol_ratio / 2.0)  # Normalize, cap at 2x average
            else:
                vol_strength = 0.5
            
            # Volume trend analysis
            recent_volumes = volumes.tail(5).tolist()
            vol_trend_slope = self._calculate_trend(recent_volumes)
            
            if vol_trend_slope > 0.1:
                vol_trend = "increasing"
            elif vol_trend_slope < -0.1:
                vol_trend = "decreasing"
            else:
                vol_trend = "neutral"
            
            # Price-volume relationship
            if closes is not None and len(closes) >= 5:
                recent_prices = closes.tail(5).tolist()
                price_trend = self._calculate_trend(recent_prices)
                
                # Volume confirmation: rising price with rising volume is bullish
                if price_trend > 0 and vol_trend == "increasing":
                    vol_strength = min(1.0, vol_strength * 1.2)
                elif price_trend < 0 and vol_trend == "increasing":
                    vol_strength = min(1.0, vol_strength * 1.1)  # Selling pressure
            
            return {
                'strength': vol_strength,
                'trend': vol_trend,
                'ratio': vol_ratio if 'vol_ratio' in locals() else 1.0
            }
            
        except Exception:
            return {'strength': 0.5, 'trend': 'neutral'}
    
    def _analyze_relative_strength(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relative strength vs market/sector."""
        try:
            if 'close' not in data.columns or len(data) < 20:
                return {'score': 0.5, 'rank': 0.5}
            
            # Get symbol performance
            symbol_prices = data['close']
            symbol_return = (symbol_prices.iloc[-1] / symbol_prices.iloc[-20] - 1.0) * 100
            
            # Try to get market benchmark (SPY) performance
            market_data = self._get_market_data('SPY')
            if market_data is not None and 'close' in market_data.columns and len(market_data) >= 20:
                market_prices = market_data['close']
                market_return = (market_prices.iloc[-1] / market_prices.iloc[-20] - 1.0) * 100
                
                # Calculate relative strength
                relative_strength = symbol_return - market_return
                
                # Normalize to 0-1 (assuming +/-20% relative performance range)
                rs_normalized = max(0.0, min(1.0, (relative_strength + 20.0) / 40.0))
            else:
                # Fallback: use absolute performance
                rs_normalized = max(0.0, min(1.0, (symbol_return + 10.0) / 20.0))
            
            # Calculate percentile rank (simplified - would need sector data for full implementation)
            # For now, use relative strength as proxy
            rank = rs_normalized
            
            return {
                'score': rs_normalized,
                'rank': rank,
                'symbol_return': symbol_return,
                'relative_strength': relative_strength if 'relative_strength' in locals() else 0.0
            }
            
        except Exception:
            return {'score': 0.5, 'rank': 0.5}
    
    def _analyze_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify support/resistance levels and current price proximity."""
        try:
            if 'high' not in data.columns or 'low' not in data.columns or len(data) < 50:
                return {
                    'support_distance': 10.0,
                    'resistance_distance': 10.0,
                    'confidence': 0.5
                }
            
            highs = data['high']
            lows = data['low']
            closes = data['close']
            current_price = closes.iloc[-1]
            
            # Find recent swing highs and lows
            support_levels = []
            resistance_levels = []
            
            # Simple pivot point detection
            for period in self.sr_periods:
                if len(data) >= period:
                    # Recent lows for support
                    recent_lows = lows.tail(period)
                    min_low = recent_lows.min()
                    support_levels.append(min_low)
                    
                    # Recent highs for resistance
                    recent_highs = highs.tail(period)
                    max_high = recent_highs.max()
                    resistance_levels.append(max_high)
            
            # Find nearest support and resistance
            valid_support = [s for s in support_levels if s < current_price]
            valid_resistance = [r for r in resistance_levels if r > current_price]
            
            if valid_support:
                nearest_support = max(valid_support)
                support_distance = ((current_price - nearest_support) / current_price) * 100
            else:
                support_distance = 10.0  # Default distance
            
            if valid_resistance:
                nearest_resistance = min(valid_resistance)
                resistance_distance = ((nearest_resistance - current_price) / current_price) * 100
            else:
                resistance_distance = 10.0  # Default distance
            
            # Calculate confidence based on level validation
            # More occurrences of similar levels = higher confidence
            confidence = min(1.0, len(support_levels) / 3.0)
            
            return {
                'support_distance': support_distance,
                'resistance_distance': resistance_distance,
                'confidence': confidence,
                'support_levels': valid_support[-3:] if valid_support else [],
                'resistance_levels': valid_resistance[:3] if valid_resistance else []
            }
            
        except Exception:
            return {
                'support_distance': 10.0,
                'resistance_distance': 10.0,
                'confidence': 0.5
            }
    
    def _calculate_hold_recommendation(self, momentum: Dict, divergence: Dict,
                                     volume: Dict, rel_strength: Dict,
                                     support_resistance: Dict) -> SignalStrength:
        """Calculate overall hold recommendation based on technical signals."""
        try:
            score = 0.0
            
            # Momentum component (30% weight)
            momentum_score = momentum.get('score', 0.5)
            if momentum_score > 0.7:
                score += 0.3
            elif momentum_score > 0.6:
                score += 0.2
            elif momentum_score < 0.3:
                score -= 0.2
            elif momentum_score < 0.4:
                score -= 0.1
            
            # Divergence component (25% weight)
            div_type = divergence.get('type', DivergenceType.NONE)
            div_strength = divergence.get('strength', 0.0)
            
            if div_type == DivergenceType.BEARISH and div_strength > self.divergence_threshold:
                score -= 0.25  # Strong sell signal
            elif div_type == DivergenceType.BULLISH and div_strength > self.divergence_threshold:
                score += 0.25  # Strong hold signal
            
            # Volume component (20% weight)
            vol_strength = volume.get('strength', 0.5)
            vol_trend = volume.get('trend', 'neutral')
            
            if vol_strength > 0.7 and vol_trend == "increasing":
                score += 0.2
            elif vol_strength < 0.3:
                score -= 0.1
            
            # Relative strength component (15% weight)
            rs_score = rel_strength.get('score', 0.5)
            if rs_score > 0.6:
                score += 0.15
            elif rs_score < 0.4:
                score -= 0.15
            
            # Support/Resistance component (10% weight)
            resistance_distance = support_resistance.get('resistance_distance', 10.0)
            support_distance = support_resistance.get('support_distance', 10.0)
            
            if resistance_distance < 2.0:  # Very close to resistance
                score -= 0.1
            elif support_distance < 2.0:  # Very close to support
                score -= 0.05
            
            # Convert score to recommendation
            if score >= 0.3:
                return SignalStrength.VERY_STRONG
            elif score >= 0.15:
                return SignalStrength.STRONG
            elif score >= -0.15:
                return SignalStrength.NEUTRAL
            elif score >= -0.3:
                return SignalStrength.WEAK
            else:
                return SignalStrength.VERY_WEAK
                
        except Exception:
            return SignalStrength.NEUTRAL
    
    def _calculate_exit_urgency(self, momentum: Dict, divergence: Dict, volume: Dict) -> float:
        """Calculate urgency to exit position (0-1 scale)."""
        try:
            urgency = 0.0
            
            # Strong bearish divergence = high urgency
            if (divergence.get('type') == DivergenceType.BEARISH and
                divergence.get('strength', 0) > self.divergence_threshold):
                urgency += 0.4
            
            # Weak momentum = moderate urgency
            momentum_score = momentum.get('score', 0.5)
            if momentum_score < 0.3:
                urgency += 0.3
            elif momentum_score < 0.4:
                urgency += 0.2
            
            # Volume confirmation of weakness
            if volume.get('trend') == "increasing" and momentum_score < 0.5:
                urgency += 0.2  # High volume selling
            
            # Cap at 1.0
            return min(1.0, urgency)
            
        except Exception:
            return 0.0
    
    def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for technical analysis."""
        try:
            if self.ctx and hasattr(self.ctx, 'data_fetcher'):
                # Try minute data first for more granular analysis
                df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
                if df is not None and not df.empty and len(df) >= 50:
                    return df
                
                # Fallback to daily data
                df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
                if df is not None and not df.empty:
                    return df
            
            return None
            
        except Exception:
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = prices.diff()
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD line and signal line."""
        try:
            if len(prices) < slow + signal:
                return 0.0, 0.0
            
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            macd_signal = macd_line.ewm(span=signal).mean()
            
            return macd_line.iloc[-1], macd_signal.iloc[-1]
            
        except Exception:
            return 0.0, 0.0
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of a series of values."""
        try:
            if len(values) < 3:
                return 0.0
            
            n = len(values)
            x = list(range(n))
            y = values
            
            # Simple linear regression slope
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i]**2 for i in range(n))
            
            denominator = n * sum_x2 - sum_x**2
            if denominator == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Normalize slope relative to value range
            value_range = max(values) - min(values)
            if value_range > 0:
                normalized_slope = slope / value_range * len(values)
            else:
                normalized_slope = 0.0
            
            return normalized_slope
            
        except Exception:
            return 0.0
    
    def _get_default_signals(self, symbol: str) -> TechnicalSignals:
        """Return default signals when analysis fails."""
        return TechnicalSignals(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            momentum_score=0.5,
            divergence_type=DivergenceType.NONE,
            divergence_strength=0.0,
            volume_strength=0.5,
            volume_trend='neutral',
            relative_strength_score=0.5,
            outperformance_rank=0.5,
            distance_to_support=10.0,
            distance_to_resistance=10.0,
            sr_confidence=0.5,
            hold_recommendation=SignalStrength.NEUTRAL,
            exit_urgency=0.0
        )