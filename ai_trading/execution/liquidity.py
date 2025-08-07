"""
Liquidity management system for production trading.

Provides comprehensive liquidity analysis, volume screening,
and execution optimization for institutional-grade trading operations.
"""

import time
import statistics
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

# Use the centralized logger as per AGENTS.md
try:
    from ai_trading.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.enums import TimeFrame, OrderType
from ..core.constants import RISK_PARAMETERS, EXECUTION_PARAMETERS


class LiquidityLevel(Enum):
    """Liquidity level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MarketHours(Enum):
    """Market hours classifications."""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    REGULAR_HOURS = "regular_hours"
    MARKET_CLOSE = "market_close"
    AFTER_HOURS = "after_hours"


class LiquidityAnalyzer:
    """
    Comprehensive liquidity analysis engine.
    
    Analyzes market liquidity conditions and provides
    recommendations for optimal order execution timing.
    """
    
    def __init__(self):
        """Initialize liquidity analyzer."""
        # AI-AGENT-REF: Comprehensive liquidity analysis system
        self.liquidity_thresholds = {
            "very_low": 100000,      # $100k daily volume
            "low": 500000,           # $500k daily volume  
            "normal": 2000000,       # $2M daily volume
            "high": 10000000,        # $10M daily volume
            "very_high": 50000000    # $50M daily volume
        }
        
        # Volume analysis parameters
        self.volume_lookback_days = 20
        self.participation_thresholds = {
            "conservative": 0.05,    # 5% of volume
            "moderate": 0.10,        # 10% of volume
            "aggressive": 0.20       # 20% of volume
        }
        
        # Liquidity history for trend analysis
        self.liquidity_history = {}
        
        logger.info("LiquidityAnalyzer initialized")
    
    def analyze_liquidity(self, symbol: str, market_data: Dict, 
                         current_price: float = None) -> Dict[str, Any]:
        """
        Analyze liquidity conditions for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data with volume
            current_price: Current market price
            
        Returns:
            Comprehensive liquidity analysis
        """
        try:
            analysis_start = datetime.now(timezone.utc)
            
            # Extract volume data
            volume_analysis = self._analyze_volume_patterns(market_data)
            
            # Analyze bid-ask spread if available
            spread_analysis = self._analyze_bid_ask_spread(market_data, current_price)
            
            # Calculate daily dollar volume
            dollar_volume_analysis = self._analyze_dollar_volume(market_data, current_price)
            
            # Analyze market hours impact
            market_hours_analysis = self._analyze_market_hours_liquidity()
            
            # Determine overall liquidity level
            liquidity_level = self._determine_liquidity_level(
                volume_analysis, dollar_volume_analysis, spread_analysis
            )
            
            # Calculate execution recommendations
            execution_recommendations = self._generate_execution_recommendations(
                liquidity_level, volume_analysis, market_hours_analysis
            )
            
            # Update liquidity history
            analysis_result = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc),
                "liquidity_level": liquidity_level,
                "volume_analysis": volume_analysis,
                "spread_analysis": spread_analysis,
                "dollar_volume_analysis": dollar_volume_analysis,
                "market_hours_analysis": market_hours_analysis,
                "execution_recommendations": execution_recommendations,
                "analysis_time_seconds": (datetime.now(timezone.utc) - analysis_start).total_seconds()
            }
            
            self._update_liquidity_history(symbol, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def _analyze_volume_patterns(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze volume patterns and trends."""
        try:
            volume_data = market_data.get("volume", [])
            if not volume_data or len(volume_data) < 10:
                return {"error": "Insufficient volume data"}
            
            # Convert to list if pandas Series
            if hasattr(volume_data, 'tolist'):
                volume_data = volume_data.tolist()
            
            # Recent volume statistics
            recent_volume = volume_data[-10:] if len(volume_data) >= 10 else volume_data
            avg_volume = statistics.mean(recent_volume)
            volume_std = statistics.stdev(recent_volume) if len(recent_volume) > 1 else 0
            
            # Volume trend
            if len(volume_data) >= 20:
                recent_avg = statistics.mean(volume_data[-10:])
                older_avg = statistics.mean(volume_data[-20:-10])
                volume_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            else:
                volume_trend = 0
            
            # Current volume relative to average
            current_volume = volume_data[-1] if volume_data else 0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Volume volatility (consistency measure)
            volume_cv = volume_std / avg_volume if avg_volume > 0 else 0
            
            # Intraday volume pattern analysis (simplified)
            volume_pattern = self._classify_volume_pattern(volume_ratio, volume_trend)
            
            return {
                "avg_volume": avg_volume,
                "current_volume": current_volume,
                "volume_ratio": volume_ratio,
                "volume_trend": volume_trend,
                "volume_volatility": volume_cv,
                "volume_pattern": volume_pattern,
                "data_points": len(volume_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {e}")
            return {"error": str(e)}
    
    def _analyze_bid_ask_spread(self, market_data: Dict, current_price: float = None) -> Dict[str, Any]:
        """Analyze bid-ask spread for liquidity assessment."""
        try:
            bid_data = market_data.get("bid", [])
            ask_data = market_data.get("ask", [])
            
            if not bid_data or not ask_data:
                # Estimate spread based on price volatility if actual spread not available
                price_data = market_data.get("close", [])
                if price_data and len(price_data) >= 10:
                    recent_prices = price_data[-10:] if hasattr(price_data, '__getitem__') else [price_data]
                    price_std = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
                    estimated_spread = price_std * 2  # Simple estimation
                    
                    return {
                        "estimated_spread": estimated_spread,
                        "spread_basis_points": (estimated_spread / current_price * 10000) if current_price else 0,
                        "spread_quality": "estimated"
                    }
                else:
                    return {"error": "Insufficient price data for spread estimation"}
            
            # Calculate actual spreads
            spreads = []
            for bid, ask in zip(bid_data, ask_data):
                if bid > 0 and ask > bid:
                    spreads.append(ask - bid)
            
            if not spreads:
                return {"error": "No valid bid-ask spreads found"}
            
            avg_spread = statistics.mean(spreads)
            spread_std = statistics.stdev(spreads) if len(spreads) > 1 else 0
            current_spread = spreads[-1] if spreads else 0
            
            # Convert to basis points
            if current_price and current_price > 0:
                spread_bps = (avg_spread / current_price) * 10000
                current_spread_bps = (current_spread / current_price) * 10000
            else:
                spread_bps = 0
                current_spread_bps = 0
            
            # Spread quality assessment
            if spread_bps < 5:
                spread_quality = "excellent"
            elif spread_bps < 15:
                spread_quality = "good"
            elif spread_bps < 30:
                spread_quality = "fair"
            else:
                spread_quality = "poor"
            
            return {
                "avg_spread": avg_spread,
                "current_spread": current_spread,
                "spread_std": spread_std,
                "spread_basis_points": spread_bps,
                "current_spread_bps": current_spread_bps,
                "spread_quality": spread_quality,
                "data_points": len(spreads)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bid-ask spread: {e}")
            return {"error": str(e)}
    
    def _analyze_dollar_volume(self, market_data: Dict, current_price: float = None) -> Dict[str, Any]:
        """Analyze dollar volume for liquidity assessment."""
        try:
            volume_data = market_data.get("volume", [])
            price_data = market_data.get("close", [])
            
            if not volume_data or not price_data:
                return {"error": "Insufficient volume or price data"}
            
            # Ensure same length
            min_length = min(len(volume_data), len(price_data))
            if min_length < 5:
                return {"error": "Insufficient data points"}
            
            volume_data = volume_data[-min_length:] if hasattr(volume_data, '__getitem__') else [volume_data]
            price_data = price_data[-min_length:] if hasattr(price_data, '__getitem__') else [price_data]
            
            # Calculate dollar volumes
            dollar_volumes = []
            for vol, price in zip(volume_data, price_data):
                if vol > 0 and price > 0:
                    dollar_volumes.append(vol * price)
            
            if not dollar_volumes:
                return {"error": "No valid dollar volume data"}
            
            # Statistics
            avg_dollar_volume = statistics.mean(dollar_volumes)
            current_dollar_volume = dollar_volumes[-1] if dollar_volumes else 0
            
            # 30-day average if enough data
            if len(dollar_volumes) >= 20:
                monthly_avg = statistics.mean(dollar_volumes[-20:])
            else:
                monthly_avg = avg_dollar_volume
            
            # Volume relative to 30-day average
            volume_relative_to_monthly = current_dollar_volume / monthly_avg if monthly_avg > 0 else 0
            
            return {
                "avg_dollar_volume": avg_dollar_volume,
                "current_dollar_volume": current_dollar_volume,
                "monthly_avg_dollar_volume": monthly_avg,
                "volume_relative_to_monthly": volume_relative_to_monthly,
                "data_points": len(dollar_volumes)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dollar volume: {e}")
            return {"error": str(e)}
    
    def _analyze_market_hours_liquidity(self) -> Dict[str, Any]:
        """Analyze current market hours impact on liquidity."""
        try:
            current_time = datetime.now(timezone.utc)
            hour = current_time.hour
            minute = current_time.minute
            weekday = current_time.weekday()
            
            # US market hours (assuming UTC timing needs adjustment)
            # This is a simplified version - production would use proper timezone handling
            if weekday >= 5:  # Weekend
                market_session = MarketHours.AFTER_HOURS
                liquidity_impact = "very_low"
            elif 9 <= hour < 16:  # Regular market hours (simplified)
                if hour == 9 and minute < 30:
                    market_session = MarketHours.MARKET_OPEN
                    liquidity_impact = "high"
                elif hour >= 15 and minute >= 30:
                    market_session = MarketHours.MARKET_CLOSE
                    liquidity_impact = "high"
                else:
                    market_session = MarketHours.REGULAR_HOURS
                    liquidity_impact = "normal"
            elif 4 <= hour < 9:
                market_session = MarketHours.PRE_MARKET
                liquidity_impact = "low"
            else:
                market_session = MarketHours.AFTER_HOURS
                liquidity_impact = "very_low"
            
            # Additional factors
            is_friday = weekday == 4
            is_monday = weekday == 0
            is_option_expiry = self._is_option_expiry_week(current_time)
            
            # Adjust liquidity impact based on special conditions
            if is_friday and hour >= 15:
                liquidity_impact = "low"  # Friday afternoon
            elif is_monday and hour <= 10:
                liquidity_impact = "reduced"  # Monday morning
            elif is_option_expiry:
                liquidity_impact = "elevated"  # Options expiry week
            
            return {
                "market_session": market_session,
                "liquidity_impact": liquidity_impact,
                "is_weekend": weekday >= 5,
                "is_friday": is_friday,
                "is_monday": is_monday,
                "is_option_expiry_week": is_option_expiry,
                "current_hour": hour
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market hours liquidity: {e}")
            return {"market_session": MarketHours.REGULAR_HOURS, "liquidity_impact": "normal"}
    
    def _determine_liquidity_level(self, volume_analysis: Dict, dollar_volume_analysis: Dict, 
                                 spread_analysis: Dict) -> LiquidityLevel:
        """Determine overall liquidity level."""
        try:
            # Score components
            volume_score = 0
            dollar_volume_score = 0
            spread_score = 0
            
            # Volume score
            volume_ratio = volume_analysis.get("volume_ratio", 0)
            if volume_ratio > 2.0:
                volume_score = 4
            elif volume_ratio > 1.5:
                volume_score = 3
            elif volume_ratio > 0.8:
                volume_score = 2
            elif volume_ratio > 0.5:
                volume_score = 1
            else:
                volume_score = 0
            
            # Dollar volume score
            avg_dollar_volume = dollar_volume_analysis.get("avg_dollar_volume", 0)
            if avg_dollar_volume >= self.liquidity_thresholds["very_high"]:
                dollar_volume_score = 4
            elif avg_dollar_volume >= self.liquidity_thresholds["high"]:
                dollar_volume_score = 3
            elif avg_dollar_volume >= self.liquidity_thresholds["normal"]:
                dollar_volume_score = 2
            elif avg_dollar_volume >= self.liquidity_thresholds["low"]:
                dollar_volume_score = 1
            else:
                dollar_volume_score = 0
            
            # Spread score
            spread_bps = spread_analysis.get("spread_basis_points", 50)
            if spread_bps < 5:
                spread_score = 4
            elif spread_bps < 15:
                spread_score = 3
            elif spread_bps < 30:
                spread_score = 2
            elif spread_bps < 50:
                spread_score = 1
            else:
                spread_score = 0
            
            # Weighted average (dollar volume most important, then spread, then volume ratio)
            weighted_score = (dollar_volume_score * 0.5 + spread_score * 0.3 + volume_score * 0.2)
            
            # Convert to liquidity level
            if weighted_score >= 3.5:
                return LiquidityLevel.VERY_HIGH
            elif weighted_score >= 2.5:
                return LiquidityLevel.HIGH
            elif weighted_score >= 1.5:
                return LiquidityLevel.NORMAL
            elif weighted_score >= 0.5:
                return LiquidityLevel.LOW
            else:
                return LiquidityLevel.VERY_LOW
                
        except Exception as e:
            logger.error(f"Error determining liquidity level: {e}")
            return LiquidityLevel.NORMAL
    
    def _generate_execution_recommendations(self, liquidity_level: LiquidityLevel, 
                                          volume_analysis: Dict, 
                                          market_hours_analysis: Dict) -> Dict[str, Any]:
        """Generate execution recommendations based on liquidity analysis."""
        try:
            recommendations = {
                "recommended_order_type": OrderType.MARKET,
                "max_participation_rate": 0.10,
                "execution_strategy": "standard",
                "timing_recommendations": [],
                "risk_warnings": []
            }
            
            # Base recommendations by liquidity level
            if liquidity_level == LiquidityLevel.VERY_HIGH:
                recommendations.update({
                    "recommended_order_type": OrderType.MARKET,
                    "max_participation_rate": 0.20,
                    "execution_strategy": "aggressive",
                    "timing_recommendations": ["Can execute large orders immediately"]
                })
                
            elif liquidity_level == LiquidityLevel.HIGH:
                recommendations.update({
                    "recommended_order_type": OrderType.MARKET,
                    "max_participation_rate": 0.15,
                    "execution_strategy": "normal",
                    "timing_recommendations": ["Good conditions for immediate execution"]
                })
                
            elif liquidity_level == LiquidityLevel.NORMAL:
                recommendations.update({
                    "recommended_order_type": OrderType.LIMIT,
                    "max_participation_rate": 0.10,
                    "execution_strategy": "patient",
                    "timing_recommendations": ["Use limit orders for better pricing"]
                })
                
            elif liquidity_level == LiquidityLevel.LOW:
                recommendations.update({
                    "recommended_order_type": OrderType.LIMIT,
                    "max_participation_rate": 0.05,
                    "execution_strategy": "cautious",
                    "timing_recommendations": ["Break up large orders", "Use TWAP strategy"],
                    "risk_warnings": ["Low liquidity may cause slippage"]
                })
                
            else:  # VERY_LOW
                recommendations.update({
                    "recommended_order_type": OrderType.LIMIT,
                    "max_participation_rate": 0.02,
                    "execution_strategy": "very_cautious",
                    "timing_recommendations": ["Avoid large orders", "Consider waiting for better liquidity"],
                    "risk_warnings": ["Very low liquidity", "High slippage risk", "Consider alternative timing"]
                })
            
            # Market hours adjustments
            market_session = market_hours_analysis.get("market_session", MarketHours.REGULAR_HOURS)
            liquidity_impact = market_hours_analysis.get("liquidity_impact", "normal")
            
            if market_session in [MarketHours.PRE_MARKET, MarketHours.AFTER_HOURS]:
                recommendations["max_participation_rate"] *= 0.5
                recommendations["risk_warnings"].append("Extended hours trading - reduced liquidity")
                
            if liquidity_impact == "very_low":
                recommendations["timing_recommendations"].append("Consider waiting for regular market hours")
                
            # Volume pattern adjustments
            volume_pattern = volume_analysis.get("volume_pattern", "normal")
            if volume_pattern == "declining":
                recommendations["risk_warnings"].append("Declining volume trend")
                recommendations["max_participation_rate"] *= 0.8
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating execution recommendations: {e}")
            return {"error": str(e)}
    
    def _classify_volume_pattern(self, volume_ratio: float, volume_trend: float) -> str:
        """Classify current volume pattern."""
        try:
            if volume_ratio > 1.5 and volume_trend > 0.1:
                return "surging"
            elif volume_ratio > 1.2:
                return "elevated"
            elif volume_ratio < 0.5 and volume_trend < -0.1:
                return "declining"
            elif volume_ratio < 0.8:
                return "below_average"
            else:
                return "normal"
                
        except Exception as e:
            logger.error(f"Error classifying volume pattern: {e}")
            return "normal"
    
    def _is_option_expiry_week(self, current_time: datetime) -> bool:
        """Check if current week is options expiry week (simplified)."""
        try:
            # Third Friday of the month is typically options expiry
            # This is a simplified check
            day = current_time.day
            weekday = current_time.weekday()
            
            # Rough estimate: if it's the third week of the month
            if 15 <= day <= 21:
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking option expiry: {e}")
            return False
    
    def _update_liquidity_history(self, symbol: str, analysis_result: Dict):
        """Update liquidity history for trend analysis."""
        try:
            if symbol not in self.liquidity_history:
                self.liquidity_history[symbol] = []
            
            # Store essential data only
            history_entry = {
                "timestamp": analysis_result["timestamp"],
                "liquidity_level": analysis_result["liquidity_level"],
                "avg_dollar_volume": analysis_result["dollar_volume_analysis"].get("avg_dollar_volume", 0),
                "volume_ratio": analysis_result["volume_analysis"].get("volume_ratio", 0)
            }
            
            self.liquidity_history[symbol].append(history_entry)
            
            # Keep only last 50 entries
            if len(self.liquidity_history[symbol]) > 50:
                self.liquidity_history[symbol] = self.liquidity_history[symbol][-50:]
                
        except Exception as e:
            logger.error(f"Error updating liquidity history: {e}")
    
    def get_liquidity_trend(self, symbol: str, lookback_periods: int = 10) -> Dict[str, Any]:
        """Get liquidity trend analysis for a symbol."""
        try:
            if symbol not in self.liquidity_history or len(self.liquidity_history[symbol]) < 2:
                return {"error": "Insufficient history"}
            
            recent_history = self.liquidity_history[symbol][-lookback_periods:]
            
            # Analyze trends
            dollar_volumes = [entry["avg_dollar_volume"] for entry in recent_history]
            volume_ratios = [entry["volume_ratio"] for entry in recent_history]
            
            # Calculate trends
            if len(dollar_volumes) >= 2:
                dollar_volume_trend = (dollar_volumes[-1] - dollar_volumes[0]) / dollar_volumes[0] if dollar_volumes[0] > 0 else 0
                avg_volume_ratio = statistics.mean(volume_ratios)
            else:
                dollar_volume_trend = 0
                avg_volume_ratio = 0
            
            # Liquidity level trend
            liquidity_levels = [entry["liquidity_level"] for entry in recent_history]
            level_values = [self._liquidity_level_to_numeric(level) for level in liquidity_levels]
            
            if len(level_values) >= 2:
                level_trend = level_values[-1] - level_values[0]
                if level_trend > 0:
                    trend_direction = "improving"
                elif level_trend < 0:
                    trend_direction = "deteriorating"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "unknown"
            
            return {
                "trend_direction": trend_direction,
                "dollar_volume_trend": dollar_volume_trend,
                "avg_volume_ratio": avg_volume_ratio,
                "periods_analyzed": len(recent_history),
                "current_level": liquidity_levels[-1] if liquidity_levels else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Error getting liquidity trend: {e}")
            return {"error": str(e)}
    
    def _liquidity_level_to_numeric(self, level: LiquidityLevel) -> int:
        """Convert liquidity level to numeric value for trend analysis."""
        level_map = {
            LiquidityLevel.VERY_LOW: 1,
            LiquidityLevel.LOW: 2,
            LiquidityLevel.NORMAL: 3,
            LiquidityLevel.HIGH: 4,
            LiquidityLevel.VERY_HIGH: 5
        }
        return level_map.get(level, 3)
    
    def calculate_optimal_order_size(self, target_size: int, liquidity_level: LiquidityLevel, 
                                   avg_volume: float, participation_rate: float = None) -> Dict[str, Any]:
        """Calculate optimal order size based on liquidity constraints."""
        try:
            # Use recommended participation rate if not provided
            if participation_rate is None:
                if liquidity_level == LiquidityLevel.VERY_HIGH:
                    participation_rate = 0.20
                elif liquidity_level == LiquidityLevel.HIGH:
                    participation_rate = 0.15
                elif liquidity_level == LiquidityLevel.NORMAL:
                    participation_rate = 0.10
                elif liquidity_level == LiquidityLevel.LOW:
                    participation_rate = 0.05
                else:  # VERY_LOW
                    participation_rate = 0.02
            
            # Calculate maximum size based on volume participation
            max_size_from_volume = int(avg_volume * participation_rate)
            
            # Apply liquidity constraints
            if target_size <= max_size_from_volume:
                return {
                    "recommended_size": target_size,
                    "execution_method": "single_order",
                    "estimated_slices": 1,
                    "size_adjustment": "none"
                }
            else:
                # Need to slice the order
                recommended_size = max_size_from_volume
                estimated_slices = max(2, int(target_size / max_size_from_volume))
                
                return {
                    "recommended_size": recommended_size,
                    "execution_method": "sliced_orders",
                    "estimated_slices": estimated_slices,
                    "size_adjustment": "reduced_for_liquidity",
                    "original_size": target_size,
                    "reduction_ratio": recommended_size / target_size
                }
                
        except Exception as e:
            logger.error(f"Error calculating optimal order size: {e}")
            return {"error": str(e)}


class LiquidityManager:
    """
    High-level liquidity management coordinator.
    
    Manages liquidity analysis across multiple symbols and provides
    portfolio-level liquidity insights and recommendations.
    """
    
    def __init__(self):
        """Initialize liquidity manager."""
        # AI-AGENT-REF: Portfolio-level liquidity management
        self.analyzer = LiquidityAnalyzer()
        self.symbol_liquidity = {}
        self.portfolio_liquidity_score = 0.0
        
        logger.info("LiquidityManager initialized")
    
    def update_symbol_liquidity(self, symbol: str, market_data: Dict, 
                              current_price: float = None) -> Dict[str, Any]:
        """Update liquidity analysis for a symbol."""
        try:
            analysis = self.analyzer.analyze_liquidity(symbol, market_data, current_price)
            
            if "error" not in analysis:
                self.symbol_liquidity[symbol] = analysis
                self._update_portfolio_liquidity_score()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error updating symbol liquidity: {e}")
            return {"error": str(e)}
    
    def get_portfolio_liquidity_summary(self) -> Dict[str, Any]:
        """Get portfolio-wide liquidity summary."""
        try:
            if not self.symbol_liquidity:
                return {"error": "No liquidity data available"}
            
            # Aggregate liquidity levels
            liquidity_distribution = {}
            total_symbols = len(self.symbol_liquidity)
            
            for symbol, analysis in self.symbol_liquidity.items():
                level = analysis.get("liquidity_level", LiquidityLevel.NORMAL)
                level_str = level.value if hasattr(level, 'value') else str(level)
                liquidity_distribution[level_str] = liquidity_distribution.get(level_str, 0) + 1
            
            # Calculate percentages
            liquidity_percentages = {
                level: count / total_symbols * 100 
                for level, count in liquidity_distribution.items()
            }
            
            # Overall portfolio liquidity assessment
            high_liquidity_pct = (liquidity_percentages.get("high", 0) + 
                                liquidity_percentages.get("very_high", 0))
            low_liquidity_pct = (liquidity_percentages.get("low", 0) + 
                               liquidity_percentages.get("very_low", 0))
            
            if high_liquidity_pct > 60:
                portfolio_assessment = "excellent"
            elif high_liquidity_pct > 40:
                portfolio_assessment = "good"
            elif low_liquidity_pct < 20:
                portfolio_assessment = "adequate"
            else:
                portfolio_assessment = "poor"
            
            return {
                "total_symbols": total_symbols,
                "liquidity_distribution": liquidity_distribution,
                "liquidity_percentages": liquidity_percentages,
                "portfolio_liquidity_score": self.portfolio_liquidity_score,
                "portfolio_assessment": portfolio_assessment,
                "high_liquidity_percentage": high_liquidity_pct,
                "low_liquidity_percentage": low_liquidity_pct,
                "last_updated": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio liquidity summary: {e}")
            return {"error": str(e)}
    
    def get_illiquid_positions(self, threshold: LiquidityLevel = LiquidityLevel.LOW) -> List[Dict]:
        """Get list of positions with liquidity below threshold."""
        try:
            threshold_numeric = self.analyzer._liquidity_level_to_numeric(threshold)
            illiquid_positions = []
            
            for symbol, analysis in self.symbol_liquidity.items():
                liquidity_level = analysis.get("liquidity_level", LiquidityLevel.NORMAL)
                level_numeric = self.analyzer._liquidity_level_to_numeric(liquidity_level)
                
                if level_numeric <= threshold_numeric:
                    illiquid_positions.append({
                        "symbol": symbol,
                        "liquidity_level": liquidity_level.value if hasattr(liquidity_level, 'value') else str(liquidity_level),
                        "avg_dollar_volume": analysis.get("dollar_volume_analysis", {}).get("avg_dollar_volume", 0),
                        "spread_bps": analysis.get("spread_analysis", {}).get("spread_basis_points", 0),
                        "last_updated": analysis.get("timestamp")
                    })
            
            return sorted(illiquid_positions, key=lambda x: x["avg_dollar_volume"])
            
        except Exception as e:
            logger.error(f"Error getting illiquid positions: {e}")
            return []
    
    def _update_portfolio_liquidity_score(self):
        """Update overall portfolio liquidity score."""
        try:
            if not self.symbol_liquidity:
                self.portfolio_liquidity_score = 0.0
                return
            
            # Calculate weighted average based on liquidity levels
            total_score = 0.0
            total_symbols = len(self.symbol_liquidity)
            
            for analysis in self.symbol_liquidity.values():
                liquidity_level = analysis.get("liquidity_level", LiquidityLevel.NORMAL)
                level_score = self.analyzer._liquidity_level_to_numeric(liquidity_level)
                total_score += level_score
            
            # Normalize to 0-1 scale
            self.portfolio_liquidity_score = (total_score / total_symbols - 1) / 4 if total_symbols > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error updating portfolio liquidity score: {e}")
            self.portfolio_liquidity_score = 0.0