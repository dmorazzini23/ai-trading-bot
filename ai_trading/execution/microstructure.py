"""
Market Microstructure Feature Engineering.

Advanced market microstructure analysis including bid-ask spreads, order flow,
market impact estimation, and liquidity metrics for institutional trading.
"""

import math
import statistics
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.constants import EXECUTION_PARAMETERS, RISK_PARAMETERS


class MarketRegimeFeature(Enum):
    """Market microstructure regime classification."""
    HIGH_FREQUENCY = "high_frequency"      # Tight spreads, high turnover
    INSTITUTIONAL = "institutional"        # Large block trades, wider spreads
    RETAIL_DOMINATED = "retail_dominated"  # Small orders, moderate spreads
    ILLIQUID = "illiquid"                 # Wide spreads, low volume
    STRESSED = "stressed"                  # Very wide spreads, erratic flow


@dataclass
class MarketMicrostructureData:
    """Container for market microstructure features."""
    symbol: str
    timestamp: datetime
    
    # Basic market data
    bid_price: float
    ask_price: float
    last_price: float
    bid_size: int
    ask_size: int
    volume: int
    
    # Derived features
    bid_ask_spread: float
    spread_bps: float
    quoted_spread_pct: float
    effective_spread_pct: float
    realized_spread_pct: float
    
    # Liquidity features
    market_depth: float
    order_imbalance: float
    price_impact_estimate: float
    
    # Flow features
    trade_intensity: float
    order_flow_toxicity: float
    information_content: float
    
    # Volatility features
    realized_volatility: float
    price_variance_ratio: float
    microstructure_noise: float


class BidAskSpreadAnalyzer:
    """Analyzes bid-ask spreads and related microstructure features."""
    
    def __init__(self):
        """Initialize bid-ask spread analyzer."""
        # AI-AGENT-REF: Bid-ask spread and microstructure analysis
        self.lookback_periods = 100  # Number of observations for calculations
        self.min_spread_bps = 1.0    # Minimum spread threshold
        
        logger.debug("BidAskSpreadAnalyzer initialized")
    
    def analyze_spread_features(self, market_data: Dict, trade_history: List[Dict]) -> Dict[str, float]:
        """
        Analyze spread-related features from market data.
        
        Args:
            market_data: Current market data (bid, ask, last, sizes)
            trade_history: Recent trade history for analysis
            
        Returns:
            Dictionary of spread features
        """
        try:
            # Extract basic data
            bid_price = market_data.get("bid_price", 0.0)
            ask_price = market_data.get("ask_price", 0.0)
            last_price = market_data.get("last_price", 0.0)
            bid_size = market_data.get("bid_size", 0)
            ask_size = market_data.get("ask_size", 0)
            
            features = {}
            
            # Basic spread calculations
            if bid_price > 0 and ask_price > 0:
                bid_ask_spread = ask_price - bid_price
                features["bid_ask_spread"] = bid_ask_spread
                
                # Spread in basis points
                mid_price = (bid_price + ask_price) / 2
                features["spread_bps"] = (bid_ask_spread / mid_price) * 10000 if mid_price > 0 else 0
                
                # Quoted spread percentage
                features["quoted_spread_pct"] = (bid_ask_spread / mid_price) * 100 if mid_price > 0 else 0
                
                # Relative spread position
                if last_price > 0:
                    relative_position = (last_price - mid_price) / bid_ask_spread if bid_ask_spread > 0 else 0
                    features["relative_spread_position"] = relative_position
            
            # Effective and realized spreads from trade history
            if trade_history:
                effective_spread, realized_spread = self._calculate_trade_spreads(trade_history)
                features["effective_spread_pct"] = effective_spread
                features["realized_spread_pct"] = realized_spread
            
            # Market depth features
            if bid_size > 0 and ask_size > 0:
                features["market_depth"] = bid_size + ask_size
                features["depth_imbalance"] = (bid_size - ask_size) / (bid_size + ask_size)
                features["bid_ask_size_ratio"] = bid_size / ask_size if ask_size > 0 else 0
            
            # Spread quality metrics
            features["spread_quality"] = self._assess_spread_quality(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing spread features: {e}")
            return {}
    
    def classify_spread_regime(self, spread_features: Dict[str, float]) -> MarketRegimeFeature:
        """
        Classify market microstructure regime based on spread characteristics.
        
        Args:
            spread_features: Dictionary of calculated spread features
            
        Returns:
            Market regime classification
        """
        try:
            spread_bps = spread_features.get("spread_bps", 0)
            market_depth = spread_features.get("market_depth", 0)
            spread_quality = spread_features.get("spread_quality", 0)
            
            # Classification logic
            if spread_bps < 5 and market_depth > 10000 and spread_quality > 0.8:
                return MarketRegimeFeature.HIGH_FREQUENCY
            elif spread_bps > 50 or market_depth < 1000:
                return MarketRegimeFeature.ILLIQUID
            elif spread_bps > 20 and spread_quality < 0.4:
                return MarketRegimeFeature.STRESSED
            elif market_depth > 5000 and 5 <= spread_bps <= 15:
                return MarketRegimeFeature.INSTITUTIONAL
            else:
                return MarketRegimeFeature.RETAIL_DOMINATED
            
        except Exception as e:
            logger.error(f"Error classifying spread regime: {e}")
            return MarketRegimeFeature.RETAIL_DOMINATED
    
    def _calculate_trade_spreads(self, trade_history: List[Dict]) -> Tuple[float, float]:
        """Calculate effective and realized spreads from trade history."""
        try:
            if len(trade_history) < 2:
                return 0.0, 0.0
            
            effective_spreads = []
            realized_spreads = []
            
            for i, trade in enumerate(trade_history[-self.lookback_periods:]):
                trade_price = trade.get("price", 0)
                trade_size = trade.get("size", 0)
                trade_side = trade.get("side", "unknown")
                
                if trade_price <= 0:
                    continue
                
                # Quote data at trade time
                quote_mid = trade.get("quote_mid", trade_price)
                
                # Effective spread (immediate cost)
                if quote_mid > 0:
                    if trade_side == "buy":
                        effective_spread = 2 * (trade_price - quote_mid) / quote_mid
                    elif trade_side == "sell":
                        effective_spread = 2 * (quote_mid - trade_price) / quote_mid
                    else:
                        effective_spread = 2 * abs(trade_price - quote_mid) / quote_mid
                    
                    effective_spreads.append(effective_spread * 100)  # Convert to percentage
                
                # Realized spread (permanent impact after 5 minutes)
                if i < len(trade_history) - 10:  # Need future data
                    future_price = trade_history[i + 10].get("price", trade_price)
                    if trade_side == "buy":
                        realized_spread = 2 * (trade_price - future_price) / trade_price
                    elif trade_side == "sell":
                        realized_spread = 2 * (future_price - trade_price) / trade_price
                    else:
                        realized_spread = 0
                    
                    realized_spreads.append(realized_spread * 100)
            
            avg_effective = statistics.mean(effective_spreads) if effective_spreads else 0.0
            avg_realized = statistics.mean(realized_spreads) if realized_spreads else 0.0
            
            return avg_effective, avg_realized
            
        except Exception as e:
            logger.error(f"Error calculating trade spreads: {e}")
            return 0.0, 0.0
    
    def _assess_spread_quality(self, spread_features: Dict[str, float]) -> float:
        """Assess overall spread quality (0-1 scale)."""
        try:
            spread_bps = spread_features.get("spread_bps", 100)
            market_depth = spread_features.get("market_depth", 0)
            depth_imbalance = abs(spread_features.get("depth_imbalance", 0))
            
            # Tighter spreads are better (score decreases with spread)
            spread_score = max(0, 1 - spread_bps / 50)  # Normalize around 50 bps
            
            # Higher depth is better
            depth_score = min(1, market_depth / 10000)  # Normalize around 10k shares
            
            # Lower imbalance is better
            imbalance_score = max(0, 1 - depth_imbalance)
            
            # Weighted average
            quality_score = spread_score * 0.5 + depth_score * 0.3 + imbalance_score * 0.2
            return max(0, min(1, quality_score))
            
        except Exception:
            return 0.5


class OrderFlowAnalyzer:
    """Analyzes order flow patterns and toxicity."""
    
    def __init__(self):
        """Initialize order flow analyzer."""
        # AI-AGENT-REF: Order flow analysis for market microstructure
        self.flow_window = 20      # Window for flow calculations
        self.toxicity_threshold = 0.3  # Threshold for toxic flow detection
        
        logger.debug("OrderFlowAnalyzer initialized")
    
    def analyze_order_flow(self, trade_data: List[Dict], quote_data: List[Dict]) -> Dict[str, float]:
        """
        Analyze order flow characteristics and toxicity.
        
        Args:
            trade_data: Recent trade data with prices, sizes, sides
            quote_data: Recent quote data with bid/ask updates
            
        Returns:
            Dictionary of order flow features
        """
        try:
            features = {}
            
            # Trade intensity analysis
            features["trade_intensity"] = self._calculate_trade_intensity(trade_data)
            
            # Order imbalance from quotes
            features["order_imbalance"] = self._calculate_order_imbalance(quote_data)
            
            # Flow toxicity
            features["order_flow_toxicity"] = self._calculate_flow_toxicity(trade_data, quote_data)
            
            # Information content
            features["information_content"] = self._estimate_information_content(trade_data)
            
            # Volume-weighted average price deviation
            features["vwap_deviation"] = self._calculate_vwap_deviation(trade_data)
            
            # Trade size distribution
            trade_sizes = [trade.get("size", 0) for trade in trade_data[-self.flow_window:]]
            if trade_sizes:
                features["avg_trade_size"] = statistics.mean(trade_sizes)
                features["trade_size_volatility"] = statistics.stdev(trade_sizes) if len(trade_sizes) > 1 else 0
                features["large_trade_ratio"] = sum(1 for size in trade_sizes if size > 1000) / len(trade_sizes)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing order flow: {e}")
            return {}
    
    def detect_toxic_flow(self, flow_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect potentially toxic order flow.
        
        Args:
            flow_features: Calculated order flow features
            
        Returns:
            Dictionary with toxicity assessment
        """
        try:
            toxicity_score = flow_features.get("order_flow_toxicity", 0)
            trade_intensity = flow_features.get("trade_intensity", 0)
            order_imbalance = abs(flow_features.get("order_imbalance", 0))
            
            # Combined toxicity assessment
            is_toxic = (
                toxicity_score > self.toxicity_threshold or
                trade_intensity > 5.0 or  # Very high intensity
                order_imbalance > 0.8     # Severe imbalance
            )
            
            risk_level = "high" if is_toxic else "normal"
            if toxicity_score > 0.7 or trade_intensity > 10:
                risk_level = "extreme"
            
            return {
                "is_toxic": is_toxic,
                "risk_level": risk_level,
                "toxicity_score": toxicity_score,
                "primary_concern": self._identify_primary_concern(flow_features),
                "recommendations": self._generate_flow_recommendations(flow_features)
            }
            
        except Exception as e:
            logger.error(f"Error detecting toxic flow: {e}")
            return {"is_toxic": False, "risk_level": "unknown", "error": str(e)}
    
    def _calculate_trade_intensity(self, trade_data: List[Dict]) -> float:
        """Calculate trade intensity (trades per minute)."""
        try:
            if len(trade_data) < 2:
                return 0.0
            
            recent_trades = trade_data[-self.flow_window:]
            
            # Calculate time span
            timestamps = [trade.get("timestamp") for trade in recent_trades if trade.get("timestamp")]
            if len(timestamps) < 2:
                return 0.0
            
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 60  # Minutes
            if time_span <= 0:
                return 0.0
            
            return len(recent_trades) / time_span
            
        except Exception:
            return 0.0
    
    def _calculate_order_imbalance(self, quote_data: List[Dict]) -> float:
        """Calculate order imbalance from quote updates."""
        try:
            if not quote_data:
                return 0.0
            
            recent_quotes = quote_data[-self.flow_window:]
            
            total_bid_size = sum(quote.get("bid_size", 0) for quote in recent_quotes)
            total_ask_size = sum(quote.get("ask_size", 0) for quote in recent_quotes)
            
            if total_bid_size + total_ask_size == 0:
                return 0.0
            
            return (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
            
        except Exception:
            return 0.0
    
    def _calculate_flow_toxicity(self, trade_data: List[Dict], quote_data: List[Dict]) -> float:
        """Calculate order flow toxicity using adverse selection measure."""
        try:
            if len(trade_data) < 5 or len(quote_data) < 5:
                return 0.0
            
            toxicity_measures = []
            
            # For each trade, measure subsequent quote movement
            for i, trade in enumerate(trade_data[-10:]):  # Last 10 trades
                trade_price = trade.get("price", 0)
                trade_side = trade.get("side", "unknown")
                trade_time = trade.get("timestamp")
                
                if not all([trade_price, trade_side, trade_time]):
                    continue
                
                # Find quotes after this trade
                future_quotes = [
                    q for q in quote_data 
                    if q.get("timestamp", datetime.min) > trade_time
                ][:3]  # Next 3 quotes
                
                if not future_quotes:
                    continue
                
                # Measure price movement in direction of trade
                pre_trade_mid = trade.get("quote_mid_before", trade_price)
                post_trade_mid = statistics.mean([
                    (q.get("bid_price", 0) + q.get("ask_price", 0)) / 2 
                    for q in future_quotes
                    if q.get("bid_price", 0) > 0 and q.get("ask_price", 0) > 0
                ])
                
                if pre_trade_mid > 0 and post_trade_mid > 0:
                    price_movement = (post_trade_mid - pre_trade_mid) / pre_trade_mid
                    
                    # Toxicity: trades that move price in same direction
                    if trade_side == "buy" and price_movement > 0:
                        toxicity_measures.append(abs(price_movement))
                    elif trade_side == "sell" and price_movement < 0:
                        toxicity_measures.append(abs(price_movement))
            
            return statistics.mean(toxicity_measures) if toxicity_measures else 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_information_content(self, trade_data: List[Dict]) -> float:
        """Estimate information content of trades."""
        try:
            if len(trade_data) < 3:
                return 0.0
            
            # Use price variance to volume ratio as proxy for information
            recent_trades = trade_data[-self.flow_window:]
            
            prices = [trade.get("price", 0) for trade in recent_trades if trade.get("price", 0) > 0]
            volumes = [trade.get("size", 0) for trade in recent_trades if trade.get("size", 0) > 0]
            
            if len(prices) < 2 or not volumes:
                return 0.0
            
            # Calculate price variance
            price_variance = statistics.variance(prices)
            total_volume = sum(volumes)
            
            # Information content proxy
            information_content = price_variance / (total_volume / len(volumes)) if total_volume > 0 else 0
            
            # Normalize to 0-1 scale
            return min(1.0, information_content * 1000)  # Scaling factor
            
        except Exception:
            return 0.0
    
    def _calculate_vwap_deviation(self, trade_data: List[Dict]) -> float:
        """Calculate deviation from volume-weighted average price."""
        try:
            if len(trade_data) < 2:
                return 0.0
            
            recent_trades = trade_data[-self.flow_window:]
            
            total_volume = 0
            vwap_numerator = 0
            
            for trade in recent_trades:
                price = trade.get("price", 0)
                size = trade.get("size", 0)
                
                if price > 0 and size > 0:
                    vwap_numerator += price * size
                    total_volume += size
            
            if total_volume == 0:
                return 0.0
            
            vwap = vwap_numerator / total_volume
            current_price = recent_trades[-1].get("price", 0)
            
            if vwap > 0 and current_price > 0:
                return abs(current_price - vwap) / vwap
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _identify_primary_concern(self, flow_features: Dict[str, float]) -> str:
        """Identify the primary concern from flow analysis."""
        concerns = []
        
        if flow_features.get("order_flow_toxicity", 0) > 0.4:
            concerns.append("High flow toxicity")
        
        if flow_features.get("trade_intensity", 0) > 5:
            concerns.append("Elevated trade intensity")
        
        if abs(flow_features.get("order_imbalance", 0)) > 0.6:
            concerns.append("Significant order imbalance")
        
        if flow_features.get("information_content", 0) > 0.7:
            concerns.append("High information content")
        
        return concerns[0] if concerns else "Normal flow characteristics"
    
    def _generate_flow_recommendations(self, flow_features: Dict[str, float]) -> List[str]:
        """Generate recommendations based on flow analysis."""
        recommendations = []
        
        if flow_features.get("order_flow_toxicity", 0) > 0.3:
            recommendations.append("Use limit orders to avoid adverse selection")
            recommendations.append("Consider breaking large orders into smaller sizes")
        
        if flow_features.get("trade_intensity", 0) > 5:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider delaying execution until flow normalizes")
        
        if abs(flow_features.get("order_imbalance", 0)) > 0.5:
            recommendations.append("Monitor for potential price movement")
            recommendations.append("Adjust order timing based on imbalance direction")
        
        return recommendations


class MarketMicrostructureEngine:
    """
    Comprehensive market microstructure analysis engine.
    
    Combines spread analysis, order flow analysis, and impact estimation
    for advanced execution decision making.
    """
    
    def __init__(self):
        """Initialize market microstructure engine."""
        # AI-AGENT-REF: Comprehensive market microstructure analysis
        self.spread_analyzer = BidAskSpreadAnalyzer()
        self.flow_analyzer = OrderFlowAnalyzer()
        
        # Impact estimation parameters
        self.impact_parameters = {
            "temporary_impact_factor": 0.5,
            "permanent_impact_factor": 0.3,
            "participation_threshold": 0.1
        }
        
        logger.info("MarketMicrostructureEngine initialized")
    
    def analyze_market_microstructure(self, symbol: str, market_data: Dict,
                                    trade_history: List[Dict], quote_history: List[Dict]) -> MarketMicrostructureData:
        """
        Perform comprehensive market microstructure analysis.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            trade_history: Recent trade history
            quote_history: Recent quote history
            
        Returns:
            MarketMicrostructureData with all calculated features
        """
        try:
            # Extract basic market data
            bid_price = market_data.get("bid_price", 0.0)
            ask_price = market_data.get("ask_price", 0.0)
            last_price = market_data.get("last_price", 0.0)
            bid_size = market_data.get("bid_size", 0)
            ask_size = market_data.get("ask_size", 0)
            volume = market_data.get("volume", 0)
            
            # Spread analysis
            spread_features = self.spread_analyzer.analyze_spread_features(market_data, trade_history)
            
            # Order flow analysis
            flow_features = self.flow_analyzer.analyze_order_flow(trade_history, quote_history)
            
            # Price impact estimation
            impact_features = self._estimate_price_impact(market_data, trade_history)
            
            # Volatility analysis
            volatility_features = self._analyze_microstructure_volatility(trade_history)
            
            # Create comprehensive microstructure data
            microstructure_data = MarketMicrostructureData(
                symbol=symbol,
                timestamp=datetime.now(),
                
                # Basic market data
                bid_price=bid_price,
                ask_price=ask_price,
                last_price=last_price,
                bid_size=bid_size,
                ask_size=ask_size,
                volume=volume,
                
                # Spread features
                bid_ask_spread=spread_features.get("bid_ask_spread", 0.0),
                spread_bps=spread_features.get("spread_bps", 0.0),
                quoted_spread_pct=spread_features.get("quoted_spread_pct", 0.0),
                effective_spread_pct=spread_features.get("effective_spread_pct", 0.0),
                realized_spread_pct=spread_features.get("realized_spread_pct", 0.0),
                
                # Liquidity features
                market_depth=spread_features.get("market_depth", 0.0),
                order_imbalance=flow_features.get("order_imbalance", 0.0),
                price_impact_estimate=impact_features.get("estimated_impact", 0.0),
                
                # Flow features
                trade_intensity=flow_features.get("trade_intensity", 0.0),
                order_flow_toxicity=flow_features.get("order_flow_toxicity", 0.0),
                information_content=flow_features.get("information_content", 0.0),
                
                # Volatility features
                realized_volatility=volatility_features.get("realized_volatility", 0.0),
                price_variance_ratio=volatility_features.get("price_variance_ratio", 0.0),
                microstructure_noise=volatility_features.get("microstructure_noise", 0.0)
            )
            
            return microstructure_data
            
        except Exception as e:
            logger.error(f"Error analyzing market microstructure for {symbol}: {e}")
            return self._create_default_microstructure_data(symbol)
    
    def estimate_execution_impact(self, order_size: int, microstructure_data: MarketMicrostructureData) -> Dict[str, float]:
        """
        Estimate execution impact based on microstructure analysis.
        
        Args:
            order_size: Proposed order size
            microstructure_data: Current microstructure data
            
        Returns:
            Dictionary with impact estimates
        """
        try:
            # Calculate participation rate
            avg_volume = microstructure_data.volume
            participation_rate = order_size / avg_volume if avg_volume > 0 else 1.0
            
            # Base impact from spread
            spread_impact_bps = microstructure_data.spread_bps * 0.5  # Half spread crossing
            
            # Market impact based on participation
            market_impact_bps = self._calculate_market_impact(participation_rate, microstructure_data)
            
            # Temporary impact (recovers quickly)
            temporary_impact_bps = market_impact_bps * self.impact_parameters["temporary_impact_factor"]
            
            # Permanent impact (persists)
            permanent_impact_bps = market_impact_bps * self.impact_parameters["permanent_impact_factor"]
            
            # Total expected impact
            total_impact_bps = spread_impact_bps + temporary_impact_bps + permanent_impact_bps
            
            # Timing risk based on volatility
            timing_risk_bps = microstructure_data.realized_volatility * 100 * math.sqrt(participation_rate)
            
            return {
                "spread_impact_bps": spread_impact_bps,
                "market_impact_bps": market_impact_bps,
                "temporary_impact_bps": temporary_impact_bps,
                "permanent_impact_bps": permanent_impact_bps,
                "total_impact_bps": total_impact_bps,
                "timing_risk_bps": timing_risk_bps,
                "participation_rate": participation_rate,
                "confidence_level": self._assess_impact_confidence(microstructure_data)
            }
            
        except Exception as e:
            logger.error(f"Error estimating execution impact: {e}")
            return {"error": str(e), "total_impact_bps": 999.0}
    
    def _estimate_price_impact(self, market_data: Dict, trade_history: List[Dict]) -> Dict[str, float]:
        """Estimate price impact features."""
        try:
            features = {}
            
            # Current market depth
            bid_size = market_data.get("bid_size", 0)
            ask_size = market_data.get("ask_size", 0)
            depth = bid_size + ask_size
            
            # Average trade size
            recent_trades = trade_history[-20:] if len(trade_history) >= 20 else trade_history
            trade_sizes = [trade.get("size", 0) for trade in recent_trades if trade.get("size", 0) > 0]
            avg_trade_size = statistics.mean(trade_sizes) if trade_sizes else 0
            
            # Estimated impact per share (simplified model)
            if depth > 0 and avg_trade_size > 0:
                impact_per_1000_shares = (avg_trade_size / depth) * 0.1  # 10% of ratio as impact
                features["estimated_impact"] = impact_per_1000_shares
            else:
                features["estimated_impact"] = 0.05  # Default 5% impact
            
            return features
            
        except Exception:
            return {"estimated_impact": 0.05}
    
    def _analyze_microstructure_volatility(self, trade_history: List[Dict]) -> Dict[str, float]:
        """Analyze microstructure-specific volatility measures."""
        try:
            features = {}
            
            if len(trade_history) < 10:
                return {"realized_volatility": 0.0, "price_variance_ratio": 1.0, "microstructure_noise": 0.0}
            
            # Extract prices
            prices = [trade.get("price", 0) for trade in trade_history[-50:] if trade.get("price", 0) > 0]
            
            if len(prices) < 10:
                return {"realized_volatility": 0.0, "price_variance_ratio": 1.0, "microstructure_noise": 0.0}
            
            # Calculate returns
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            
            # Realized volatility
            if len(returns) > 1:
                features["realized_volatility"] = statistics.stdev(returns) * math.sqrt(252 * 390)  # Annualized
            else:
                features["realized_volatility"] = 0.0
            
            # Price variance ratio (test for mean reversion)
            if len(prices) >= 20:
                variance_1 = statistics.variance(prices[-10:]) if len(prices[-10:]) > 1 else 0
                variance_2 = statistics.variance(prices[-20:-10]) if len(prices[-20:-10]) > 1 else 0
                features["price_variance_ratio"] = variance_1 / variance_2 if variance_2 > 0 else 1.0
            else:
                features["price_variance_ratio"] = 1.0
            
            # Microstructure noise estimate
            if len(returns) >= 5:
                # Autocorrelation of returns as noise proxy
                lag1_corr = self._calculate_autocorrelation(returns, lag=1)
                features["microstructure_noise"] = abs(lag1_corr)
            else:
                features["microstructure_noise"] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing microstructure volatility: {e}")
            return {"realized_volatility": 0.0, "price_variance_ratio": 1.0, "microstructure_noise": 0.0}
    
    def _calculate_market_impact(self, participation_rate: float, microstructure_data: MarketMicrostructureData) -> float:
        """Calculate market impact in basis points."""
        try:
            # Base impact from square-root law
            base_impact = 100 * math.sqrt(participation_rate)  # 100 bps at 100% participation
            
            # Adjust for market conditions
            spread_adjustment = microstructure_data.spread_bps / 10  # Higher spreads = higher impact
            volatility_adjustment = microstructure_data.realized_volatility * 100  # Higher vol = higher impact
            toxicity_adjustment = microstructure_data.order_flow_toxicity * 50  # Toxic flow = higher impact
            
            total_impact = base_impact + spread_adjustment + volatility_adjustment + toxicity_adjustment
            
            return min(200, max(1, total_impact))  # Cap between 1-200 bps
            
        except Exception:
            return 50.0  # Default impact estimate
    
    def _assess_impact_confidence(self, microstructure_data: MarketMicrostructureData) -> float:
        """Assess confidence in impact estimates (0-1)."""
        try:
            confidence_factors = []
            
            # Data quality factors
            if microstructure_data.bid_size > 0 and microstructure_data.ask_size > 0:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)
            
            # Spread quality
            if microstructure_data.spread_bps < 20:
                confidence_factors.append(0.9)
            elif microstructure_data.spread_bps < 50:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
            
            # Flow characteristics
            if microstructure_data.order_flow_toxicity < 0.3:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            return statistics.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_autocorrelation(self, data: List[float], lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        try:
            if len(data) < lag + 2:
                return 0.0
            
            n = len(data)
            mean_val = statistics.mean(data)
            
            # Calculate autocovariance
            autocovariance = sum(
                (data[i] - mean_val) * (data[i - lag] - mean_val)
                for i in range(lag, n)
            ) / (n - lag)
            
            # Calculate variance
            variance = sum((x - mean_val) ** 2 for x in data) / n
            
            return autocovariance / variance if variance > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _create_default_microstructure_data(self, symbol: str) -> MarketMicrostructureData:
        """Create default microstructure data when analysis fails."""
        return MarketMicrostructureData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=0.0,
            ask_price=0.0,
            last_price=0.0,
            bid_size=0,
            ask_size=0,
            volume=0,
            bid_ask_spread=0.0,
            spread_bps=0.0,
            quoted_spread_pct=0.0,
            effective_spread_pct=0.0,
            realized_spread_pct=0.0,
            market_depth=0.0,
            order_imbalance=0.0,
            price_impact_estimate=0.0,
            trade_intensity=0.0,
            order_flow_toxicity=0.0,
            information_content=0.0,
            realized_volatility=0.0,
            price_variance_ratio=1.0,
            microstructure_noise=0.0
        )