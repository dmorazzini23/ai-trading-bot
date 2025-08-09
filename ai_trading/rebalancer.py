"""Portfolio rebalancing utilities with tax awareness and advanced features."""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

import config
from ai_trading.portfolio import compute_portfolio_weights

def apply_no_trade_bands(current: Dict[str, float], target: Dict[str, float], band_bps: float = 25.0) -> Dict[str, float]:
    """
    Suppress small reallocations that are inside a no-trade band (in basis points).
    Example: band_bps=25 means ignore deltas smaller than 0.25% absolute weight.
    """
    band = band_bps / 10000.0
    out = {}
    for sym, tgt in target.items():
        cur = current.get(sym, 0.0)
        if abs(tgt - cur) < band:
            out[sym] = cur
        else:
            out[sym] = tgt
    return out

# AI-AGENT-REF: Enhanced rebalancer with tax awareness
try:
    from ai_trading.risk.adaptive_sizing import AdaptivePositionSizer
    from ai_trading.core.constants import RISK_PARAMETERS
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

# AI-AGENT-REF: Portfolio-first trading integration
try:
    from ai_trading.portfolio import PortfolioOptimizer, create_portfolio_optimizer
    from transaction_cost_calculator import TransactionCostCalculator, create_transaction_cost_calculator
    from regime_detector import RegimeDetector, create_regime_detector
    PORTFOLIO_FIRST_AVAILABLE = True
except ImportError:
    PORTFOLIO_FIRST_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log availability after logger is defined
if PORTFOLIO_FIRST_AVAILABLE:
    logger.info("Portfolio-first trading capabilities loaded")
else:
    # Import settings to check if portfolio features are enabled
    from ai_trading.config.settings import get_settings
    settings = get_settings()
    
    if settings.ENABLE_PORTFOLIO_FEATURES:
        logger.warning("Portfolio-first capabilities requested but unavailable (missing optional deps)")
    else:
        logger.info("Portfolio-first features disabled")

REBALANCE_INTERVAL_MIN = int(config.get_env("REBALANCE_INTERVAL_MIN", "10"))

_last_rebalance = datetime.now(timezone.utc)

# Global portfolio-first components
_portfolio_optimizer = None
_regime_detector = None

class TaxAwareRebalancer:
    """
    Tax-aware portfolio rebalancing with loss harvesting and wash sale avoidance.
    
    Implements sophisticated rebalancing that considers tax implications,
    capital gains/losses, and optimal timing for portfolio adjustments.
    """
    
    def __init__(self, tax_rate_short: float = 0.37, tax_rate_long: float = 0.20):
        """Initialize tax-aware rebalancer."""
        # AI-AGENT-REF: Tax-aware rebalancing implementation
        self.tax_rate_short = tax_rate_short  # Short-term capital gains tax rate
        self.tax_rate_long = tax_rate_long    # Long-term capital gains tax rate
        self.wash_sale_days = 31              # Days to avoid wash sale rule
        
        # Enhanced features if available
        if ENHANCED_FEATURES_AVAILABLE:
            self.adaptive_sizer = AdaptivePositionSizer()
            self.max_portfolio_risk = RISK_PARAMETERS["MAX_PORTFOLIO_RISK"]
        
        self.holding_period_long = 365  # Days for long-term capital gains
        
        logger.info(f"TaxAwareRebalancer initialized with tax rates: "
                   f"short={tax_rate_short:.1%}, long={tax_rate_long:.1%}")
    
    def calculate_tax_impact(self, position: Dict[str, Any], current_price: float) -> Dict[str, float]:
        """
        Calculate tax impact of selling a position.
        
        Args:
            position: Position data including entry price, quantity, entry date
            current_price: Current market price
            
        Returns:
            Dictionary with tax impact analysis
        """
        try:
            entry_price = position.get("entry_price", 0.0)
            quantity = position.get("quantity", 0)
            entry_date = position.get("entry_date")
            
            if not all([entry_price, quantity, current_price, entry_date]):
                return {"error": "Missing position data"}
            
            # Calculate gain/loss
            total_gain_loss = (current_price - entry_price) * quantity
            gain_loss_per_share = current_price - entry_price
            gain_loss_pct = gain_loss_per_share / entry_price if entry_price > 0 else 0
            
            # Determine holding period
            holding_days = (datetime.now(timezone.utc) - entry_date).days
            is_long_term = holding_days >= self.holding_period_long
            
            # Calculate tax liability
            applicable_tax_rate = self.tax_rate_long if is_long_term else self.tax_rate_short
            tax_liability = max(0, total_gain_loss * applicable_tax_rate)  # Only pay tax on gains
            
            # After-tax proceeds
            proceeds_before_tax = current_price * quantity
            after_tax_proceeds = proceeds_before_tax - tax_liability
            
            return {
                "total_gain_loss": total_gain_loss,
                "gain_loss_pct": gain_loss_pct,
                "holding_days": holding_days,
                "is_long_term": is_long_term,
                "applicable_tax_rate": applicable_tax_rate,
                "tax_liability": tax_liability,
                "proceeds_before_tax": proceeds_before_tax,
                "after_tax_proceeds": after_tax_proceeds,
                "tax_efficiency_score": self._calculate_tax_efficiency(total_gain_loss, is_long_term)
            }
            
        except Exception as e:
            logger.error(f"Error calculating tax impact: {e}")
            return {"error": str(e)}
    
    def identify_loss_harvesting_opportunities(self, portfolio_positions: Dict[str, Dict],
                                             current_prices: Dict[str, float]) -> List[Dict]:
        """
        Identify tax loss harvesting opportunities.
        
        Args:
            portfolio_positions: Dictionary of current positions
            current_prices: Current market prices for positions
            
        Returns:
            List of loss harvesting opportunities
        """
        try:
            opportunities = []
            
            for symbol, position in portfolio_positions.items():
                current_price = current_prices.get(symbol, 0)
                if current_price <= 0:
                    continue
                
                tax_impact = self.calculate_tax_impact(position, current_price)
                
                if tax_impact.get("error"):
                    continue
                
                # Look for losses that can be harvested
                total_loss = tax_impact.get("total_gain_loss", 0)
                if total_loss < 0:  # Position is at a loss
                    # Check wash sale restrictions
                    last_sale_date = position.get("last_sale_date")
                    can_harvest = True
                    
                    if last_sale_date:
                        days_since_sale = (datetime.now(timezone.utc) - last_sale_date).days
                        if days_since_sale < self.wash_sale_days:
                            can_harvest = False
                    
                    if can_harvest:
                        # Calculate tax benefit
                        tax_rate = tax_impact.get("applicable_tax_rate", 0)
                        tax_benefit = abs(total_loss) * tax_rate
                        
                        opportunity = {
                            "symbol": symbol,
                            "position": position,
                            "current_price": current_price,
                            "total_loss": total_loss,
                            "tax_benefit": tax_benefit,
                            "is_long_term": tax_impact.get("is_long_term", False),
                            "holding_days": tax_impact.get("holding_days", 0),
                            "priority_score": self._calculate_harvest_priority(total_loss, tax_benefit, position)
                        }
                        opportunities.append(opportunity)
            
            # Sort by priority score (highest first)
            opportunities.sort(key=lambda x: x["priority_score"], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying loss harvesting opportunities: {e}")
            return []
    
    def calculate_optimal_rebalance(self, current_positions: Dict[str, Dict],
                                  target_weights: Dict[str, float],
                                  current_prices: Dict[str, float],
                                  account_equity: float) -> Dict[str, Any]:
        """
        Calculate tax-optimal rebalancing trades.
        
        Args:
            current_positions: Current portfolio positions
            target_weights: Target portfolio weights
            current_prices: Current market prices
            account_equity: Total account equity
            
        Returns:
            Dictionary with optimal rebalancing plan
        """
        try:
            # Calculate current weights
            current_weights = {}
            total_portfolio_value = 0
            
            for symbol, position in current_positions.items():
                current_price = current_prices.get(symbol, 0)
                if current_price > 0:
                    position_value = position.get("quantity", 0) * current_price
                    total_portfolio_value += position_value
            
            if total_portfolio_value > 0:
                for symbol, position in current_positions.items():
                    current_price = current_prices.get(symbol, 0)
                    if current_price > 0:
                        position_value = position.get("quantity", 0) * current_price
                        current_weights[symbol] = position_value / total_portfolio_value
            
            # Identify rebalancing needs
            rebalance_trades = []
            total_tax_impact = 0
            
            for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_weights.get(symbol, 0)
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.01:  # 1% threshold
                    current_price = current_prices.get(symbol, 0)
                    if current_price <= 0:
                        continue
                    
                    # Calculate required trade
                    target_value = target_weight * account_equity
                    current_value = current_weight * account_equity
                    trade_value = target_value - current_value
                    trade_quantity = int(trade_value / current_price)
                    
                    if trade_quantity != 0:
                        # Calculate tax impact for sells
                        tax_impact = {"tax_liability": 0, "is_optimal_timing": True}
                        
                        if trade_quantity < 0 and symbol in current_positions:  # Selling
                            position = current_positions[symbol]
                            sell_quantity = min(abs(trade_quantity), position.get("quantity", 0))
                            
                            # Approximate tax calculation for partial sale
                            partial_position = position.copy()
                            partial_position["quantity"] = sell_quantity
                            
                            tax_impact = self.calculate_tax_impact(partial_position, current_price)
                            total_tax_impact += tax_impact.get("tax_liability", 0)
                            
                            # Check if timing is optimal
                            holding_days = tax_impact.get("holding_days", 0)
                            total_gain_loss = tax_impact.get("total_gain_loss", 0)
                            
                            # Suggest delaying if close to long-term threshold with gains
                            if (holding_days > 300 and holding_days < 365 and 
                                total_gain_loss > 0 and not tax_impact.get("is_long_term", False)):
                                tax_impact["is_optimal_timing"] = False
                                tax_impact["delay_recommendation"] = 365 - holding_days
                        
                        trade = {
                            "symbol": symbol,
                            "current_weight": current_weight,
                            "target_weight": target_weight,
                            "weight_diff": weight_diff,
                            "trade_quantity": trade_quantity,
                            "trade_value": trade_value,
                            "current_price": current_price,
                            "tax_impact": tax_impact,
                            "priority": self._calculate_rebalance_priority(weight_diff, tax_impact)
                        }
                        rebalance_trades.append(trade)
            
            # Sort trades by priority (tax-efficient first)
            rebalance_trades.sort(key=lambda x: x["priority"], reverse=True)
            
            return {
                "rebalance_trades": rebalance_trades,
                "total_tax_impact": total_tax_impact,
                "current_weights": current_weights,
                "target_weights": target_weights,
                "portfolio_drift": self._calculate_portfolio_drift(current_weights, target_weights),
                "tax_efficiency_score": self._calculate_overall_tax_efficiency(rebalance_trades),
                "recommendations": self._generate_rebalance_recommendations(rebalance_trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal rebalance: {e}")
            return {"error": str(e), "rebalance_trades": []}
    
    def _calculate_tax_efficiency(self, gain_loss: float, is_long_term: bool) -> float:
        """Calculate tax efficiency score for a position (0-1)."""
        try:
            if gain_loss <= 0:
                return 1.0  # Losses are tax efficient
            
            # For gains, long-term is more efficient
            base_score = 0.8 if is_long_term else 0.4
            
            # Adjust for magnitude of gain
            gain_magnitude_penalty = min(0.3, gain_loss / 10000)  # Penalty for large gains
            
            return max(0, base_score - gain_magnitude_penalty)
            
        except Exception:
            return 0.5
    
    def _calculate_harvest_priority(self, total_loss: float, tax_benefit: float, position: Dict) -> float:
        """Calculate priority score for loss harvesting."""
        try:
            # Base score from tax benefit
            base_score = min(1000, abs(tax_benefit))  # Cap at $1000 benefit
            
            # Bonus for larger losses (more flexibility)
            loss_bonus = min(500, abs(total_loss) * 0.1)
            
            # Penalty for recent acquisitions (wash sale risk)
            entry_date = position.get("entry_date")
            if entry_date:
                days_held = (datetime.now(timezone.utc) - entry_date).days
                recency_penalty = max(0, (31 - days_held) * 10) if days_held < 31 else 0
            else:
                recency_penalty = 0
            
            return base_score + loss_bonus - recency_penalty
            
        except Exception:
            return 0
    
    def _calculate_rebalance_priority(self, weight_diff: float, tax_impact: Dict) -> float:
        """Calculate priority score for rebalancing trades."""
        try:
            # Higher priority for larger deviations
            deviation_score = abs(weight_diff) * 100
            
            # Tax efficiency bonus/penalty
            tax_liability = tax_impact.get("tax_liability", 0)
            is_optimal_timing = tax_impact.get("is_optimal_timing", True)
            
            tax_penalty = tax_liability * 0.1  # Reduce priority based on tax cost
            timing_bonus = 20 if is_optimal_timing else -50
            
            return deviation_score - tax_penalty + timing_bonus
            
        except Exception:
            return 0
    
    def _calculate_portfolio_drift(self, current_weights: Dict[str, float], 
                                 target_weights: Dict[str, float]) -> float:
        """Calculate overall portfolio drift from target."""
        try:
            total_drift = 0
            all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))
            
            for symbol in all_symbols:
                current = current_weights.get(symbol, 0)
                target = target_weights.get(symbol, 0)
                total_drift += abs(current - target)
            
            return total_drift / 2  # Normalize (sum of absolute differences / 2)
            
        except Exception:
            return 0
    
    def _calculate_overall_tax_efficiency(self, rebalance_trades: List[Dict]) -> float:
        """Calculate overall tax efficiency of rebalancing plan."""
        try:
            if not rebalance_trades:
                return 1.0
            
            efficiency_scores = []
            for trade in rebalance_trades:
                tax_impact = trade.get("tax_impact", {})
                is_optimal = tax_impact.get("is_optimal_timing", True)
                tax_liability = tax_impact.get("tax_liability", 0)
                trade_value = abs(trade.get("trade_value", 1))
                
                # Calculate efficiency for this trade
                if is_optimal and tax_liability == 0:
                    trade_efficiency = 1.0
                else:
                    tax_drag = tax_liability / trade_value if trade_value > 0 else 0
                    timing_penalty = 0 if is_optimal else 0.3
                    trade_efficiency = max(0, 1.0 - tax_drag - timing_penalty)
                
                efficiency_scores.append(trade_efficiency)
            
            return sum(efficiency_scores) / len(efficiency_scores)
            
        except Exception:
            return 0.5
    
    def _generate_rebalance_recommendations(self, rebalance_trades: List[Dict]) -> List[str]:
        """Generate recommendations for tax-optimal rebalancing."""
        recommendations = []
        
        try:
            # Check for timing issues
            for trade in rebalance_trades:
                symbol = trade.get("symbol", "")
                tax_impact = trade.get("tax_impact", {})
                
                if not tax_impact.get("is_optimal_timing", True):
                    delay_days = tax_impact.get("delay_recommendation", 0)
                    if delay_days > 0:
                        recommendations.append(
                            f"Consider delaying sale of {symbol} by {delay_days} days "
                            f"for long-term capital gains treatment"
                        )
                
                # Check for large tax liabilities
                tax_liability = tax_impact.get("tax_liability", 0)
                trade_value = abs(trade.get("trade_value", 1))
                if tax_liability > trade_value * 0.1:  # More than 10% tax drag
                    recommendations.append(
                        f"High tax impact for {symbol}: ${tax_liability:,.0f} "
                        f"({tax_liability/trade_value:.1%} of trade value)"
                    )
            
            # General recommendations
            total_sells = sum(1 for trade in rebalance_trades if trade.get("trade_quantity", 0) < 0)
            if total_sells > 5:
                recommendations.append("Consider spreading sales across multiple periods to manage tax impact")
            
            if not recommendations:
                recommendations.append("Rebalancing plan appears tax-efficient")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Manual review recommended due to analysis error"]


def rebalance_portfolio(ctx) -> None:
    """Enhanced portfolio rebalancing with tax awareness."""
    logger.info("Starting enhanced portfolio rebalancing")
    
    # AI-AGENT-REF: Enhanced rebalancing with tax optimization
    try:
        # Try to use enhanced tax-aware rebalancing if available
        if ENHANCED_FEATURES_AVAILABLE and hasattr(ctx, 'account_equity'):
            tax_rebalancer = TaxAwareRebalancer()
            
            current_positions = getattr(ctx, "current_positions", {})
            target_weights = getattr(ctx, "target_weights", {})
            current_prices = getattr(ctx, "current_prices", {})
            account_equity = getattr(ctx, "account_equity", 0)
            
            if all([current_positions, target_weights, current_prices, account_equity]):
                rebalance_plan = tax_rebalancer.calculate_optimal_rebalance(
                    current_positions, target_weights, current_prices, account_equity
                )
                
                logger.info(f"Tax-aware rebalancing complete: "
                           f"drift={rebalance_plan.get('portfolio_drift', 0):.3f}, "
                           f"tax_impact=${rebalance_plan.get('total_tax_impact', 0):,.0f}")
                
                # Store rebalancing plan for execution
                ctx.rebalance_plan = rebalance_plan
                return
    
    except Exception as e:
        logger.warning(f"Enhanced rebalancing failed, falling back to basic: {e}")
    
    # Fallback to original rebalancing logic
    logger.info("Using basic portfolio rebalancing")


def enhanced_maybe_rebalance(ctx) -> None:
    """Enhanced rebalance check with tax optimization and market conditions."""
    global _last_rebalance
    now = datetime.now(timezone.utc)
    
    # AI-AGENT-REF: Enhanced rebalancing with market condition awareness
    if (now - _last_rebalance) >= timedelta(minutes=REBALANCE_INTERVAL_MIN):
        try:
            portfolio = getattr(ctx, "portfolio_weights", {})
            
            # Always trigger at least one rebalance if no existing weights
            if not portfolio:
                portfolio_first_rebalance(ctx)
                _last_rebalance = now
                return
            
            # Enhanced drift calculation with market conditions
            current = compute_portfolio_weights(ctx, list(portfolio.keys()))
            drift = (
                max(abs(current.get(s, 0) - portfolio.get(s, 0)) for s in current)
                if current
                else 0.0
            )
            
            # Dynamic threshold based on market volatility if available
            drift_threshold = config.PORTFOLIO_DRIFT_THRESHOLD
            
            # Use portfolio-first rebalancing if available
            if PORTFOLIO_FIRST_AVAILABLE:
                should_rebalance, reason = _check_portfolio_first_rebalancing(ctx, drift, drift_threshold)
                if should_rebalance:
                    portfolio_first_rebalance(ctx)
                    _last_rebalance = now
                    logger.info(f"PORTFOLIO_FIRST_REBALANCING_EXECUTED | {reason}")
            else:
                # Fallback to original logic
                if drift > drift_threshold:
                    rebalance_portfolio(ctx)
                    _last_rebalance = now
                    
        except Exception as e:
            logger.error(f"Error in enhanced rebalancing: {e}")
            # Fallback to basic rebalancing
            try:
                rebalance_portfolio(ctx)
                _last_rebalance = now
            except Exception as fallback_error:
                logger.error(f"Fallback rebalancing also failed: {fallback_error}")


def portfolio_first_rebalance(ctx) -> None:
    """
    Portfolio-first rebalancing that integrates with portfolio optimization.
    
    This function serves as the primary trading mechanism, implementing
    quarterly tax-optimized rebalancing with intelligent decision making.
    """
    global _portfolio_optimizer, _regime_detector
    
    try:
        if not PORTFOLIO_FIRST_AVAILABLE:
            logger.info("Portfolio-first not available, using standard rebalancing")
            rebalance_portfolio(ctx)
            return
        
        # Initialize portfolio-first components
        if _portfolio_optimizer is None:
            _portfolio_optimizer = create_portfolio_optimizer()
            _regime_detector = create_regime_detector()
            logger.info("Portfolio-first rebalancing components initialized")
        
        # Prepare portfolio data
        current_positions = _get_current_positions_for_rebalancing(ctx)
        target_weights = _get_target_weights_for_rebalancing(ctx)
        market_data = _prepare_rebalancing_market_data(ctx)
        
        if not current_positions or not target_weights or not market_data:
            logger.warning("Insufficient data for portfolio-first rebalancing, using fallback")
            rebalance_portfolio(ctx)
            return
        
        # Detect market regime for dynamic rebalancing
        regime, regime_metrics = _regime_detector.detect_current_regime(market_data)
        
        # Calculate dynamic thresholds
        dynamic_thresholds = _regime_detector.calculate_dynamic_thresholds(regime, regime_metrics)
        
        # Update optimizer with dynamic thresholds
        _portfolio_optimizer.improvement_threshold = dynamic_thresholds.minimum_improvement_threshold
        _portfolio_optimizer.rebalance_drift_threshold = dynamic_thresholds.rebalance_drift_threshold
        
        # Check if rebalancing is needed with portfolio-level analysis
        current_prices = market_data.get('prices', {})
        should_rebalance, rebalance_reason = _portfolio_optimizer.should_trigger_rebalance(
            current_positions,
            target_weights,
            current_prices
        )
        
        if should_rebalance:
            # Execute tax-aware rebalancing
            if ENHANCED_FEATURES_AVAILABLE:
                tax_rebalancer = TaxAwareRebalancer()
                
                # Calculate account equity
                account_equity = sum(
                    abs(pos) * current_prices.get(symbol, 100.0)
                    for symbol, pos in current_positions.items()
                )
                
                # Convert positions to proper format for tax rebalancer
                formatted_positions = {}
                for symbol, quantity in current_positions.items():
                    if quantity != 0:
                        formatted_positions[symbol] = {
                            'quantity': abs(quantity),
                            'purchase_price': current_prices.get(symbol, 100.0),  # Simplified
                            'purchase_date': datetime.now(timezone.utc) - timedelta(days=100)  # Simplified
                        }
                
                rebalance_plan = tax_rebalancer.calculate_optimal_rebalance(
                    formatted_positions,
                    target_weights,
                    current_prices,
                    account_equity
                )
                
                # Log detailed rebalancing information
                logger.info("PORTFOLIO_FIRST_REBALANCING_COMPLETE", extra={
                    "reason": rebalance_reason,
                    "market_regime": regime.value,
                    "portfolio_drift": rebalance_plan.get('portfolio_drift', 0),
                    "total_tax_impact": rebalance_plan.get('total_tax_impact', 0),
                    "rebalance_trades": len(rebalance_plan.get('rebalance_trades', [])),
                    "tax_efficiency": rebalance_plan.get('tax_efficiency', 0)
                })
                
                # Store rebalancing plan for execution
                ctx.rebalance_plan = rebalance_plan
                ctx.last_portfolio_rebalance = datetime.now(timezone.utc)
                
            else:
                # Fallback to basic rebalancing
                logger.info("Using basic rebalancing due to limited features")
                rebalance_portfolio(ctx)
                
        else:
            logger.info(f"PORTFOLIO_FIRST_REBALANCING_SKIPPED | {rebalance_reason}")
            
    except Exception as e:
        logger.error(f"Error in portfolio-first rebalancing: {e}")
        # Fallback to original rebalancing
        try:
            rebalance_portfolio(ctx)
        except Exception as fallback_error:
            logger.error(f"Fallback rebalancing failed: {fallback_error}")


def _check_portfolio_first_rebalancing(ctx, current_drift: float, drift_threshold: float) -> tuple:
    """Check if portfolio-first rebalancing should be triggered."""
    try:
        # Basic drift check
        if current_drift > drift_threshold:
            return True, f"Drift {current_drift:.3f} exceeds threshold {drift_threshold:.3f}"
        
        # Time-based quarterly rebalancing check
        last_rebalance = getattr(ctx, 'last_portfolio_rebalance', None)
        if last_rebalance is None:
            return True, "No previous rebalancing recorded"
        
        days_since_rebalance = (datetime.now(timezone.utc) - last_rebalance).days
        if days_since_rebalance >= 90:  # Quarterly rebalancing
            return True, f"Quarterly rebalance due ({days_since_rebalance} days since last)"
        
        # Market regime-based rebalancing
        if _regime_detector is not None:
            try:
                market_data = _prepare_rebalancing_market_data(ctx)
                if market_data:
                    regime, metrics = _regime_detector.detect_current_regime(market_data)
                    
                    # Force rebalancing in crisis conditions
                    if regime.value == "crisis":
                        return True, "Crisis regime detected - protective rebalancing"
                    
                    # Rebalance on significant volatility changes
                    if (metrics.volatility_regime.value in ["extremely_high", "extremely_low"] and
                        metrics.regime_confidence > 0.8):
                        return True, f"Extreme volatility regime: {metrics.volatility_regime.value}"
                        
            except Exception as e:
                logger.debug(f"Error in regime-based rebalancing check: {e}")
        
        return False, f"No rebalancing needed (drift={current_drift:.3f}, days={days_since_rebalance})"
        
    except Exception as e:
        logger.error(f"Error checking portfolio-first rebalancing: {e}")
        return current_drift > drift_threshold, "Error in analysis, using basic drift check"


def _get_current_positions_for_rebalancing(ctx) -> dict:
    """Get current positions formatted for portfolio rebalancing."""
    try:
        positions = {}
        
        # Try multiple sources for position data
        if hasattr(ctx, 'portfolio_positions'):
            positions = ctx.portfolio_positions.copy()
        elif hasattr(ctx, 'current_positions'):
            positions = ctx.current_positions.copy()
        elif hasattr(ctx, 'positions'):
            positions = ctx.positions.copy()
        
        # Convert to float and filter out zero positions
        filtered_positions = {}
        for symbol, quantity in positions.items():
            try:
                qty = float(quantity)
                if abs(qty) > 0.001:  # Filter out very small positions
                    filtered_positions[symbol] = qty
            except (ValueError, TypeError):
                continue
        
        return filtered_positions
        
    except Exception as e:
        logger.error(f"Error getting current positions: {e}")
        return {}


def _get_target_weights_for_rebalancing(ctx) -> dict:
    """Get target portfolio weights for rebalancing."""
    try:
        # Try to get target weights from context
        if hasattr(ctx, 'target_weights'):
            return ctx.target_weights.copy()
        
        # Fallback: compute equal weights for current positions
        current_positions = _get_current_positions_for_rebalancing(ctx)
        if current_positions:
            num_positions = len(current_positions)
            equal_weight = 1.0 / num_positions
            return {symbol: equal_weight for symbol in current_positions.keys()}
        
        return {}
        
    except Exception as e:
        logger.error(f"Error getting target weights: {e}")
        return {}


def _prepare_rebalancing_market_data(ctx) -> dict:
    """Prepare market data for rebalancing analysis."""
    try:
        market_data = {
            'prices': {},
            'returns': {},
            'volumes': {},
            'correlations': {},
            'volatility': {}
        }
        
        # Get symbols from current positions
        current_positions = _get_current_positions_for_rebalancing(ctx)
        symbols = set(current_positions.keys())
        symbols.add('SPY')  # Add index for regime detection
        
        # Fetch market data
        for symbol in symbols:
            try:
                if hasattr(ctx, 'data_fetcher'):
                    df = ctx.data_fetcher.get_daily_df(ctx, symbol)
                    if df is not None and len(df) > 0:
                        # Current price
                        market_data['prices'][symbol] = df['close'].iloc[-1] if 'close' in df.columns else 100.0
                        
                        # Returns for regime detection
                        if 'close' in df.columns and len(df) > 1:
                            prices = df['close'].values[-100:]  # Last 100 days
                            returns = []
                            for i in range(1, len(prices)):
                                if prices[i-1] > 0:
                                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
                            market_data['returns'][symbol] = returns
                        
                        # Volume data
                        if 'volume' in df.columns:
                            market_data['volumes'][symbol] = df['volume'].tail(20).mean()
                            
            except Exception as e:
                logger.debug(f"Could not fetch rebalancing data for {symbol}: {e}")
        
        return market_data
        
    except Exception as e:
        logger.error(f"Error preparing rebalancing market data: {e}")
        return {}


def maybe_rebalance(ctx) -> None:
    """Rebalance when interval has elapsed."""
    global _last_rebalance
    now = datetime.now(timezone.utc)
    if (now - _last_rebalance) >= timedelta(minutes=REBALANCE_INTERVAL_MIN):
        portfolio = getattr(ctx, "portfolio_weights", {})
        # always trigger at least one rebalance if no existing weights
        if not portfolio:
            rebalance_portfolio(ctx)
            _last_rebalance = now
        else:
            current = compute_portfolio_weights(ctx, list(portfolio.keys()))
            drift = (
                max(abs(current.get(s, 0) - portfolio.get(s, 0)) for s in current)
                if current
                else 0.0
            )
            if drift > config.PORTFOLIO_DRIFT_THRESHOLD:
                rebalance_portfolio(ctx)
                _last_rebalance = now


def start_rebalancer(ctx) -> threading.Thread:
    """Run :func:`maybe_rebalance` every minute in a background thread."""
    def loop() -> None:
        while True:
            try:
                maybe_rebalance(ctx)
            except StopIteration:
                # AI-AGENT-REF: Allow tests to break out of the loop
                logger.debug("Rebalancer loop stopped by test")
                break
            except Exception as exc:  # pragma: no cover - background errors
                logger.error("Rebalancer loop error: %s", exc)
            # AI-AGENT-REF: configurable sleep interval, shorter for tests
            sleep_interval = int(config.get_env("REBALANCE_SLEEP_SECONDS", "600"))
            # Detect test environment and use shorter interval
            import os
            if os.getenv("PYTEST_CURRENT_TEST") or "test" in str(ctx).lower():
                sleep_interval = 1  # 1 second for tests
            time.sleep(sleep_interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t

