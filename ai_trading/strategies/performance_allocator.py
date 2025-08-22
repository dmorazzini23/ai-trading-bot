"""
Rolling performance weights for dynamic strategy allocation.

Implements performance-based capital allocation with decay and bounds
to allocate more capital to better-performing strategies while maintaining
diversification and risk controls.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List

import numpy as np
import pandas as pd  # noqa: F401  # AI-AGENT-REF: retained for downstream usage

# Use centralized logger as per AGENTS.md
from ai_trading.logging import logger
from ai_trading.config.settings import get_settings  # AI-AGENT-REF: env-backed settings
from ai_trading.config.management import TradingConfig  # AI-AGENT-REF: config type


@dataclass
class AllocatorConfig:
    score_confidence_min: float = 0.6  # AI-AGENT-REF: default confidence gate


def _resolve_conf_threshold(cfg: TradingConfig | None) -> float:
    """Resolve confidence threshold in [0,1]."""
    # AI-AGENT-REF: priority - TradingConfig → Settings → default
    v = getattr(cfg, "score_confidence_min", None) if cfg else None
    try:
        if v is not None:
            v = float(v)
            if 0.0 <= v <= 1.0:
                return v
    except Exception:
        pass
    s = get_settings()
    for cand in (
        getattr(s, "score_confidence_min", None),
        getattr(s, "conf_threshold", None),
    ):
        try:
            if cand is not None:
                x = float(cand)
                if 0.0 <= x <= 1.0:
                    return x
        except Exception:
            continue
    return 0.60


def _compute_conf_multiplier(
    conf: float,
    th: float,
    max_boost: float,
    gamma: float,
) -> float:
    """Monotonic size multiplier in [1, max_boost]."""  # AI-AGENT-REF: size curve
    if not (0.0 <= conf <= 1.0):
        return 1.0
    th = min(max(th, 0.0), 0.999999)
    max_boost = max(1.0, float(max_boost))
    gamma = max(1e-6, float(gamma))
    if conf <= th:
        return 1.0
    span = max(1e-9, 1.0 - th)
    frac = (conf - th) / span
    mult = 1.0 + (max_boost - 1.0) * (frac ** gamma)
    return float(min(max(mult, 1.0), max_boost))


class PerformanceBasedAllocator:
    """
    Dynamic strategy allocator based on rolling performance metrics.
    
    Tracks strategy performance over a rolling window and adjusts capital
    allocation to favor better-performing strategies while maintaining
    diversification and risk bounds.
    """
    
    def __init__(self, config: AllocatorConfig | Dict | None = None):
        """Initialize performance-based allocator."""
        if config is None:
            self.config = {}
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = config.__dict__
        self._logged_conf_gate_once = False  # AI-AGENT-REF: confidence gate log guard
        
        # Performance tracking configuration
        self.window_days = self.config.get("performance_window_days", 20)  # 20 trading days
        self.min_trades_threshold = self.config.get("min_trades_threshold", 5)  # Minimum trades for evaluation
        self.decay_factor = self.config.get("decay_factor", 0.95)  # Exponential decay for older performance
        
        # Allocation bounds
        self.min_allocation = self.config.get("min_allocation_pct", 0.05)  # 5% minimum per strategy
        self.max_allocation = self.config.get("max_allocation_pct", 0.40)  # 40% maximum per strategy
        self.default_allocation = self.config.get("default_allocation_pct", 0.20)  # 20% default
        
        # Performance metrics weights
        self.sharpe_weight = self.config.get("sharpe_weight", 0.4)
        self.return_weight = self.config.get("return_weight", 0.3)
        self.hit_rate_weight = self.config.get("hit_rate_weight", 0.2)
        self.drawdown_weight = self.config.get("drawdown_weight", 0.1)
        
        # Track strategy performance history
        self.strategy_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.strategy_allocations: Dict[str, float] = {}
        self.last_update = datetime.now(UTC)

        logger.info("PerformanceBasedAllocator initialized with %d day window", self.window_days)

    def score_to_weight(self, score: float) -> float:
        """Map a confidence score to a weight with a minimum gate."""  # AI-AGENT-REF
        th = float(self.config.get("score_confidence_min", 0.6))
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = 0.0
        if s < th:
            if not self._logged_conf_gate_once:
                logger.info(
                    "ALLOC_CONFIDENCE_GATE",
                    extra={
                        "threshold": th,
                        "note": "candidates below this score are zero-weighted",
                    },
                )
                self._logged_conf_gate_once = True
            return 0.0
        return max(0.0, min(1.0, s))

    def allocate(self, strategies: Dict[str, List[Any]], config: TradingConfig) -> Dict[str, List[Any]]:
        """Filter low-confidence signals and bias weights by confidence."""  # AI-AGENT-REF
        th = _resolve_conf_threshold(config)
        s = get_settings()
        max_boost = float(getattr(s, "score_size_max_boost", 1.0) or 1.0)
        gamma = float(getattr(s, "score_size_gamma", 1.0) or 1.0)
        use_boost = max_boost > 1.0
        gated: Dict[str, List[Any]] = {}
        boost_stats: List[tuple[str, float, float]] = []
        for name, sigs in (strategies or {}).items():
            kept: List[Any] = []
            dropped = 0
            mults: List[float] = []
            for s_ in sigs or []:
                try:
                    c = float(getattr(s_, "confidence", 0.0))
                except Exception:
                    c = 0.0
                if c >= th:
                    if use_boost:
                        m = _compute_conf_multiplier(c, th, max_boost, gamma)
                        try:
                            base = float(getattr(s_, "weight", 1.0))
                        except Exception:
                            base = 1.0
                        s_.weight = base * m
                        mults.append(m)
                    kept.append(s_)
                else:
                    dropped += 1
            if dropped:
                logger.info(
                    "CONFIDENCE_DROP",
                    extra={"strategy": name, "threshold": th, "dropped": dropped, "kept": len(kept)},
                )
            if kept:
                gated[name] = kept
                if use_boost and mults:
                    boost_stats.append((name, sum(mults) / len(mults), max(mults)))
        if use_boost and boost_stats:
            for name, avg_mult, mx in boost_stats:
                logger.info(
                    "CONFIDENCE_BOOST",
                    extra={
                        "strategy": name,
                        "threshold": th,
                        "max_boost": max_boost,
                        "gamma": gamma,
                        "avg_mult": round(float(avg_mult), 4),
                        "max_mult": round(float(mx), 4),
                    },
                )
        return gated
    
    def record_trade_result(self, strategy_name: str, trade_result: Dict):
        """
        Record a trade result for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            trade_result: Dictionary containing trade details:
                - symbol: str
                - entry_price: float
                - exit_price: float
                - quantity: float
                - pnl: float
                - timestamp: datetime
                - success: bool
        """
        try:
            # Validate trade result
            required_fields = ['symbol', 'entry_price', 'exit_price', 'pnl', 'timestamp']
            for field in required_fields:
                if field not in trade_result:
                    logger.warning("Trade result missing required field %s for strategy %s", 
                                 field, strategy_name)
                    return
            
            # Add to strategy's trade history
            trade_record = {
                **trade_result,
                'recorded_at': datetime.now(UTC),
                'return_pct': trade_result['pnl'] / abs(trade_result['entry_price'] * trade_result.get('quantity', 1))
            }
            
            self.strategy_trades[strategy_name].append(trade_record)
            
            logger.debug("Recorded trade for strategy %s: PnL=%.2f, Return=%.4f", 
                        strategy_name, trade_result['pnl'], trade_record['return_pct'])
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to record trade result for strategy %s: %s", strategy_name, e,
                          extra={"component": "performance_allocator", "strategy": strategy_name, "error_type": "trade_record"})
        except Exception as e:
            logger.error("Unexpected error recording trade for strategy %s: %s", strategy_name, e,
                        extra={"component": "performance_allocator", "strategy": strategy_name, "error_type": "unexpected"})
    
    def calculate_strategy_allocations(self, strategies: List[str], 
                                     total_capital: float) -> Dict[str, float]:
        """
        Calculate optimal capital allocation across strategies based on performance.
        
        Args:
            strategies: List of strategy names
            total_capital: Total capital to allocate
            
        Returns:
            Dictionary mapping strategy name to allocated capital amount
        """
        try:
            if not strategies:
                logger.warning("No strategies provided for allocation")
                return {}
            
            # Calculate performance scores for each strategy
            performance_scores = {}
            for strategy in strategies:
                score = self._calculate_performance_score(strategy)
                performance_scores[strategy] = score
                
                logger.debug("Strategy %s performance score: %.4f", strategy, score)
            
            # Convert scores to allocation weights
            allocation_weights = self._scores_to_weights(performance_scores)
            
            # Apply bounds and constraints
            bounded_weights = self._apply_allocation_bounds(allocation_weights)
            
            # Convert weights to capital amounts
            allocations = {
                strategy: weight * total_capital 
                for strategy, weight in bounded_weights.items()
            }
            
            # Update internal tracking
            self.strategy_allocations = bounded_weights.copy()
            self.last_update = datetime.now(UTC)
            
            # Log allocation summary
            logger.info("Strategy allocations updated: %s", 
                       {s: f"{w:.1%}" for s, w in bounded_weights.items()})
            
            return allocations
            
        except Exception as e:
            logger.error("Strategy allocation calculation failed: %s", e,
                        extra={"component": "performance_allocator", "error_type": "allocation"})
            
            # Return equal allocation as fallback
            equal_weight = 1.0 / len(strategies)
            return {strategy: equal_weight * total_capital for strategy in strategies}
    
    def _calculate_performance_score(self, strategy_name: str) -> float:
        """Calculate composite performance score for a strategy."""
        try:
            trades = list(self.strategy_trades[strategy_name])
            
            if len(trades) < self.min_trades_threshold:
                # Insufficient data - return default score
                logger.debug("Insufficient trades for strategy %s (%d < %d) - using default score", 
                           strategy_name, len(trades), self.min_trades_threshold)
                return 0.5  # Neutral score
            
            # Filter to recent trades within window
            cutoff_date = datetime.now(UTC) - timedelta(days=self.window_days)
            recent_trades = [t for t in trades if t['timestamp'] >= cutoff_date]
            
            if len(recent_trades) < 3:  # Need minimum recent activity
                return 0.3  # Below average for inactive strategies
            
            # Calculate performance metrics
            returns = np.array([t['return_pct'] for t in recent_trades])
            
            # Apply exponential decay to older trades
            weights = np.array([
                self.decay_factor ** i for i in range(len(recent_trades) - 1, -1, -1)
            ])
            weights = weights / weights.sum()  # Normalize
            
            # Weighted performance metrics
            avg_return = np.average(returns, weights=weights)
            volatility = np.sqrt(np.average((returns - avg_return) ** 2, weights=weights))
            
            # Sharpe ratio (annualized)
            sharpe_ratio = (avg_return / max(volatility, 1e-6)) * np.sqrt(252)
            
            # Hit rate
            winning_trades = np.sum(returns > 0)
            hit_rate = winning_trades / len(returns)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # Composite score calculation
            sharpe_component = self._normalize_metric(sharpe_ratio, -2, 3) * self.sharpe_weight
            return_component = self._normalize_metric(avg_return * 252, -0.5, 0.5) * self.return_weight  # Annualized
            hit_rate_component = self._normalize_metric(hit_rate, 0.3, 0.7) * self.hit_rate_weight
            drawdown_component = self._normalize_metric(-max_drawdown, -0.3, 0) * self.drawdown_weight
            
            composite_score = (sharpe_component + return_component + 
                             hit_rate_component + drawdown_component)
            
            # Bound the score between 0 and 1
            composite_score = max(0.0, min(1.0, composite_score))
            
            logger.debug("Strategy %s metrics: Sharpe=%.2f, Return=%.3f, HitRate=%.1f%%, DD=%.2f%%, Score=%.3f",
                        strategy_name, sharpe_ratio, avg_return * 252, hit_rate * 100, 
                        max_drawdown * 100, composite_score)
            
            return composite_score
            
        except (ValueError, TypeError) as e:
            logger.warning("Performance calculation failed for strategy %s: %s", strategy_name, e)
            return 0.3  # Below average for calculation errors
        except Exception as e:
            logger.error("Unexpected error calculating performance for strategy %s: %s", strategy_name, e)
            return 0.3
    
    def _normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a metric to 0-1 range."""
        if max_val <= min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def _scores_to_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert performance scores to allocation weights."""
        if not scores:
            return {}
        
        # Apply softmax transformation to convert scores to weights
        score_values = np.array(list(scores.values()))
        
        # Add small bias to prevent zero allocations
        score_values = score_values + 0.1
        
        # Softmax with temperature parameter for smoothing
        temperature = self.config.get("allocation_temperature", 2.0)
        exp_scores = np.exp(score_values / temperature)
        weights = exp_scores / exp_scores.sum()
        
        return dict(zip(scores.keys(), weights))
    
    def _apply_allocation_bounds(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum allocation bounds."""
        if not weights:
            return {}
        
        bounded_weights = weights.copy()
        
        # Apply minimum allocation
        for strategy in bounded_weights:
            if bounded_weights[strategy] < self.min_allocation:
                bounded_weights[strategy] = self.min_allocation
        
        # Renormalize after applying minimums
        total_weight = sum(bounded_weights.values())
        if total_weight > 1.0:
            for strategy in bounded_weights:
                bounded_weights[strategy] /= total_weight
        
        # Apply maximum allocation
        excess_total = 0.0
        capped_strategies = []
        
        for strategy in bounded_weights:
            if bounded_weights[strategy] > self.max_allocation:
                excess = bounded_weights[strategy] - self.max_allocation
                bounded_weights[strategy] = self.max_allocation
                excess_total += excess
                capped_strategies.append(strategy)
        
        # Redistribute excess to non-capped strategies
        if excess_total > 0:
            non_capped = [s for s in bounded_weights if s not in capped_strategies]
            if non_capped:
                excess_per_strategy = excess_total / len(non_capped)
                for strategy in non_capped:
                    bounded_weights[strategy] += excess_per_strategy
                    # Ensure we don't exceed max for redistribution
                    bounded_weights[strategy] = min(bounded_weights[strategy], self.max_allocation)
        
        # Final normalization
        total_weight = sum(bounded_weights.values())
        if total_weight > 0:
            for strategy in bounded_weights:
                bounded_weights[strategy] /= total_weight
        
        return bounded_weights
    
    def get_strategy_performance_report(self, strategy_name: str) -> Dict:
        """Generate detailed performance report for a strategy."""
        try:
            trades = list(self.strategy_trades[strategy_name])
            
            if not trades:
                return {
                    "strategy": strategy_name,
                    "error": "No trade history available"
                }
            
            # Calculate metrics over different time windows
            windows = [5, 10, 20, 60]  # days
            report = {
                "strategy": strategy_name,
                "total_trades": len(trades),
                "performance_score": self._calculate_performance_score(strategy_name),
                "current_allocation": self.strategy_allocations.get(strategy_name, 0.0),
                "windows": {}
            }
            
            for window_days in windows:
                cutoff_date = datetime.now(UTC) - timedelta(days=window_days)
                window_trades = [t for t in trades if t['timestamp'] >= cutoff_date]
                
                if window_trades:
                    returns = [t['return_pct'] for t in window_trades]
                    pnls = [t['pnl'] for t in window_trades]
                    
                    report["windows"][f"{window_days}d"] = {
                        "trades": len(window_trades),
                        "total_pnl": sum(pnls),
                        "avg_return": np.mean(returns),
                        "hit_rate": sum(1 for r in returns if r > 0) / len(returns),
                        "sharpe": (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
                    }
            
            return report
            
        except Exception as e:
            logger.error("Performance report generation failed for strategy %s: %s", strategy_name, e)
            return {
                "strategy": strategy_name,
                "error": f"Report generation failed: {e}"
            }
    
    def should_rebalance_allocations(self) -> bool:
        """Determine if allocations should be rebalanced based on recent performance."""
        try:
            # Rebalance daily or if significant performance divergence
            hours_since_update = (datetime.now(UTC) - self.last_update).total_seconds() / 3600
            
            if hours_since_update >= 24:  # Daily rebalancing
                return True
            
            # Check for significant performance divergence
            if len(self.strategy_allocations) < 2:
                return False
            
            current_scores = {}
            for strategy in self.strategy_allocations:
                current_scores[strategy] = self._calculate_performance_score(strategy)
            
            # Check if performance ordering has changed significantly
            old_ranking = sorted(self.strategy_allocations.items(), key=lambda x: x[1], reverse=True)
            new_ranking = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Simple ranking change detection
            rank_changes = sum(1 for i, (old_s, _) in enumerate(old_ranking) 
                             if i < len(new_ranking) and new_ranking[i][0] != old_s)
            
            significant_change_threshold = len(self.strategy_allocations) // 2
            
            return rank_changes >= significant_change_threshold
            
        except Exception as e:
            logger.warning("Rebalance decision failed: %s", e)
            return False  # Conservative - don't rebalance on errors