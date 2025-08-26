"""
Rolling performance weights for dynamic strategy allocation.

Implements performance-based capital allocation with decay and bounds
to allocate more capital to better-performing strategies while maintaining
diversification and risk controls.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
import numpy as np
from ai_trading.config.management import TradingConfig
from ai_trading.config.settings import get_settings
from ai_trading.logging import logger

@dataclass
class AllocatorConfig:
    score_confidence_min: float = 0.6

def _resolve_conf_threshold(cfg: TradingConfig | None) -> float:
    """Resolve confidence threshold in [0,1]."""
    v = getattr(cfg, 'score_confidence_min', None) if cfg else None
    try:
        if v is not None:
            v = float(v)
            if 0.0 <= v <= 1.0:
                return v
    except (ValueError, TypeError):
        pass
    s = get_settings()
    for cand in (getattr(s, 'score_confidence_min', None), getattr(s, 'conf_threshold', None)):
        try:
            if cand is not None:
                x = float(cand)
                if 0.0 <= x <= 1.0:
                    return x
        except (ValueError, TypeError):
            continue
    return 0.6

def _compute_conf_multiplier(conf: float, th: float, max_boost: float, gamma: float) -> float:
    """Monotonic size multiplier in [1, max_boost]."""
    if not 0.0 <= conf <= 1.0:
        return 1.0
    th = min(max(th, 0.0), 0.999999)
    max_boost = max(1.0, float(max_boost))
    gamma = max(1e-06, float(gamma))
    if conf <= th:
        return 1.0
    span = max(1e-09, 1.0 - th)
    frac = (conf - th) / span
    mult = 1.0 + (max_boost - 1.0) * frac ** gamma
    return float(min(max(mult, 1.0), max_boost))

class PerformanceBasedAllocator:
    """
    Dynamic strategy allocator based on rolling performance metrics.
    
    Tracks strategy performance over a rolling window and adjusts capital
    allocation to favor better-performing strategies while maintaining
    diversification and risk bounds.
    """

    def __init__(self, config: AllocatorConfig | dict | None=None):
        """Initialize performance-based allocator."""
        if config is None:
            self.config = {}
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = config.__dict__
        self._logged_conf_gate_once = False
        self.window_days = self.config.get('performance_window_days', 20)
        self.min_trades_threshold = self.config.get('min_trades_threshold', 5)
        self.decay_factor = self.config.get('decay_factor', 0.95)
        self.min_allocation = self.config.get('min_allocation_pct', 0.05)
        self.max_allocation = self.config.get('max_allocation_pct', 0.4)
        self.default_allocation = self.config.get('default_allocation_pct', 0.2)
        self.sharpe_weight = self.config.get('sharpe_weight', 0.4)
        self.return_weight = self.config.get('return_weight', 0.3)
        self.hit_rate_weight = self.config.get('hit_rate_weight', 0.2)
        self.drawdown_weight = self.config.get('drawdown_weight', 0.1)
        self.strategy_trades: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.strategy_allocations: dict[str, float] = {}
        self.last_update = datetime.now(UTC)
        logger.info('PerformanceBasedAllocator initialized with %d day window', self.window_days)

    def score_to_weight(self, score: float) -> float:
        """Map a confidence score to a weight with a minimum gate."""
        th = float(self.config.get('score_confidence_min', 0.6))
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = 0.0
        if s < th:
            if not self._logged_conf_gate_once:
                logger.info('ALLOC_CONFIDENCE_GATE', extra={'threshold': th, 'note': 'candidates below this score are zero-weighted'})
                self._logged_conf_gate_once = True
            return 0.0
        return max(0.0, min(1.0, s))

    def allocate(self, strategies: dict[str, list[Any]], config: TradingConfig) -> dict[str, list[Any]]:
        """Filter low-confidence signals and bias weights by confidence."""
        th = _resolve_conf_threshold(config)
        s = get_settings()
        max_boost = float(getattr(s, 'score_size_max_boost', 1.0) or 1.0)
        gamma = float(getattr(s, 'score_size_gamma', 1.0) or 1.0)
        use_boost = max_boost > 1.0
        gated: dict[str, list[Any]] = {}
        boost_stats: list[tuple[str, float, float]] = []
        for name, sigs in (strategies or {}).items():
            kept: list[Any] = []
            dropped = 0
            mults: list[float] = []
            for s_ in sigs or []:
                try:
                    c = float(getattr(s_, 'confidence', 0.0))
                except (ValueError, TypeError):
                    c = 0.0
                if c >= th:
                    if use_boost:
                        m = _compute_conf_multiplier(c, th, max_boost, gamma)
                        try:
                            base = float(getattr(s_, 'weight', 1.0))
                        except (ValueError, TypeError):
                            base = 1.0
                        s_.weight = base * m
                        mults.append(m)
                    kept.append(s_)
                else:
                    dropped += 1
            if dropped:
                logger.info('CONFIDENCE_DROP', extra={'strategy': name, 'threshold': th, 'dropped': dropped, 'kept': len(kept)})
            if kept:
                gated[name] = kept
                if use_boost and mults:
                    boost_stats.append((name, sum(mults) / len(mults), max(mults)))
        if use_boost and boost_stats:
            for name, avg_mult, mx in boost_stats:
                logger.info('CONFIDENCE_BOOST', extra={'strategy': name, 'threshold': th, 'max_boost': max_boost, 'gamma': gamma, 'avg_mult': round(float(avg_mult), 4), 'max_mult': round(float(mx), 4)})
        return gated

    def record_trade_result(self, strategy_name: str, trade_result: dict):
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
            required_fields = ['symbol', 'entry_price', 'exit_price', 'pnl', 'timestamp']
            for field in required_fields:
                if field not in trade_result:
                    logger.warning('Trade result missing required field %s for strategy %s', field, strategy_name)
                    return
            trade_record = {**trade_result, 'recorded_at': datetime.now(UTC), 'return_pct': trade_result['pnl'] / abs(trade_result['entry_price'] * trade_result.get('quantity', 1))}
            self.strategy_trades[strategy_name].append(trade_record)
            logger.debug('Recorded trade for strategy %s: PnL=%.2f, Return=%.4f', strategy_name, trade_result['pnl'], trade_record['return_pct'])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning('Failed to record trade result for strategy %s: %s', strategy_name, e, extra={'component': 'performance_allocator', 'strategy': strategy_name, 'error_type': 'trade_record'})
        except (ValueError, TypeError) as e:
            logger.error('Unexpected error recording trade for strategy %s: %s', strategy_name, e, extra={'component': 'performance_allocator', 'strategy': strategy_name, 'error_type': 'unexpected'})

    def calculate_strategy_allocations(self, strategies: list[str], total_capital: float) -> dict[str, float]:
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
                logger.warning('No strategies provided for allocation')
                return {}
            performance_scores = {}
            for strategy in strategies:
                score = self._calculate_performance_score(strategy)
                performance_scores[strategy] = score
                logger.debug('Strategy %s performance score: %.4f', strategy, score)
            allocation_weights = self._scores_to_weights(performance_scores)
            bounded_weights = self._apply_allocation_bounds(allocation_weights)
            allocations = {strategy: weight * total_capital for strategy, weight in bounded_weights.items()}
            self.strategy_allocations = bounded_weights.copy()
            self.last_update = datetime.now(UTC)
            logger.info('Strategy allocations updated: %s', {s: f'{w:.1%}' for s, w in bounded_weights.items()})
            return allocations
        except (ValueError, TypeError) as e:
            logger.error('Strategy allocation calculation failed: %s', e, extra={'component': 'performance_allocator', 'error_type': 'allocation'})
            equal_weight = 1.0 / len(strategies)
            return {strategy: equal_weight * total_capital for strategy in strategies}

    def _calculate_performance_score(self, strategy_name: str) -> float:
        """Calculate composite performance score for a strategy."""
        try:
            trades = list(self.strategy_trades[strategy_name])
            if len(trades) < self.min_trades_threshold:
                logger.debug('Insufficient trades for strategy %s (%d < %d) - using default score', strategy_name, len(trades), self.min_trades_threshold)
                return 0.5
            cutoff_date = datetime.now(UTC) - timedelta(days=self.window_days)
            recent_trades = [t for t in trades if t['timestamp'] >= cutoff_date]
            if len(recent_trades) < 3:
                return 0.3
            returns = np.array([t['return_pct'] for t in recent_trades])
            weights = np.array([self.decay_factor ** i for i in range(len(recent_trades) - 1, -1, -1)])
            weights = weights / weights.sum()
            avg_return = np.average(returns, weights=weights)
            volatility = np.sqrt(np.average((returns - avg_return) ** 2, weights=weights))
            sharpe_ratio = avg_return / max(volatility, 1e-06) * np.sqrt(252)
            winning_trades = np.sum(returns > 0)
            hit_rate = winning_trades / len(returns)
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            sharpe_component = self._normalize_metric(sharpe_ratio, -2, 3) * self.sharpe_weight
            return_component = self._normalize_metric(avg_return * 252, -0.5, 0.5) * self.return_weight
            hit_rate_component = self._normalize_metric(hit_rate, 0.3, 0.7) * self.hit_rate_weight
            drawdown_component = self._normalize_metric(-max_drawdown, -0.3, 0) * self.drawdown_weight
            composite_score = sharpe_component + return_component + hit_rate_component + drawdown_component
            composite_score = max(0.0, min(1.0, composite_score))
            logger.debug('Strategy %s metrics: Sharpe=%.2f, Return=%.3f, HitRate=%.1f%%, DD=%.2f%%, Score=%.3f', strategy_name, sharpe_ratio, avg_return * 252, hit_rate * 100, max_drawdown * 100, composite_score)
            return composite_score
        except (ValueError, TypeError) as e:
            logger.warning('Performance calculation failed for strategy %s: %s', strategy_name, e)
            return 0.3
        except (ValueError, TypeError) as e:
            logger.error('Unexpected error calculating performance for strategy %s: %s', strategy_name, e)
            return 0.3

    def _normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a metric to 0-1 range."""
        if max_val <= min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def _scores_to_weights(self, scores: dict[str, float]) -> dict[str, float]:
        """Convert performance scores to allocation weights."""
        if not scores:
            return {}
        score_values = np.array(list(scores.values()))
        score_values = score_values + 0.1
        temperature = self.config.get('allocation_temperature', 2.0)
        exp_scores = np.exp(score_values / temperature)
        weights = exp_scores / exp_scores.sum()
        return dict(zip(scores.keys(), weights))

    def _apply_allocation_bounds(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply minimum and maximum allocation bounds."""
        if not weights:
            return {}
        bounded_weights = weights.copy()
        for strategy in bounded_weights:
            bounded_weights[strategy] = max(bounded_weights[strategy], self.min_allocation)
        total_weight = sum(bounded_weights.values())
        if total_weight > 1.0:
            for strategy in bounded_weights:
                bounded_weights[strategy] /= total_weight
        excess_total = 0.0
        capped_strategies = []
        for strategy in bounded_weights:
            if bounded_weights[strategy] > self.max_allocation:
                excess = bounded_weights[strategy] - self.max_allocation
                bounded_weights[strategy] = self.max_allocation
                excess_total += excess
                capped_strategies.append(strategy)
        if excess_total > 0:
            non_capped = [s for s in bounded_weights if s not in capped_strategies]
            if non_capped:
                excess_per_strategy = excess_total / len(non_capped)
                for strategy in non_capped:
                    bounded_weights[strategy] += excess_per_strategy
                    bounded_weights[strategy] = min(bounded_weights[strategy], self.max_allocation)
        total_weight = sum(bounded_weights.values())
        if total_weight > 0:
            for strategy in bounded_weights:
                bounded_weights[strategy] /= total_weight
        return bounded_weights

    def get_strategy_performance_report(self, strategy_name: str) -> dict:
        """Generate detailed performance report for a strategy."""
        try:
            trades = list(self.strategy_trades[strategy_name])
            if not trades:
                return {'strategy': strategy_name, 'error': 'No trade history available'}
            windows = [5, 10, 20, 60]
            report = {'strategy': strategy_name, 'total_trades': len(trades), 'performance_score': self._calculate_performance_score(strategy_name), 'current_allocation': self.strategy_allocations.get(strategy_name, 0.0), 'windows': {}}
            for window_days in windows:
                cutoff_date = datetime.now(UTC) - timedelta(days=window_days)
                window_trades = [t for t in trades if t['timestamp'] >= cutoff_date]
                if window_trades:
                    returns = [t['return_pct'] for t in window_trades]
                    pnls = [t['pnl'] for t in window_trades]
                    report['windows'][f'{window_days}d'] = {'trades': len(window_trades), 'total_pnl': sum(pnls), 'avg_return': np.mean(returns), 'hit_rate': sum((1 for r in returns if r > 0)) / len(returns), 'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0}
            return report
        except (ValueError, TypeError) as e:
            logger.error('Performance report generation failed for strategy %s: %s', strategy_name, e)
            return {'strategy': strategy_name, 'error': f'Report generation failed: {e}'}

    def should_rebalance_allocations(self) -> bool:
        """Determine if allocations should be rebalanced based on recent performance."""
        try:
            hours_since_update = (datetime.now(UTC) - self.last_update).total_seconds() / 3600
            if hours_since_update >= 24:
                return True
            if len(self.strategy_allocations) < 2:
                return False
            current_scores = {}
            for strategy in self.strategy_allocations:
                current_scores[strategy] = self._calculate_performance_score(strategy)
            old_ranking = sorted(self.strategy_allocations.items(), key=lambda x: x[1], reverse=True)
            new_ranking = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            rank_changes = sum((1 for i, (old_s, _) in enumerate(old_ranking) if i < len(new_ranking) and new_ranking[i][0] != old_s))
            significant_change_threshold = len(self.strategy_allocations) // 2
            return rank_changes >= significant_change_threshold
        except (ValueError, TypeError) as e:
            logger.warning('Rebalance decision failed: %s', e)
            return False