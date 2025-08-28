"""
Advanced position sizing and portfolio allocation strategies.

Provides volatility targeting, risk parity, correlation-based sizing,
and other institutional-grade position sizing methodologies.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
import numpy as np
from json import JSONDecodeError
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.logging import logger
from ai_trading.exc import RequestException

COMMON_EXC = (TypeError, ValueError, KeyError, JSONDecodeError, RequestException, TimeoutError, ImportError)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

def _import_clustering():
    from ai_trading.config import get_settings
    S = get_settings()
    if not S.ENABLE_PORTFOLIO_FEATURES:
        return (None, None, None, False)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

class VolatilityTargetingSizer:
    """
    Volatility targeting position sizer.

    Sizes positions to achieve a target portfolio volatility,
    adjusting for asset volatilities and correlations.
    """

    def __init__(self, target_vol: float=0.15, lookback_days: int=60, min_weight: float=0.01, max_weight: float=0.25, rebalance_threshold: float=0.05):
        """
        Initialize volatility targeting sizer.

        Args:
            target_vol: Target portfolio volatility (annual)
            lookback_days: Lookback period for volatility estimation
            min_weight: Minimum position weight
            max_weight: Maximum position weight
            rebalance_threshold: Threshold for rebalancing
        """
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rebalance_threshold = rebalance_threshold
        self.price_history = {}
        self.volatility_estimates = {}
        self.correlation_matrix = None
        logger.info(f'VolatilityTargetingSizer initialized with {target_vol:.1%} target volatility')

    def calculate_position_sizes(self, signals: dict[str, float], current_prices: dict[str, float], portfolio_value: float, price_history: dict[str, pd.Series] | None=None) -> dict[str, dict[str, float]]:
        """
        Calculate position sizes for given signals.

        Args:
            signals: Dict of {symbol: signal_strength}
            current_prices: Dict of {symbol: current_price}
            portfolio_value: Current portfolio value
            price_history: Optional price history for volatility estimation

        Returns:
            Dict with position sizing details for each symbol
        """
        try:
            if not signals:
                return {}
            if price_history:
                self.price_history.update(price_history)
            volatilities = self._estimate_volatilities(list(signals.keys()))
            self.correlation_matrix = self._estimate_correlations(list(signals.keys()))
            raw_weights = self._calculate_inverse_vol_weights(signals, volatilities)
            adjusted_weights = self._apply_position_limits(raw_weights)
            scaled_weights = self._scale_to_target_volatility(adjusted_weights, volatilities, self.correlation_matrix)
            position_details = {}
            for symbol, weight in scaled_weights.items():
                if symbol in current_prices and current_prices[symbol] > 0:
                    dollar_allocation = portfolio_value * weight
                    shares = int(dollar_allocation / current_prices[symbol])
                    position_details[symbol] = {'weight': weight, 'dollar_allocation': dollar_allocation, 'shares': shares, 'signal_strength': signals[symbol], 'estimated_vol': volatilities.get(symbol, 0.0), 'price': current_prices[symbol]}
            logger.debug(f'Calculated position sizes for {len(position_details)} symbols')
            return position_details
        except COMMON_EXC as e:
            logger.error(f'Error calculating position sizes: {e}')
            return {}

    def _estimate_volatilities(self, symbols: list[str]) -> dict[str, float]:
        """Estimate asset volatilities from price history."""
        try:
            volatilities = {}
            for symbol in symbols:
                if symbol in self.price_history:
                    prices = self.price_history[symbol]
                    recent_prices = prices.tail(self.lookback_days)
                    if len(recent_prices) > 5:
                        returns = recent_prices.pct_change().dropna()
                        vol = returns.std() * np.sqrt(252)
                        volatilities[symbol] = max(0.05, vol)
                    else:
                        volatilities[symbol] = 0.2
                else:
                    volatilities[symbol] = self.volatility_estimates.get(symbol, 0.2)
            self.volatility_estimates.update(volatilities)
            return volatilities
        except COMMON_EXC as e:
            logger.error(f'Error estimating volatilities: {e}')
            return dict.fromkeys(symbols, 0.2)

    def _estimate_correlations(self, symbols: list[str]) -> np.ndarray:
        """Estimate correlation matrix from price history."""
        pd = load_pandas()
        try:
            if len(symbols) < 2:
                return np.array([[1.0]])
            return_series = {}
            for symbol in symbols:
                if symbol in self.price_history:
                    prices = self.price_history[symbol].tail(self.lookback_days)
                    if len(prices) > 5:
                        returns = prices.pct_change().dropna()
                        return_series[symbol] = returns
            if len(return_series) < 2:
                n = len(symbols)
                corr_matrix = np.eye(n) * 0.8 + np.ones((n, n)) * 0.2
                return corr_matrix
            df = pd.DataFrame(return_series)
            df = df.dropna()
            if len(df) > 10:
                corr_matrix = df.corr().values
                full_corr = np.eye(len(symbols))
                for i, sym1 in enumerate(symbols):
                    for j, sym2 in enumerate(symbols):
                        if sym1 in df.columns and sym2 in df.columns:
                            idx1 = df.columns.get_loc(sym1)
                            idx2 = df.columns.get_loc(sym2)
                            full_corr[i, j] = corr_matrix[idx1, idx2]
                        elif i != j:
                            full_corr[i, j] = 0.3
                return full_corr
            else:
                n = len(symbols)
                return np.eye(n) * 0.7 + np.ones((n, n)) * 0.3
        except COMMON_EXC as e:
            logger.error(f'Error estimating correlations: {e}')
            n = len(symbols)
            return np.eye(n) * 0.7 + np.ones((n, n)) * 0.3

    def _calculate_inverse_vol_weights(self, signals: dict[str, float], volatilities: dict[str, float]) -> dict[str, float]:
        """Calculate weights inversely proportional to volatility."""
        try:
            inv_vol_weights = {}
            for symbol, signal in signals.items():
                vol = volatilities.get(symbol, 0.2)
                raw_weight = abs(signal) / vol
                inv_vol_weights[symbol] = raw_weight
            total_weight = sum(inv_vol_weights.values())
            if total_weight > 0:
                for symbol in inv_vol_weights:
                    inv_vol_weights[symbol] /= total_weight
            return inv_vol_weights
        except COMMON_EXC as e:
            logger.error(f'Error calculating inverse vol weights: {e}')
            n = len(signals)
            return dict.fromkeys(signals, 1.0 / n)

    def _apply_position_limits(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply minimum and maximum position limits."""
        try:
            adjusted_weights = {}
            for symbol, weight in weights.items():
                if weight < self.min_weight:
                    adjusted_weights[symbol] = 0.0
                else:
                    adjusted_weights[symbol] = min(weight, self.max_weight)
            adjusted_weights = {k: v for k, v in adjusted_weights.items() if v > 0}
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                for symbol in adjusted_weights:
                    adjusted_weights[symbol] /= total_weight
            return adjusted_weights
        except COMMON_EXC as e:
            logger.error(f'Error applying position limits: {e}')
            return weights

    def _scale_to_target_volatility(self, weights: dict[str, float], volatilities: dict[str, float], correlation_matrix: np.ndarray) -> dict[str, float]:
        """Scale weights to achieve target portfolio volatility."""
        try:
            if not weights:
                return weights
            symbols = list(weights.keys())
            weight_vector = np.array([weights[sym] for sym in symbols])
            vol_vector = np.array([volatilities[sym] for sym in symbols])
            vol_matrix = np.outer(vol_vector, vol_vector) * correlation_matrix
            portfolio_vol = np.sqrt(np.dot(weight_vector, np.dot(vol_matrix, weight_vector)))
            if portfolio_vol > 0:
                scale_factor = self.target_vol / portfolio_vol
                scale_factor = max(0.1, min(5.0, scale_factor))
                scaled_weights = {}
                for symbol in symbols:
                    scaled_weights[symbol] = weights[symbol] * scale_factor
                scaled_weights = self._apply_position_limits(scaled_weights)
                return scaled_weights
            else:
                return weights
        except COMMON_EXC as e:
            logger.error(f'Error scaling to target volatility: {e}')
            return weights

class RiskParitySizer:
    """
    Risk parity position sizer.

    Allocates capital such that each position contributes equally
    to total portfolio risk.
    """

    def __init__(self, lookback_days: int=60, max_iterations: int=100, tolerance: float=1e-06, max_weight: float=0.4):
        """
        Initialize risk parity sizer.

        Args:
            lookback_days: Lookback period for covariance estimation
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance
            max_weight: Maximum weight per position
        """
        self.lookback_days = lookback_days
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_weight = max_weight
        logger.info('RiskParitySizer initialized')

    def calculate_risk_parity_weights(self, signals: dict[str, float], price_history: dict[str, pd.Series]) -> dict[str, float]:
        """
        Calculate risk parity weights.

        Args:
            signals: Signal strengths for each asset
            price_history: Historical prices for covariance estimation

        Returns:
            Risk parity weights
        """
        try:
            symbols = list(signals.keys())
            if len(symbols) < 2:
                return {symbols[0]: 1.0} if symbols else {}
            cov_matrix = self._calculate_covariance_matrix(symbols, price_history)
            if cov_matrix is None:
                n = len(symbols)
                return dict.fromkeys(symbols, 1.0 / n)
            weights = self._optimize_risk_parity(cov_matrix, symbols)
            for symbol in symbols:
                if signals[symbol] < 0:
                    weights[symbol] *= -1
            return weights
        except COMMON_EXC as e:
            logger.error(f'Error calculating risk parity weights: {e}')
            n = len(signals)
            return dict.fromkeys(signals, 1.0 / n)

    def _calculate_covariance_matrix(self, symbols: list[str], price_history: dict[str, pd.Series]) -> np.ndarray | None:
        """Calculate covariance matrix from price history."""
        pd = load_pandas()
        try:
            return_series = {}
            for symbol in symbols:
                if symbol in price_history:
                    prices = price_history[symbol].tail(self.lookback_days)
                    if len(prices) > 10:
                        returns = prices.pct_change().dropna()
                        return_series[symbol] = returns
            if len(return_series) < 2:
                return None
            df = pd.DataFrame(return_series)
            df = df.dropna()
            if len(df) < 10:
                return None
            cov_matrix = df.cov().values * 252
            return cov_matrix
        except COMMON_EXC as e:
            logger.error(f'Error calculating covariance matrix: {e}')
            return None

    def _optimize_risk_parity(self, cov_matrix: np.ndarray, symbols: list[str]) -> dict[str, float]:
        """Optimize for risk parity using iterative method."""
        try:
            n = len(symbols)
            weights = np.ones(n) / n
            for _iteration in range(self.max_iterations):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contributions = weights * marginal_risk
                target_risk = portfolio_vol / n
                new_weights = weights * (target_risk / risk_contributions)
                new_weights = np.minimum(new_weights, self.max_weight)
                new_weights = new_weights / np.sum(new_weights)
                if np.max(np.abs(new_weights - weights)) < self.tolerance:
                    weights = new_weights
                    break
                weights = new_weights
            weight_dict = {}
            for i, symbol in enumerate(symbols):
                weight_dict[symbol] = weights[i]
            return weight_dict
        except COMMON_EXC as e:
            logger.error(f'Error optimizing risk parity: {e}')
            n = len(symbols)
            return dict.fromkeys(symbols, 1.0 / n)

class CorrelationClusterSizer:
    """
    Correlation-based cluster sizing.

    Groups assets by correlation and applies cluster-level
    position limits to manage concentration risk.
    """

    def __init__(self, correlation_threshold: float=0.7, max_cluster_weight: float=0.5, lookback_days: int=60):
        """
        Initialize correlation cluster sizer.

        Args:
            correlation_threshold: Correlation threshold for clustering
            max_cluster_weight: Maximum weight per cluster
            lookback_days: Lookback period for correlation estimation
        """
        self.correlation_threshold = correlation_threshold
        self.max_cluster_weight = max_cluster_weight
        self.lookback_days = lookback_days
        logger.info('CorrelationClusterSizer initialized')

    def apply_cluster_limits(self, base_weights: dict[str, float], price_history: dict[str, pd.Series]) -> dict[str, float]:
        """
        Apply cluster-based position limits.

        Args:
            base_weights: Base position weights
            price_history: Historical prices for correlation estimation

        Returns:
            Cluster-adjusted weights
        """
        try:
            symbols = list(base_weights.keys())
            if len(symbols) < 3:
                return base_weights
            correlation_matrix = self._calculate_correlation_matrix(symbols, price_history)
            if correlation_matrix is None:
                return base_weights
            clusters = self._perform_clustering(symbols, correlation_matrix)
            adjusted_weights = self._apply_cluster_constraints(base_weights, clusters)
            return adjusted_weights
        except COMMON_EXC as e:
            logger.error(f'Error applying cluster limits: {e}')
            return base_weights

    def _calculate_correlation_matrix(self, symbols: list[str], price_history: dict[str, pd.Series]) -> np.ndarray | None:
        """Calculate correlation matrix."""
        pd = load_pandas()
        try:
            return_series = {}
            for symbol in symbols:
                if symbol in price_history:
                    prices = price_history[symbol].tail(self.lookback_days)
                    if len(prices) > 10:
                        returns = prices.pct_change().dropna()
                        return_series[symbol] = returns
            if len(return_series) < 2:
                return None
            df = pd.DataFrame(return_series)
            df = df.dropna()
            if len(df) < 10:
                return None
            correlation_matrix = df.corr().values
            return correlation_matrix
        except COMMON_EXC as e:
            logger.error(f'Error calculating correlation matrix: {e}')
            return None

    def _perform_clustering(self, symbols: list[str], correlation_matrix: np.ndarray) -> dict[int, list[str]]:
        """Perform hierarchical clustering based on correlations."""
        try:
            fcluster, linkage, squareform, clustering_available = _import_clustering()
            if not clustering_available:
                n = len(symbols)
                clusters = {}
                cluster_size = max(1, n // 3)
                for i, symbol in enumerate(symbols):
                    cluster_id = i // cluster_size
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(symbol)
                return clusters
            distance_matrix = 1 - np.abs(correlation_matrix)
            np.fill_diagonal(distance_matrix, 0)
            distance_matrix = squareform(distance_matrix)
            linkage_matrix = linkage(distance_matrix, method='ward')
            distance_threshold = 1 - self.correlation_threshold
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
            clusters = {}
            for i, symbol in enumerate(symbols):
                cluster_id = cluster_labels[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(symbol)
            return clusters
        except COMMON_EXC as e:
            logger.error(f'Error performing clustering: {e}')
            return {i: [symbol] for i, symbol in enumerate(symbols)}

    def _apply_cluster_constraints(self, base_weights: dict[str, float], clusters: dict[int, list[str]]) -> dict[str, float]:
        """Apply cluster weight constraints."""
        try:
            adjusted_weights = base_weights.copy()
            for _cluster_id, cluster_symbols in clusters.items():
                cluster_weight = sum((base_weights.get(symbol, 0) for symbol in cluster_symbols))
                if cluster_weight > self.max_cluster_weight:
                    scale_factor = self.max_cluster_weight / cluster_weight
                    for symbol in cluster_symbols:
                        if symbol in adjusted_weights:
                            adjusted_weights[symbol] *= scale_factor
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                for symbol in adjusted_weights:
                    adjusted_weights[symbol] /= total_weight
            return adjusted_weights
        except COMMON_EXC as e:
            logger.error(f'Error applying cluster constraints: {e}')
            return base_weights

class TurnoverPenaltySizer:
    """
    Position sizer with turnover penalty.

    Penalizes excessive turnover to reduce transaction costs
    and improve net returns.
    """

    def __init__(self, turnover_penalty: float=0.001, max_turnover: float=0.5, lookback_periods: int=20):
        """
        Initialize turnover penalty sizer.

        Args:
            turnover_penalty: Penalty factor for turnover
            max_turnover: Maximum allowed turnover
            lookback_periods: Lookback for turnover calculation
        """
        self.turnover_penalty = turnover_penalty
        self.max_turnover = max_turnover
        self.lookback_periods = lookback_periods
        self.position_history = []
        logger.info('TurnoverPenaltySizer initialized')

    def apply_turnover_penalty(self, proposed_weights: dict[str, float], current_weights: dict[str, float]) -> dict[str, float]:
        """
        Apply turnover penalty to proposed weights.

        Args:
            proposed_weights: Proposed new weights
            current_weights: Current position weights

        Returns:
            Turnover-adjusted weights
        """
        try:
            turnover = self._calculate_turnover(proposed_weights, current_weights)
            if turnover <= self.max_turnover:
                self._update_position_history(proposed_weights)
                return proposed_weights
            adjusted_weights = self._reduce_turnover(proposed_weights, current_weights, turnover)
            self._update_position_history(adjusted_weights)
            return adjusted_weights
        except COMMON_EXC as e:
            logger.error(f'Error applying turnover penalty: {e}')
            return proposed_weights

    def _calculate_turnover(self, new_weights: dict[str, float], old_weights: dict[str, float]) -> float:
        """Calculate portfolio turnover."""
        try:
            all_symbols = set(new_weights.keys()) | set(old_weights.keys())
            total_change = 0.0
            for symbol in all_symbols:
                old_weight = old_weights.get(symbol, 0.0)
                new_weight = new_weights.get(symbol, 0.0)
                total_change += abs(new_weight - old_weight)
            turnover = total_change / 2.0
            return turnover
        except COMMON_EXC as e:
            logger.error(f'Error calculating turnover: {e}')
            return 0.0

    def _reduce_turnover(self, proposed_weights: dict[str, float], current_weights: dict[str, float], current_turnover: float) -> dict[str, float]:
        """Reduce turnover by dampening position changes."""
        try:
            target_turnover = self.max_turnover
            dampen_factor = target_turnover / current_turnover
            adjusted_weights = {}
            all_symbols = set(proposed_weights.keys()) | set(current_weights.keys())
            for symbol in all_symbols:
                current_weight = current_weights.get(symbol, 0.0)
                proposed_weight = proposed_weights.get(symbol, 0.0)
                weight_change = proposed_weight - current_weight
                dampened_change = weight_change * dampen_factor
                adjusted_weights[symbol] = current_weight + dampened_change
            adjusted_weights = {k: v for k, v in adjusted_weights.items() if abs(v) > 0.001}
            total_weight = sum((abs(w) for w in adjusted_weights.values()))
            if total_weight > 0:
                for symbol in adjusted_weights:
                    adjusted_weights[symbol] /= total_weight
            return adjusted_weights
        except COMMON_EXC as e:
            logger.error(f'Error reducing turnover: {e}')
            return proposed_weights

    def _update_position_history(self, weights: dict[str, float]) -> None:
        """Update position history for turnover tracking."""
        try:
            self.position_history.append({'timestamp': datetime.now(UTC), 'weights': weights.copy()})
            if len(self.position_history) > self.lookback_periods:
                self.position_history = self.position_history[-self.lookback_periods:]
        except COMMON_EXC as e:
            logger.error(f'Error updating position history: {e}')

    def get_historical_turnover(self) -> list[float]:
        """Get historical turnover rates."""
        try:
            if len(self.position_history) < 2:
                return []
            turnovers = []
            for i in range(1, len(self.position_history)):
                current = self.position_history[i]['weights']
                previous = self.position_history[i - 1]['weights']
                turnover = self._calculate_turnover(current, previous)
                turnovers.append(turnover)
            return turnovers
        except COMMON_EXC as e:
            logger.error(f'Error getting historical turnover: {e}')
            return []
