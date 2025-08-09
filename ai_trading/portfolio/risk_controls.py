"""
Adaptive risk controls and exposure management.

Provides vol-targeting, adaptive Kelly sizing, correlation clustering,
cluster exposure caps, and turnover budget enforcement.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class RiskBudget:
    """Risk budget parameters."""
    total_risk_target: float = 0.10  # 10% volatility target
    max_position_risk: float = 0.02  # 2% max per position
    max_cluster_risk: float = 0.25   # 25% max per cluster
    max_turnover_daily: float = 1.0  # 100% daily turnover limit
    drawdown_threshold: float = 0.08 # 8% drawdown threshold
    
    # Adaptive parameters
    kelly_multiplier: float = 0.25   # Conservative Kelly scaling
    vol_lookback_days: int = 20      # Volatility estimation window
    corr_lookback_days: int = 60     # Correlation estimation window


@dataclass
class PositionRisk:
    """Risk metrics for a position."""
    symbol: str
    position_value: float
    daily_vol: float
    risk_contribution: float
    kelly_fraction: float
    cluster_id: Optional[int] = None
    weight: float = 0.0


@dataclass 
class ClusterRisk:
    """Risk metrics for a correlation cluster."""
    cluster_id: int
    symbols: List[str]
    total_risk: float
    risk_limit: float
    is_over_limit: bool = False


@dataclass
class TurnoverBudget:
    """Daily turnover tracking."""
    date: datetime = field(default_factory=lambda: datetime.now(timezone.utc).date())
    used_turnover: float = 0.0
    remaining_turnover: float = 1.0
    total_budget: float = 1.0
    
    def add_trade(self, trade_value: float, portfolio_value: float) -> bool:
        """
        Add trade to turnover budget.
        
        Args:
            trade_value: Absolute value of trade
            portfolio_value: Total portfolio value
            
        Returns:
            True if trade is within budget
        """
        if portfolio_value <= 0:
            return False
            
        trade_turnover = trade_value / portfolio_value
        
        if self.used_turnover + trade_turnover <= self.total_budget:
            self.used_turnover += trade_turnover
            self.remaining_turnover = self.total_budget - self.used_turnover
            return True
        
        return False


class AdaptiveRiskController:
    """
    Adaptive risk control system with volatility targeting, Kelly sizing,
    correlation clustering, and turnover management.
    """
    
    def __init__(self, risk_budget: Optional[RiskBudget] = None):
        """
        Initialize risk controller.
        
        Args:
            risk_budget: Risk budget parameters
        """
        self.risk_budget = risk_budget or RiskBudget()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # State tracking
        self.drawdown_multiplier = 1.0
        self.green_days_count = 0
        self.turnover_budget = TurnoverBudget(total_budget=self.risk_budget.max_turnover_daily)
        
        # Cache for expensive calculations
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._cluster_assignments: Dict[str, int] = {}
        self._volatilities: Dict[str, float] = {}
        
    def calculate_volatilities(
        self, 
        returns_data: pd.DataFrame,
        lookback_days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate rolling volatilities for symbols.
        
        Args:
            returns_data: DataFrame with symbol returns
            lookback_days: Lookback period for volatility
            
        Returns:
            Dict of symbol -> annualized volatility
        """
        if lookback_days is None:
            lookback_days = self.risk_budget.vol_lookback_days
        
        volatilities = {}
        
        for symbol in returns_data.columns:
            symbol_returns = returns_data[symbol].dropna()
            
            if len(symbol_returns) >= lookback_days:
                # Use recent period for volatility estimation
                recent_returns = symbol_returns.tail(lookback_days)
                daily_vol = recent_returns.std()
                annual_vol = daily_vol * np.sqrt(252)  # Annualize
                volatilities[symbol] = annual_vol
            else:
                # Default volatility for new symbols
                volatilities[symbol] = 0.20  # 20% annual vol
        
        self._volatilities = volatilities
        return volatilities
    
    def calculate_correlation_clusters(
        self,
        returns_data: pd.DataFrame,
        lookback_days: Optional[int] = None,
        max_clusters: int = 10
    ) -> Dict[str, int]:
        """
        Calculate correlation-based clusters for risk diversification.
        
        Args:
            returns_data: DataFrame with symbol returns
            lookback_days: Lookback period for correlation
            max_clusters: Maximum number of clusters
            
        Returns:
            Dict of symbol -> cluster_id
        """
        if not CLUSTERING_AVAILABLE:
            self.logger.warning("Clustering not available, using single cluster")
            return {symbol: 0 for symbol in returns_data.columns}
        
        if lookback_days is None:
            lookback_days = self.risk_budget.corr_lookback_days
        
        # Calculate correlation matrix
        recent_data = returns_data.tail(lookback_days)
        correlation_matrix = recent_data.corr()
        
        # Handle missing data
        correlation_matrix = correlation_matrix.fillna(0)
        
        if len(correlation_matrix) < 2:
            return {symbol: 0 for symbol in returns_data.columns}
        
        try:
            # Convert correlation to distance
            distance_matrix = 1 - correlation_matrix.abs()
            
            # Perform hierarchical clustering
            distance_condensed = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(distance_condensed, method='ward')
            
            # Determine number of clusters
            num_symbols = len(correlation_matrix)
            num_clusters = min(max_clusters, max(1, num_symbols // 3))
            
            # Get cluster assignments
            cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
            
            cluster_assignments = {}
            for i, symbol in enumerate(correlation_matrix.index):
                cluster_assignments[symbol] = cluster_labels[i] - 1  # 0-indexed
            
            self._correlation_matrix = correlation_matrix
            self._cluster_assignments = cluster_assignments
            
            self.logger.info(f"Created {num_clusters} correlation clusters for {num_symbols} symbols")
            
            return cluster_assignments
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {symbol: 0 for symbol in returns_data.columns}
    
    def calculate_kelly_fractions(
        self,
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate Kelly fractions for position sizing.
        
        Args:
            expected_returns: Expected returns by symbol
            volatilities: Volatilities by symbol
            correlation_matrix: Correlation matrix for risk adjustment
            
        Returns:
            Dict of symbol -> Kelly fraction
        """
        kelly_fractions = {}
        
        for symbol in expected_returns.keys():
            if symbol not in volatilities:
                kelly_fractions[symbol] = 0.0
                continue
            
            expected_return = expected_returns[symbol]
            volatility = volatilities[symbol]
            
            if volatility <= 0:
                kelly_fractions[symbol] = 0.0
                continue
            
            # Basic Kelly fraction: μ / σ²
            kelly_fraction = expected_return / (volatility ** 2)
            
            # Apply conservative multiplier
            kelly_fraction *= self.risk_budget.kelly_multiplier
            
            # Apply drawdown governor
            kelly_fraction *= self.drawdown_multiplier
            
            # Cap at maximum position risk
            max_kelly = self.risk_budget.max_position_risk / volatility
            kelly_fraction = min(kelly_fraction, max_kelly)
            
            kelly_fractions[symbol] = max(0.0, kelly_fraction)
        
        return kelly_fractions
    
    def update_drawdown_governor(self, portfolio_return: float, is_positive_day: bool) -> None:
        """
        Update drawdown governor based on recent performance.
        
        Args:
            portfolio_return: Daily portfolio return
            is_positive_day: Whether today was a positive day
        """
        # Simple drawdown detection (would need proper equity curve tracking)
        if portfolio_return < -self.risk_budget.drawdown_threshold:
            # Reduce risk in drawdown
            self.drawdown_multiplier = 0.5
            self.green_days_count = 0
            self.logger.warning(f"Drawdown detected, reducing risk multiplier to {self.drawdown_multiplier}")
        
        elif is_positive_day:
            self.green_days_count += 1
            # Gradually recover risk capacity
            if self.green_days_count >= 5 and self.drawdown_multiplier < 1.0:
                self.drawdown_multiplier = min(1.0, self.drawdown_multiplier + 0.1)
                self.logger.info(f"Risk recovery: multiplier increased to {self.drawdown_multiplier}")
        
        else:
            self.green_days_count = 0
    
    def check_cluster_limits(
        self,
        positions: Dict[str, PositionRisk],
        cluster_assignments: Dict[str, int]
    ) -> List[ClusterRisk]:
        """
        Check cluster risk limits.
        
        Args:
            positions: Current position risks
            cluster_assignments: Symbol cluster assignments
            
        Returns:
            List of cluster risk metrics
        """
        # Group positions by cluster
        cluster_risks = defaultdict(lambda: {'symbols': [], 'total_risk': 0.0})
        
        for symbol, position in positions.items():
            cluster_id = cluster_assignments.get(symbol, 0)
            cluster_risks[cluster_id]['symbols'].append(symbol)
            cluster_risks[cluster_id]['total_risk'] += position.risk_contribution
        
        # Convert to ClusterRisk objects
        cluster_risk_objects = []
        for cluster_id, risk_data in cluster_risks.items():
            risk_limit = self.risk_budget.max_cluster_risk
            
            cluster_risk = ClusterRisk(
                cluster_id=cluster_id,
                symbols=risk_data['symbols'],
                total_risk=risk_data['total_risk'],
                risk_limit=risk_limit,
                is_over_limit=risk_data['total_risk'] > risk_limit
            )
            
            cluster_risk_objects.append(cluster_risk)
            
            if cluster_risk.is_over_limit:
                self.logger.warning(
                    f"Cluster {cluster_id} over limit: "
                    f"{cluster_risk.total_risk:.2%} > {risk_limit:.2%}"
                )
        
        return cluster_risk_objects
    
    def calculate_position_sizes(
        self,
        signals: Dict[str, float],
        returns_data: pd.DataFrame,
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate position sizes with full risk controls.
        
        Args:
            signals: Signal strengths by symbol
            returns_data: Historical returns for risk calculation
            portfolio_value: Current portfolio value
            current_positions: Current position values
            
        Returns:
            Dict of symbol -> target position value
        """
        if current_positions is None:
            current_positions = {}
        
        # Calculate risk metrics
        volatilities = self.calculate_volatilities(returns_data)
        cluster_assignments = self.calculate_correlation_clusters(returns_data)
        
        # Convert signals to expected returns (simplified)
        expected_returns = {symbol: signal * 0.01 for symbol, signal in signals.items()}  # 1% per signal unit
        
        # Calculate Kelly fractions
        kelly_fractions = self.calculate_kelly_fractions(expected_returns, volatilities)
        
        # Calculate position risks
        position_risks = {}
        for symbol in signals.keys():
            if symbol in volatilities and symbol in kelly_fractions:
                kelly_fraction = kelly_fractions[symbol]
                volatility = volatilities[symbol]
                position_value = kelly_fraction * portfolio_value
                risk_contribution = (position_value / portfolio_value) * volatility
                
                position_risks[symbol] = PositionRisk(
                    symbol=symbol,
                    position_value=position_value,
                    daily_vol=volatility / np.sqrt(252),  # Daily vol
                    risk_contribution=risk_contribution,
                    kelly_fraction=kelly_fraction,
                    cluster_id=cluster_assignments.get(symbol, 0),
                    weight=position_value / portfolio_value if portfolio_value > 0 else 0
                )
        
        # Check cluster limits and adjust if needed
        cluster_risks = self.check_cluster_limits(position_risks, cluster_assignments)
        
        # Scale down over-limit clusters
        for cluster_risk in cluster_risks:
            if cluster_risk.is_over_limit:
                scale_factor = cluster_risk.risk_limit / cluster_risk.total_risk
                
                for symbol in cluster_risk.symbols:
                    if symbol in position_risks:
                        position_risks[symbol].position_value *= scale_factor
                        position_risks[symbol].risk_contribution *= scale_factor
        
        # Apply turnover budget constraints
        target_positions = {}
        for symbol, position_risk in position_risks.items():
            current_value = current_positions.get(symbol, 0.0)
            target_value = position_risk.position_value
            trade_value = abs(target_value - current_value)
            
            # Check turnover budget
            if trade_value > 0 and not self.turnover_budget.add_trade(trade_value, portfolio_value):
                # Scale down trade to fit budget
                available_turnover = self.turnover_budget.remaining_turnover * portfolio_value
                if available_turnover > 0:
                    scale_factor = available_turnover / trade_value
                    target_value = current_value + (target_value - current_value) * scale_factor
                    
                    self.logger.warning(
                        f"Scaled trade for {symbol} due to turnover limit: "
                        f"factor={scale_factor:.2%}"
                    )
                else:
                    # No turnover budget left
                    target_value = current_value
                    
                    self.logger.warning(f"No turnover budget for {symbol}, keeping current position")
            
            target_positions[symbol] = target_value
        
        return target_positions
    
    def reset_daily_budget(self) -> None:
        """Reset daily turnover budget."""
        today = datetime.now(timezone.utc).date()
        
        if self.turnover_budget.date != today:
            self.turnover_budget = TurnoverBudget(
                date=today,
                total_budget=self.risk_budget.max_turnover_daily
            )
            self.logger.info(f"Reset daily turnover budget: {self.risk_budget.max_turnover_daily:.1%}")
    
    def get_risk_summary(self) -> Dict:
        """Get risk control summary."""
        return {
            'drawdown_multiplier': self.drawdown_multiplier,
            'green_days_count': self.green_days_count,
            'turnover_used': self.turnover_budget.used_turnover,
            'turnover_remaining': self.turnover_budget.remaining_turnover,
            'num_clusters': len(set(self._cluster_assignments.values())) if self._cluster_assignments else 0,
            'num_symbols': len(self._volatilities)
        }


# Global risk controller instance
_global_risk_controller: Optional[AdaptiveRiskController] = None


def get_risk_controller() -> AdaptiveRiskController:
    """Get or create global risk controller instance."""
    global _global_risk_controller
    if _global_risk_controller is None:
        _global_risk_controller = AdaptiveRiskController()
    return _global_risk_controller


def calculate_adaptive_positions(
    signals: Dict[str, float],
    returns_data: pd.DataFrame,
    portfolio_value: float,
    current_positions: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Convenience function for adaptive position sizing.
    
    Args:
        signals: Signal strengths by symbol
        returns_data: Historical returns
        portfolio_value: Current portfolio value
        current_positions: Current positions
        
    Returns:
        Dict of symbol -> target position value
    """
    controller = get_risk_controller()
    return controller.calculate_position_sizes(
        signals, returns_data, portfolio_value, current_positions
    )