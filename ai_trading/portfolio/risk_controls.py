"""
Adaptive risk controls and exposure management.

Provides vol-targeting, adaptive Kelly sizing, correlation clustering,
cluster exposure caps, and turnover budget enforcement.
"""
from ai_trading.logging import get_logger
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - heavy import for type checking only
    import pandas as pd

def _import_clustering():
    from ai_trading.config import get_settings
    S = get_settings()
    if not S.ENABLE_PORTFOLIO_FEATURES:
        return (None, None, None, False)
logger = get_logger(__name__)

@dataclass
class RiskBudget:
    """Risk budget parameters."""
    total_risk_target: float = 0.1
    max_position_risk: float = 0.02
    max_cluster_risk: float = 0.25
    max_turnover_daily: float = 1.0
    drawdown_threshold: float = 0.08
    kelly_multiplier: float = 0.25
    vol_lookback_days: int = 20
    corr_lookback_days: int = 60

@dataclass
class PositionRisk:
    """Risk metrics for a position."""
    symbol: str
    position_value: float
    daily_vol: float
    risk_contribution: float
    kelly_fraction: float
    cluster_id: int | None = None
    weight: float = 0.0

@dataclass
class ClusterRisk:
    """Risk metrics for a correlation cluster."""
    cluster_id: int
    symbols: list[str]
    total_risk: float
    risk_limit: float
    is_over_limit: bool = False

@dataclass
class TurnoverBudget:
    """Daily turnover tracking."""
    date: datetime = field(default_factory=lambda: datetime.now(UTC).date())
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

    def __init__(self, risk_budget: RiskBudget | None=None):
        """
        Initialize risk controller.

        Args:
            risk_budget: Risk budget parameters
        """
        self.risk_budget = risk_budget or RiskBudget()
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self.drawdown_multiplier = 1.0
        self.green_days_count = 0
        self.turnover_budget = TurnoverBudget(total_budget=self.risk_budget.max_turnover_daily)
        self._correlation_matrix: 'pd.DataFrame | None' = None
        self._cluster_assignments: dict[str, int] = {}
        self._volatilities: dict[str, float] = {}

    def calculate_volatilities(self, returns_data: 'pd.DataFrame', lookback_days: int | None=None) -> dict[str, float]:
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
        import pandas as pd  # heavy import; keep local
        volatilities = {}
        for symbol in returns_data.columns:
            symbol_returns = returns_data[symbol].dropna()
            if len(symbol_returns) >= lookback_days:
                recent_returns = symbol_returns.tail(lookback_days)
                daily_vol = recent_returns.std()
                annual_vol = daily_vol * np.sqrt(252)
                volatilities[symbol] = annual_vol
            else:
                volatilities[symbol] = 0.2
        self._volatilities = volatilities
        return volatilities

    def calculate_correlation_clusters(self, returns_data: 'pd.DataFrame', lookback_days: int | None=None, max_clusters: int=10) -> dict[str, int]:
        """
        Calculate correlation-based clusters for risk diversification.

        Args:
            returns_data: DataFrame with symbol returns
            lookback_days: Lookback period for correlation
            max_clusters: Maximum number of clusters

        Returns:
            Dict of symbol -> cluster_id
        """
        import pandas as pd  # heavy import; keep local
        fcluster, linkage, squareform, clustering_available = _import_clustering()
        if not clustering_available:
            self.logger.warning('Clustering not available, using single cluster')
            return dict.fromkeys(returns_data.columns, 0)
        if lookback_days is None:
            lookback_days = self.risk_budget.corr_lookback_days
        recent_data = returns_data.tail(lookback_days)
        correlation_matrix = recent_data.corr()
        correlation_matrix = correlation_matrix.fillna(0)
        if len(correlation_matrix) < 2:
            return dict.fromkeys(returns_data.columns, 0)
        try:
            distance_matrix = 1 - correlation_matrix.abs()
            distance_condensed = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(distance_condensed, method='ward')
            num_symbols = len(correlation_matrix)
            num_clusters = min(max_clusters, max(1, num_symbols // 3))
            cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
            cluster_assignments = {}
            for i, symbol in enumerate(correlation_matrix.index):
                cluster_assignments[symbol] = cluster_labels[i] - 1
            self._correlation_matrix = correlation_matrix
            self._cluster_assignments = cluster_assignments
            self.logger.info(f'Created {num_clusters} correlation clusters for {num_symbols} symbols')
            return cluster_assignments
        except (ValueError, TypeError) as e:
            self.logger.error(f'Clustering failed: {e}')
            return dict.fromkeys(returns_data.columns, 0)

    def calculate_kelly_fractions(self, expected_returns: dict[str, float], volatilities: dict[str, float], correlation_matrix: 'pd.DataFrame | None'=None) -> dict[str, float]:
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
        for symbol in expected_returns:
            if symbol not in volatilities:
                kelly_fractions[symbol] = 0.0
                continue
            expected_return = expected_returns[symbol]
            volatility = volatilities[symbol]
            if volatility <= 0:
                kelly_fractions[symbol] = 0.0
                continue
            kelly_fraction = expected_return / volatility ** 2
            kelly_fraction *= self.risk_budget.kelly_multiplier
            kelly_fraction *= self.drawdown_multiplier
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
        if portfolio_return < -self.risk_budget.drawdown_threshold:
            self.drawdown_multiplier = 0.5
            self.green_days_count = 0
            self.logger.warning(f'Drawdown detected, reducing risk multiplier to {self.drawdown_multiplier}')
        elif is_positive_day:
            self.green_days_count += 1
            if self.green_days_count >= 5 and self.drawdown_multiplier < 1.0:
                self.drawdown_multiplier = min(1.0, self.drawdown_multiplier + 0.1)
                self.logger.info(f'Risk recovery: multiplier increased to {self.drawdown_multiplier}')
        else:
            self.green_days_count = 0

    def check_cluster_limits(self, positions: dict[str, PositionRisk], cluster_assignments: dict[str, int]) -> list[ClusterRisk]:
        """
        Check cluster risk limits.

        Args:
            positions: Current position risks
            cluster_assignments: Symbol cluster assignments

        Returns:
            List of cluster risk metrics
        """

        def _new_cluster() -> dict[str, Any]:
            return {'symbols': [], 'total_risk': 0.0}
        cluster_risks = defaultdict(_new_cluster)
        for symbol, position in positions.items():
            cluster_id = cluster_assignments.get(symbol, 0)
            cluster_risks[cluster_id]['symbols'].append(symbol)
            cluster_risks[cluster_id]['total_risk'] += position.risk_contribution
        cluster_risk_objects = []
        for cluster_id, risk_data in cluster_risks.items():
            risk_limit = self.risk_budget.max_cluster_risk
            cluster_risk = ClusterRisk(cluster_id=cluster_id, symbols=risk_data['symbols'], total_risk=risk_data['total_risk'], risk_limit=risk_limit, is_over_limit=risk_data['total_risk'] > risk_limit)
            cluster_risk_objects.append(cluster_risk)
            if cluster_risk.is_over_limit:
                self.logger.warning(f'Cluster {cluster_id} over limit: {cluster_risk.total_risk:.2%} > {risk_limit:.2%}')
        return cluster_risk_objects

    def calculate_position_sizes(self, signals: dict[str, float], returns_data: 'pd.DataFrame', portfolio_value: float, current_positions: dict[str, float] | None=None) -> dict[str, float]:
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
        volatilities = self.calculate_volatilities(returns_data)
        cluster_assignments = self.calculate_correlation_clusters(returns_data)
        expected_returns = {symbol: signal * 0.01 for symbol, signal in signals.items()}
        kelly_fractions = self.calculate_kelly_fractions(expected_returns, volatilities)
        position_risks = {}
        for symbol in signals:
            if symbol in volatilities and symbol in kelly_fractions:
                kelly_fraction = kelly_fractions[symbol]
                volatility = volatilities[symbol]
                position_value = kelly_fraction * portfolio_value
                risk_contribution = position_value / portfolio_value * volatility
                position_risks[symbol] = PositionRisk(symbol=symbol, position_value=position_value, daily_vol=volatility / np.sqrt(252), risk_contribution=risk_contribution, kelly_fraction=kelly_fraction, cluster_id=cluster_assignments.get(symbol, 0), weight=position_value / portfolio_value if portfolio_value > 0 else 0)
        cluster_risks = self.check_cluster_limits(position_risks, cluster_assignments)
        for cluster_risk in cluster_risks:
            if cluster_risk.is_over_limit:
                scale_factor = cluster_risk.risk_limit / cluster_risk.total_risk
                for symbol in cluster_risk.symbols:
                    if symbol in position_risks:
                        position_risks[symbol].position_value *= scale_factor
                        position_risks[symbol].risk_contribution *= scale_factor
        target_positions = {}
        for symbol, position_risk in position_risks.items():
            current_value = current_positions.get(symbol, 0.0)
            target_value = position_risk.position_value
            trade_value = abs(target_value - current_value)
            if trade_value > 0 and (not self.turnover_budget.add_trade(trade_value, portfolio_value)):
                available_turnover = self.turnover_budget.remaining_turnover * portfolio_value
                if available_turnover > 0:
                    scale_factor = available_turnover / trade_value
                    target_value = current_value + (target_value - current_value) * scale_factor
                    self.logger.warning(f'Scaled trade for {symbol} due to turnover limit: factor={scale_factor:.2%}')
                else:
                    target_value = current_value
                    self.logger.warning(f'No turnover budget for {symbol}, keeping current position')
            target_positions[symbol] = target_value
        return target_positions

    def reset_daily_budget(self) -> None:
        """Reset daily turnover budget."""
        today = datetime.now(UTC).date()
        if self.turnover_budget.date != today:
            self.turnover_budget = TurnoverBudget(date=today, total_budget=self.risk_budget.max_turnover_daily)
            self.logger.info(f'Reset daily turnover budget: {self.risk_budget.max_turnover_daily:.1%}')

    def get_risk_summary(self) -> dict:
        """Get risk control summary."""
        return {'drawdown_multiplier': self.drawdown_multiplier, 'green_days_count': self.green_days_count, 'turnover_used': self.turnover_budget.used_turnover, 'turnover_remaining': self.turnover_budget.remaining_turnover, 'num_clusters': len(set(self._cluster_assignments.values())) if self._cluster_assignments else 0, 'num_symbols': len(self._volatilities)}
_global_risk_controller: AdaptiveRiskController | None = None

def get_risk_controller() -> AdaptiveRiskController:
    """Get or create global risk controller instance."""
    global _global_risk_controller
    if _global_risk_controller is None:
        _global_risk_controller = AdaptiveRiskController()
    return _global_risk_controller

def calculate_adaptive_positions(signals: dict[str, float], returns_data: 'pd.DataFrame', portfolio_value: float, current_positions: dict[str, float] | None=None) -> dict[str, float]:
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
    return controller.calculate_position_sizes(signals, returns_data, portfolio_value, current_positions)