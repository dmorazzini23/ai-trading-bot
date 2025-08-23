"""
Adaptive Position Sizing with Market Condition Awareness.

Implements institutional-grade adaptive position sizing that adjusts to market conditions,
volatility regimes, correlation environments, and risk-adjusted portfolio allocation.
"""
import math
import statistics
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from ai_trading.exc import COMMON_EXC
from ai_trading.logging import logger
from ..core.constants import RISK_PARAMETERS
from ..core.enums import RiskLevel
from .kelly import KellyCalculator
from .position_sizing import DynamicPositionSizer

class MarketRegime(Enum):
    """Market regime classification for adaptive sizing."""
    BULL_TRENDING = 'bull_trending'
    BEAR_TRENDING = 'bear_trending'
    SIDEWAYS_RANGE = 'sideways_range'
    HIGH_VOLATILITY = 'high_volatility'
    LOW_VOLATILITY = 'low_volatility'
    CRISIS = 'crisis'
    NORMAL = 'normal'

class VolatilityRegime(Enum):
    """Volatility regime classification."""
    EXTREMELY_LOW = 'extremely_low'
    LOW = 'low'
    NORMAL = 'normal'
    HIGH = 'high'
    EXTREMELY_HIGH = 'extremely_high'

class MarketConditionAnalyzer:
    """
    Analyzes market conditions for adaptive position sizing.

    Provides regime detection, volatility analysis, correlation assessment,
    and risk environment classification for position sizing optimization.
    """

    def __init__(self, lookback_days: int=60):
        """Initialize market condition analyzer."""
        self.lookback_days = lookback_days
        self.volatility_window = 20
        self.correlation_window = 30
        self.vol_thresholds = {'extremely_low': 0.12, 'low': 0.28, 'high': 0.72, 'extremely_high': 0.88}
        logger.info(f'MarketConditionAnalyzer initialized with lookback_days={lookback_days}')

    def analyze_market_regime(self, price_data: dict[str, list[float]], volume_data: dict[str, list[float]] | None=None) -> MarketRegime:
        """
        Analyze current market regime based on price and volume data.

        Args:
            price_data: Dictionary mapping symbols to price lists
            volume_data: Optional volume data for confirmation

        Returns:
            Detected market regime
        """
        try:
            if not price_data:
                return MarketRegime.NORMAL
            market_symbol = self._get_market_proxy(price_data)
            if not market_symbol:
                return MarketRegime.NORMAL
            prices = price_data[market_symbol]
            if len(prices) < self.lookback_days:
                return MarketRegime.NORMAL
            recent_prices = prices[-self.lookback_days:]
            trend_strength = self._calculate_trend_strength(recent_prices)
            volatility = self._calculate_rolling_volatility(recent_prices)
            vol_percentile = self._calculate_volatility_percentile(volatility, recent_prices)
            if vol_percentile > self.vol_thresholds['extremely_high']:
                if trend_strength < -0.5:
                    return MarketRegime.CRISIS
                else:
                    return MarketRegime.HIGH_VOLATILITY
            elif vol_percentile < self.vol_thresholds['extremely_low']:
                return MarketRegime.LOW_VOLATILITY
            elif trend_strength > 0.3:
                return MarketRegime.BULL_TRENDING
            elif trend_strength < -0.3:
                return MarketRegime.BEAR_TRENDING
            else:
                return MarketRegime.SIDEWAYS_RANGE
        except COMMON_EXC as e:
            logger.error(f'Error analyzing market regime: {e}')
            return MarketRegime.NORMAL

    def assess_volatility_regime(self, returns: list[float]) -> VolatilityRegime:
        """
        Assess current volatility regime based on returns.

        Args:
            returns: List of asset returns

        Returns:
            Volatility regime classification
        """
        try:
            if len(returns) < self.volatility_window:
                return VolatilityRegime.NORMAL
            recent_returns = returns[-self.volatility_window:]
            current_vol = statistics.stdev(recent_returns) * math.sqrt(252)
            if len(returns) >= self.lookback_days:
                historical_vols = []
                for i in range(self.volatility_window, min(len(returns), self.lookback_days)):
                    window_returns = returns[i - self.volatility_window:i]
                    vol = statistics.stdev(window_returns) * math.sqrt(252)
                    historical_vols.append(vol)
                if historical_vols:
                    historical_vols.sort()
                    percentile = self._get_percentile_rank(current_vol, historical_vols)
                    if percentile >= self.vol_thresholds['extremely_high']:
                        return VolatilityRegime.EXTREMELY_HIGH
                    elif percentile >= self.vol_thresholds['high']:
                        return VolatilityRegime.HIGH
                    elif percentile <= self.vol_thresholds['extremely_low']:
                        return VolatilityRegime.EXTREMELY_LOW
                    elif percentile <= self.vol_thresholds['low']:
                        return VolatilityRegime.LOW
                    else:
                        return VolatilityRegime.NORMAL
            return VolatilityRegime.NORMAL
        except COMMON_EXC as e:
            logger.error(f'Error assessing volatility regime: {e}')
            return VolatilityRegime.NORMAL

    def calculate_correlation_matrix(self, returns_data: dict[str, list[float]]) -> dict[str, float]:
        """
        Calculate correlation matrix for portfolio assets.

        Args:
            returns_data: Dictionary mapping symbols to return lists

        Returns:
            Dictionary mapping symbol pairs to correlation coefficients
        """
        try:
            correlations = {}
            symbols = list(returns_data.keys())
            for i, symbol1 in enumerate(symbols):
                for _j, symbol2 in enumerate(symbols[i + 1:], i + 1):
                    returns1 = returns_data[symbol1]
                    returns2 = returns_data[symbol2]
                    min_length = min(len(returns1), len(returns2))
                    if min_length >= self.correlation_window:
                        r1 = returns1[-min_length:]
                        r2 = returns2[-min_length:]
                        correlation = self._calculate_correlation(r1, r2)
                        correlations[f'{symbol1}_{symbol2}'] = correlation
                        correlations[f'{symbol2}_{symbol1}'] = correlation
            return correlations
        except COMMON_EXC as e:
            logger.error(f'Error calculating correlation matrix: {e}')
            return {}

    def _get_market_proxy(self, price_data: dict[str, list[float]]) -> str | None:
        """Get market proxy symbol from available data."""
        market_proxies = ['SPY', 'QQQ', 'IWM', 'VTI', 'TQQQ']
        for proxy in market_proxies:
            if proxy in price_data:
                return proxy
        return list(price_data.keys())[0] if price_data else None

    def _calculate_trend_strength(self, prices: list[float]) -> float:
        """Calculate trend strength using linear regression slope."""
        try:
            if len(prices) < 10:
                return 0.0
            n = len(prices)
            x_values = list(range(n))
            sum_x = sum(x_values)
            sum_y = sum(prices)
            sum_xy = sum((x * y for x, y in zip(x_values, prices, strict=False)))
            sum_x2 = sum((x * x for x in x_values))
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            avg_price = sum_y / n
            normalized_slope = slope / avg_price if avg_price > 0 else 0.0
            return normalized_slope * 252
        except COMMON_EXC:
            return 0.0

    def _calculate_rolling_volatility(self, prices: list[float]) -> float:
        """Calculate rolling volatility from prices."""
        try:
            if len(prices) < 2:
                return 0.0
            returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
            recent_returns = returns[-self.volatility_window:] if len(returns) >= self.volatility_window else returns
            if len(recent_returns) < 2:
                return 0.0
            return statistics.stdev(recent_returns) * math.sqrt(252)
        except COMMON_EXC:
            return 0.0

    def _calculate_volatility_percentile(self, current_vol: float, prices: list[float]) -> float:
        """Calculate volatility percentile rank."""
        try:
            if len(prices) < self.lookback_days:
                return 0.5
            historical_vols = []
            for i in range(self.volatility_window, len(prices)):
                window_prices = prices[i - self.volatility_window:i]
                vol = self._calculate_rolling_volatility(window_prices)
                historical_vols.append(vol)
            if not historical_vols:
                return 0.5
            return self._get_percentile_rank(current_vol, historical_vols)
        except COMMON_EXC:
            return 0.5

    def _calculate_correlation(self, returns1: list[float], returns2: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        try:
            if len(returns1) != len(returns2) or len(returns1) < 2:
                return 0.0
            n = len(returns1)
            sum1 = sum(returns1)
            sum2 = sum(returns2)
            sum1_sq = sum((x * x for x in returns1))
            sum2_sq = sum((x * x for x in returns2))
            sum_products = sum((x * y for x, y in zip(returns1, returns2, strict=False)))
            numerator = n * sum_products - sum1 * sum2
            denominator = math.sqrt((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2))
            if denominator == 0:
                return 0.0
            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))
        except COMMON_EXC:
            return 0.0

    def _get_percentile_rank(self, value: float, data: list[float]) -> float:
        """Calculate percentile rank of value in data."""
        if not data:
            return 0.5
        sorted_data = sorted(data)
        below_count = sum((1 for x in sorted_data if x < value))
        at_count = sum((1 for x in sorted_data if x == value))
        return (below_count + 0.5 * at_count) / len(sorted_data)

class AdaptivePositionSizer:
    """
    Advanced adaptive position sizing engine with market condition awareness.

    Combines multiple position sizing methodologies and adapts to market conditions
    for optimal risk-adjusted position sizing in institutional trading environments.
    """

    def __init__(self, risk_level: RiskLevel=RiskLevel.MODERATE):
        """Initialize adaptive position sizer."""
        self.risk_level = risk_level
        self.market_analyzer = MarketConditionAnalyzer()
        self.dynamic_sizer = DynamicPositionSizer(risk_level)
        self.kelly_calculator = KellyCalculator()
        self.regime_multipliers = {MarketRegime.BULL_TRENDING: 1.3, MarketRegime.BEAR_TRENDING: 0.5, MarketRegime.SIDEWAYS_RANGE: 0.9, MarketRegime.HIGH_VOLATILITY: 0.4, MarketRegime.LOW_VOLATILITY: 1.2, MarketRegime.CRISIS: 0.15, MarketRegime.NORMAL: 1.0}
        self.volatility_adjustments = {VolatilityRegime.EXTREMELY_LOW: 1.4, VolatilityRegime.LOW: 1.15, VolatilityRegime.NORMAL: 1.0, VolatilityRegime.HIGH: 0.65, VolatilityRegime.EXTREMELY_HIGH: 0.3}
        logger.info(f'AdaptivePositionSizer initialized with risk_level={risk_level}')

    def calculate_adaptive_position(self, symbol: str, account_equity: float, entry_price: float, market_data: dict, portfolio_data: dict) -> dict[str, Any]:
        """
        Calculate adaptive position size based on market conditions.

        Args:
            symbol: Trading symbol
            account_equity: Current account equity
            entry_price: Planned entry price
            market_data: Current market data and indicators
            portfolio_data: Portfolio-level data including returns and correlations

        Returns:
            Comprehensive position sizing recommendation
        """
        try:
            result = {'symbol': symbol, 'recommended_size': 0, 'base_calculation': {}, 'market_adjustments': {}, 'final_multiplier': 1.0, 'risk_assessment': {}, 'warnings': [], 'metadata': {}}
            base_result = self.dynamic_sizer.calculate_optimal_position(symbol, account_equity, entry_price, market_data, portfolio_data)
            result['base_calculation'] = base_result
            base_size = base_result.get('recommended_size', 0)
            if base_size <= 0:
                result['warnings'].extend(base_result.get('warnings', []))
                return result
            price_data = portfolio_data.get('price_data', {})
            returns_data = portfolio_data.get('returns_data', {})
            market_regime = self.market_analyzer.analyze_market_regime(price_data)
            regime_multiplier = self.regime_multipliers.get(market_regime, 1.0)
            result['market_adjustments']['market_regime'] = market_regime.value
            result['market_adjustments']['regime_multiplier'] = regime_multiplier
            symbol_returns = returns_data.get(symbol, [])
            volatility_regime = self.market_analyzer.assess_volatility_regime(symbol_returns)
            vol_multiplier = self.volatility_adjustments.get(volatility_regime, 1.0)
            result['market_adjustments']['volatility_regime'] = volatility_regime.value
            result['market_adjustments']['volatility_multiplier'] = vol_multiplier
            correlation_matrix = self.market_analyzer.calculate_correlation_matrix(returns_data)
            correlation_penalty = self._calculate_correlation_penalty(symbol, correlation_matrix, portfolio_data)
            result['market_adjustments']['correlation_penalty'] = correlation_penalty
            final_multiplier = regime_multiplier * vol_multiplier * (1 - correlation_penalty)
            final_multiplier = max(0.1, min(2.0, final_multiplier))
            result['final_multiplier'] = final_multiplier
            result['recommended_size'] = int(base_size * final_multiplier)
            result['risk_assessment'] = self._assess_position_risk(result['recommended_size'], entry_price, account_equity, market_data)
            self._generate_sizing_warnings(result, market_regime, volatility_regime)
            result['metadata'] = {'calculation_timestamp': datetime.now(UTC), 'market_conditions': {'market_regime': market_regime.value, 'volatility_regime': volatility_regime.value, 'correlation_environment': 'high' if correlation_penalty > 0.3 else 'normal'}, 'sizing_rationale': self._generate_sizing_rationale(regime_multiplier, vol_multiplier, correlation_penalty)}
            logger.info(f"Adaptive sizing for {symbol}: base={base_size}, final={result['recommended_size']}, multiplier={final_multiplier:.3f}")
            return result
        except COMMON_EXC as e:
            logger.error(f'Error calculating adaptive position for {symbol}: {e}')
            return {'symbol': symbol, 'recommended_size': 0, 'base_calculation': {}, 'market_adjustments': {}, 'final_multiplier': 0.0, 'risk_assessment': {}, 'warnings': [f'Adaptive sizing error: {e}'], 'metadata': {'error': str(e)}}

    def get_regime_based_limits(self, market_regime: MarketRegime) -> dict[str, float]:
        """
        Get position sizing limits based on market regime.

        Args:
            market_regime: Current market regime

        Returns:
            Dictionary of regime-specific limits
        """
        base_limits = {'max_position_pct': RISK_PARAMETERS['MAX_POSITION_SIZE'], 'max_portfolio_risk': RISK_PARAMETERS['MAX_PORTFOLIO_RISK'], 'max_correlation_exposure': RISK_PARAMETERS['MAX_CORRELATION_EXPOSURE']}
        if market_regime == MarketRegime.CRISIS:
            return {'max_position_pct': base_limits['max_position_pct'] * 0.3, 'max_portfolio_risk': base_limits['max_portfolio_risk'] * 0.5, 'max_correlation_exposure': base_limits['max_correlation_exposure'] * 0.5}
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            return {'max_position_pct': base_limits['max_position_pct'] * 0.6, 'max_portfolio_risk': base_limits['max_portfolio_risk'] * 0.7, 'max_correlation_exposure': base_limits['max_correlation_exposure'] * 0.7}
        elif market_regime == MarketRegime.BULL_TRENDING:
            return {'max_position_pct': base_limits['max_position_pct'] * 1.2, 'max_portfolio_risk': base_limits['max_portfolio_risk'] * 1.1, 'max_correlation_exposure': base_limits['max_correlation_exposure'] * 1.2}
        else:
            return base_limits

    def _calculate_correlation_penalty(self, symbol: str, correlation_matrix: dict[str, float], portfolio_data: dict) -> float:
        """Calculate position size penalty based on correlations."""
        try:
            current_positions = portfolio_data.get('current_positions', {})
            if not current_positions or not correlation_matrix:
                return 0.0
            total_correlation_exposure = 0.0
            total_position_value = sum((pos.get('notional_value', 0) for pos in current_positions.values()))
            if total_position_value <= 0:
                return 0.0
            for other_symbol, position_info in current_positions.items():
                if other_symbol == symbol:
                    continue
                correlation_key = f'{symbol}_{other_symbol}'
                correlation = correlation_matrix.get(correlation_key, 0.0)
                position_weight = position_info.get('notional_value', 0) / total_position_value
                correlation_contribution = abs(correlation) * position_weight
                total_correlation_exposure += correlation_contribution
            penalty = min(0.5, total_correlation_exposure)
            return penalty
        except COMMON_EXC as e:
            logger.error(f'Error calculating correlation penalty: {e}')
            return 0.0

    def _assess_position_risk(self, position_size: int, entry_price: float, account_equity: float, market_data: dict) -> dict[str, Any]:
        """Assess risk metrics for the proposed position."""
        try:
            if position_size <= 0 or entry_price <= 0 or account_equity <= 0:
                return {'error': 'Invalid inputs for risk assessment'}
            notional_value = position_size * entry_price
            position_pct = notional_value / account_equity
            atr_value = market_data.get('atr', entry_price * 0.02)
            estimated_daily_risk = atr_value / entry_price * notional_value
            risk_pct_of_account = estimated_daily_risk / account_equity
            return {'notional_value': notional_value, 'position_percentage': position_pct, 'estimated_daily_risk': estimated_daily_risk, 'risk_percentage_of_account': risk_pct_of_account, 'risk_level': 'high' if risk_pct_of_account > 0.03 else 'normal', 'leverage_factor': notional_value / account_equity if account_equity > 0 else 0}
        except COMMON_EXC as e:
            logger.error(f'Error assessing position risk: {e}')
            return {'error': str(e)}

    def _generate_sizing_warnings(self, result: dict, market_regime: MarketRegime, volatility_regime: VolatilityRegime):
        """Generate appropriate warnings based on market conditions."""
        warnings = result.get('warnings', [])
        if market_regime == MarketRegime.CRISIS:
            warnings.append('Crisis regime detected - using conservative sizing')
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            warnings.append('High volatility regime - reduced position sizing')
        if volatility_regime == VolatilityRegime.EXTREMELY_HIGH:
            warnings.append('Extremely high volatility - consider smaller positions')
        elif volatility_regime == VolatilityRegime.EXTREMELY_LOW:
            warnings.append('Extremely low volatility - increased position size allowed')
        correlation_penalty = result.get('market_adjustments', {}).get('correlation_penalty', 0)
        if correlation_penalty > 0.3:
            warnings.append('High correlation with existing positions - size reduced')
        result['warnings'] = warnings

    def _generate_sizing_rationale(self, regime_mult: float, vol_mult: float, correlation_penalty: float) -> str:
        """Generate human-readable rationale for sizing decision."""
        factors = []
        if regime_mult > 1.1:
            factors.append('favorable market regime')
        elif regime_mult < 0.9:
            factors.append('unfavorable market regime')
        if vol_mult > 1.1:
            factors.append('low volatility environment')
        elif vol_mult < 0.9:
            factors.append('high volatility environment')
        if correlation_penalty > 0.2:
            factors.append('high correlation with existing positions')
        if factors:
            return f"Position adjusted for: {', '.join(factors)}"
        else:
            return 'Standard position sizing applied'