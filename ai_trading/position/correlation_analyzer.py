"""
Portfolio Correlation Analyzer for intelligent position management.

Monitors position correlations and portfolio concentrations:
- Real-time correlation analysis between positions
- Sector concentration monitoring
- Risk budget allocation optimization
- Dynamic exposure rebalancing signals

AI-AGENT-REF: Portfolio correlation analysis for risk-aware position management
"""
from ai_trading.logging import get_logger
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TYPE_CHECKING
import numpy as np
from ai_trading.exc import COMMON_EXC
from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:
    import pandas as pd
logger = get_logger(__name__)

class ConcentrationLevel(Enum):
    """Portfolio concentration risk levels."""
    LOW = 'low'
    MODERATE = 'moderate'
    HIGH = 'high'
    EXTREME = 'extreme'

class CorrelationStrength(Enum):
    """Correlation strength levels."""
    VERY_LOW = 'very_low'
    LOW = 'low'
    MODERATE = 'moderate'
    HIGH = 'high'
    VERY_HIGH = 'very_high'

@dataclass
class PositionCorrelation:
    """Correlation between two positions."""
    symbol1: str
    symbol2: str
    correlation: float
    strength: CorrelationStrength
    lookback_days: int
    last_updated: datetime

@dataclass
class SectorExposure:
    """Sector exposure analysis."""
    sector: str
    symbols: list[str]
    total_value: float
    portfolio_percentage: float
    concentration_level: ConcentrationLevel
    avg_correlation: float

@dataclass
class PortfolioAnalysis:
    """Complete portfolio correlation analysis."""
    timestamp: datetime
    total_positions: int
    total_value: float
    position_correlations: list[PositionCorrelation]
    avg_portfolio_correlation: float
    max_correlation: float
    sector_exposures: list[SectorExposure]
    largest_position_pct: float
    concentration_level: ConcentrationLevel
    diversification_ratio: float
    effective_positions: float
    reduce_exposure_symbols: list[str]
    rebalance_recommendations: list[str]

class PortfolioCorrelationAnalyzer:
    """
    Analyze portfolio correlations for intelligent position management.

    Provides:
    - Real-time correlation monitoring between positions
    - Sector concentration analysis
    - Risk budget allocation recommendations
    - Dynamic rebalancing signals
    """

    def __init__(self, ctx=None):
        self.ctx = ctx
        self.logger = get_logger(__name__ + '.PortfolioCorrelationAnalyzer')
        self.correlation_lookback_days = 30
        self.min_data_points = 20
        self.position_concentration_thresholds = {ConcentrationLevel.LOW: 20.0, ConcentrationLevel.MODERATE: 35.0, ConcentrationLevel.HIGH: 50.0}
        self.sector_concentration_thresholds = {ConcentrationLevel.LOW: 25.0, ConcentrationLevel.MODERATE: 40.0, ConcentrationLevel.HIGH: 60.0}
        self.correlation_thresholds = {CorrelationStrength.VERY_LOW: 0.3, CorrelationStrength.LOW: 0.5, CorrelationStrength.MODERATE: 0.7, CorrelationStrength.HIGH: 0.85}
        self.last_analysis: PortfolioAnalysis | None = None
        self.correlation_cache: dict[tuple[str, str], PositionCorrelation] = {}

    def analyze_portfolio(self, positions: list[Any]) -> PortfolioAnalysis:
        """
        Perform comprehensive portfolio correlation analysis.

        Args:
            positions: List of current position objects

        Returns:
            PortfolioAnalysis with complete correlation and concentration metrics
        """
        try:
            if not positions:
                return self._get_empty_analysis()
            position_data = self._extract_position_data(positions)
            if not position_data:
                return self._get_empty_analysis()
            correlations = self._calculate_position_correlations(position_data)
            sector_exposures = self._analyze_sector_exposures(position_data)
            portfolio_metrics = self._calculate_portfolio_metrics(position_data, correlations, sector_exposures)
            recommendations = self._generate_recommendations(position_data, correlations, sector_exposures, portfolio_metrics)
            analysis = PortfolioAnalysis(timestamp=datetime.now(UTC), total_positions=len(position_data), total_value=sum((pos['market_value'] for pos in position_data.values())), position_correlations=correlations, avg_portfolio_correlation=portfolio_metrics.get('avg_correlation', 0.0), max_correlation=portfolio_metrics.get('max_correlation', 0.0), sector_exposures=sector_exposures, largest_position_pct=portfolio_metrics.get('largest_position_pct', 0.0), concentration_level=portfolio_metrics.get('concentration_level', ConcentrationLevel.LOW), diversification_ratio=portfolio_metrics.get('diversification_ratio', 1.0), effective_positions=portfolio_metrics.get('effective_positions', len(position_data)), reduce_exposure_symbols=recommendations.get('reduce_exposure', []), rebalance_recommendations=recommendations.get('rebalance', []))
            self.last_analysis = analysis
            self.logger.info('PORTFOLIO_ANALYSIS | positions=%d avg_corr=%.3f max_corr=%.3f concentration=%s', len(position_data), analysis.avg_portfolio_correlation, analysis.max_correlation, analysis.concentration_level.value)
            return analysis
        except COMMON_EXC as exc:
            self.logger.warning('analyze_portfolio failed: %s', exc)
            return self._get_empty_analysis()

    def get_position_correlation(self, symbol1: str, symbol2: str) -> PositionCorrelation | None:
        """Get correlation between two specific positions."""
        cache_key = tuple(sorted([symbol1, symbol2]))
        return self.correlation_cache.get(cache_key)

    def should_reduce_position(self, symbol: str, current_positions: list[Any]) -> tuple[bool, str]:
        """
        Determine if position should be reduced due to correlation/concentration.

        Returns:
            (should_reduce, reason)
        """
        try:
            if not self.last_analysis:
                self.analyze_portfolio(current_positions)
            if not self.last_analysis:
                return (False, '')
            if symbol in self.last_analysis.reduce_exposure_symbols:
                return (True, 'High correlation/concentration risk')
            symbol_sector = self._get_symbol_sector(symbol)
            for sector_exp in self.last_analysis.sector_exposures:
                if sector_exp.sector == symbol_sector and sector_exp.concentration_level in [ConcentrationLevel.HIGH, ConcentrationLevel.EXTREME]:
                    return (True, f'Sector concentration: {sector_exp.portfolio_percentage:.1f}%')
            if self.last_analysis.largest_position_pct > 40.0:
                position_data = self._extract_position_data(current_positions)
                if symbol in position_data:
                    symbol_pct = position_data[symbol]['market_value'] / self.last_analysis.total_value * 100
                    if symbol_pct > 30.0:
                        return (True, f'Position size: {symbol_pct:.1f}%')
            return (False, '')
        except COMMON_EXC as exc:
            self.logger.warning('should_reduce_position failed for %s: %s', symbol, exc)
            return (False, '')

    def get_correlation_adjustment_factor(self, symbol: str) -> float:
        """
        Get correlation-based adjustment factor for position management.

        Returns factor between 0.5 and 1.5:
        - < 1.0: Reduce position aggressiveness (high correlation)
        - > 1.0: Can be more aggressive (low correlation)
        """
        try:
            if not self.last_analysis:
                return 1.0
            symbol_correlations = [corr for corr in self.last_analysis.position_correlations if symbol in [corr.symbol1, corr.symbol2]]
            if not symbol_correlations:
                return 1.0
            avg_correlation = sum((abs(corr.correlation) for corr in symbol_correlations)) / len(symbol_correlations)
            if avg_correlation > 0.8:
                return 0.6
            elif avg_correlation > 0.6:
                return 0.8
            elif avg_correlation < 0.3:
                return 1.2
            else:
                return 1.0
        except COMMON_EXC:
            return 1.0

    def _extract_position_data(self, positions: list[Any]) -> dict[str, dict]:
        """Extract relevant data from position objects."""
        position_data = {}
        try:
            for pos in positions:
                symbol = getattr(pos, 'symbol', '')
                if not symbol:
                    continue
                qty = int(getattr(pos, 'qty', 0))
                if qty == 0:
                    continue
                market_value = float(getattr(pos, 'market_value', 0))
                if market_value == 0:
                    current_price = self._get_current_price(symbol)
                    if current_price > 0:
                        market_value = abs(qty * current_price)
                position_data[symbol] = {'symbol': symbol, 'qty': qty, 'market_value': abs(market_value), 'sector': self._get_symbol_sector(symbol)}
            return position_data
        except COMMON_EXC as exc:
            self.logger.warning('_extract_position_data failed: %s', exc)
            return {}

    def _calculate_position_correlations(self, position_data: dict[str, dict]) -> list[PositionCorrelation]:
        """Calculate correlations between all position pairs."""
        correlations = []
        symbols = list(position_data.keys())
        try:
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i + 1:]:
                    correlation = self._calculate_pair_correlation(symbol1, symbol2)
                    if correlation is not None:
                        correlations.append(correlation)
                        cache_key = tuple(sorted([symbol1, symbol2]))
                        self.correlation_cache[cache_key] = correlation
            return correlations
        except COMMON_EXC as exc:
            self.logger.warning('_calculate_position_correlations failed: %s', exc)
            return []

    def _calculate_pair_correlation(self, symbol1: str, symbol2: str) -> PositionCorrelation | None:
        """Calculate correlation between two symbols."""
        try:
            data1 = self._get_price_data(symbol1)
            data2 = self._get_price_data(symbol2)
            if data1 is None or data2 is None:
                return None
            if len(data1) < self.min_data_points or len(data2) < self.min_data_points:
                return None
            aligned_data = self._align_price_data(data1, data2)
            if aligned_data is None:
                return None
            returns1, returns2 = aligned_data
            if len(returns1) < self.min_data_points:
                return None
            corr_matrix = np.corrcoef(returns1, returns2)
            correlation = corr_matrix[0, 1] if len(corr_matrix) == 2 else 0.0
            pd = load_pandas()
            if pd.isna(correlation):
                correlation = 0.0
            strength = self._classify_correlation_strength(abs(correlation))
            return PositionCorrelation(symbol1=symbol1, symbol2=symbol2, correlation=correlation, strength=strength, lookback_days=self.correlation_lookback_days, last_updated=datetime.now(UTC))
        except COMMON_EXC as exc:
            self.logger.warning('_calculate_pair_correlation failed for %s/%s: %s', symbol1, symbol2, exc)
            return None

    def _get_price_data(self, symbol: str) -> list[float] | None:
        """Get price return data for correlation calculation."""
        try:
            if self.ctx and hasattr(self.ctx, 'data_fetcher'):
                df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
                if df is not None and (not df.empty) and ('close' in df.columns):
                    recent_data = df.tail(self.correlation_lookback_days + 5)
                    if len(recent_data) >= self.min_data_points:
                        closes = recent_data['close']
                        returns = closes.pct_change().dropna()
                        return returns.tolist()
            return None
        except COMMON_EXC:
            return None

    def _align_price_data(self, data1: list[float], data2: list[float]) -> tuple[list[float], list[float]] | None:
        """Align price data for correlation calculation."""
        try:
            min_length = min(len(data1), len(data2))
            if min_length < self.min_data_points:
                return None
            aligned_data1 = data1[-min_length:]
            aligned_data2 = data2[-min_length:]
            return (aligned_data1, aligned_data2)
        except COMMON_EXC:
            return None

    def _classify_correlation_strength(self, abs_correlation: float) -> CorrelationStrength:
        """Classify correlation strength based on absolute value."""
        if abs_correlation >= self.correlation_thresholds[CorrelationStrength.HIGH]:
            return CorrelationStrength.VERY_HIGH
        elif abs_correlation >= self.correlation_thresholds[CorrelationStrength.MODERATE]:
            return CorrelationStrength.HIGH
        elif abs_correlation >= self.correlation_thresholds[CorrelationStrength.LOW]:
            return CorrelationStrength.MODERATE
        elif abs_correlation >= self.correlation_thresholds[CorrelationStrength.VERY_LOW]:
            return CorrelationStrength.LOW
        else:
            return CorrelationStrength.VERY_LOW

    def _analyze_sector_exposures(self, position_data: dict[str, dict]) -> list[SectorExposure]:
        """Analyze sector concentration in portfolio."""
        try:
            sector_groups = defaultdict(list)
            total_value = sum((pos['market_value'] for pos in position_data.values()))
            for _symbol, pos_data in position_data.items():
                sector = pos_data['sector']
                sector_groups[sector].append(pos_data)
            sector_exposures = []
            for sector, positions in sector_groups.items():
                sector_value = sum((pos['market_value'] for pos in positions))
                sector_pct = sector_value / total_value * 100 if total_value > 0 else 0
                concentration_level = self._classify_sector_concentration(sector_pct)
                symbols = [pos['symbol'] for pos in positions]
                avg_correlation = self._calculate_sector_correlation(symbols)
                exposure = SectorExposure(sector=sector, symbols=symbols, total_value=sector_value, portfolio_percentage=sector_pct, concentration_level=concentration_level, avg_correlation=avg_correlation)
                sector_exposures.append(exposure)
            sector_exposures.sort(key=lambda x: x.portfolio_percentage, reverse=True)
            return sector_exposures
        except COMMON_EXC as exc:
            self.logger.warning('_analyze_sector_exposures failed: %s', exc)
            return []

    def _classify_sector_concentration(self, sector_pct: float) -> ConcentrationLevel:
        """Classify sector concentration level."""
        if sector_pct >= self.sector_concentration_thresholds[ConcentrationLevel.HIGH]:
            return ConcentrationLevel.EXTREME
        elif sector_pct >= self.sector_concentration_thresholds[ConcentrationLevel.MODERATE]:
            return ConcentrationLevel.HIGH
        elif sector_pct >= self.sector_concentration_thresholds[ConcentrationLevel.LOW]:
            return ConcentrationLevel.MODERATE
        else:
            return ConcentrationLevel.LOW

    def _calculate_sector_correlation(self, symbols: list[str]) -> float:
        """Calculate average correlation within a sector."""
        try:
            if len(symbols) < 2:
                return 0.0
            correlations = []
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i + 1:]:
                    cache_key = tuple(sorted([symbol1, symbol2]))
                    if cache_key in self.correlation_cache:
                        corr = self.correlation_cache[cache_key].correlation
                        correlations.append(abs(corr))
            return sum(correlations) / len(correlations) if correlations else 0.0
        except COMMON_EXC:
            return 0.0

    def _calculate_portfolio_metrics(self, position_data: dict, correlations: list[PositionCorrelation], sector_exposures: list[SectorExposure]) -> dict[str, Any]:
        """Calculate portfolio-level risk metrics."""
        try:
            total_value = sum((pos['market_value'] for pos in position_data.values()))
            avg_correlation = 0.0
            max_correlation = 0.0
            if correlations:
                avg_correlation = sum((abs(corr.correlation) for corr in correlations)) / len(correlations)
                max_correlation = max((abs(corr.correlation) for corr in correlations))
            if total_value > 0:
                largest_position_pct = max((pos['market_value'] / total_value * 100 for pos in position_data.values()))
            else:
                largest_position_pct = 0.0
            concentration_level = self._classify_position_concentration(largest_position_pct)
            diversification_ratio = min(1.0, 0.5 + avg_correlation * 0.5)
            effective_positions = len(position_data) * (1.0 - avg_correlation * 0.5)
            return {'avg_correlation': avg_correlation, 'max_correlation': max_correlation, 'largest_position_pct': largest_position_pct, 'concentration_level': concentration_level, 'diversification_ratio': diversification_ratio, 'effective_positions': effective_positions}
        except COMMON_EXC as exc:
            self.logger.warning('_calculate_portfolio_metrics failed: %s', exc)
            return {}

    def _classify_position_concentration(self, largest_position_pct: float) -> ConcentrationLevel:
        """Classify portfolio concentration based on largest position."""
        if largest_position_pct >= self.position_concentration_thresholds[ConcentrationLevel.HIGH]:
            return ConcentrationLevel.EXTREME
        elif largest_position_pct >= self.position_concentration_thresholds[ConcentrationLevel.MODERATE]:
            return ConcentrationLevel.HIGH
        elif largest_position_pct >= self.position_concentration_thresholds[ConcentrationLevel.LOW]:
            return ConcentrationLevel.MODERATE
        else:
            return ConcentrationLevel.LOW

    def _generate_recommendations(self, position_data: dict, correlations: list[PositionCorrelation], sector_exposures: list[SectorExposure], portfolio_metrics: dict) -> dict[str, list[str]]:
        """Generate portfolio rebalancing recommendations."""
        try:
            reduce_exposure = []
            rebalance = []
            high_correlation_symbols = set()
            for corr in correlations:
                if corr.strength in [CorrelationStrength.HIGH, CorrelationStrength.VERY_HIGH]:
                    high_correlation_symbols.add(corr.symbol1)
                    high_correlation_symbols.add(corr.symbol2)
            reduce_exposure.extend(list(high_correlation_symbols))
            for sector_exp in sector_exposures:
                if sector_exp.concentration_level in [ConcentrationLevel.HIGH, ConcentrationLevel.EXTREME]:
                    sector_positions = [(symbol, position_data[symbol]['market_value']) for symbol in sector_exp.symbols if symbol in position_data]
                    sector_positions.sort(key=lambda x: x[1], reverse=True)
                    for symbol, _ in sector_positions[:2]:
                        if symbol not in reduce_exposure:
                            reduce_exposure.append(symbol)
            if portfolio_metrics.get('largest_position_pct', 0) > 35.0:
                rebalance.append('Consider reducing largest position size')
            if portfolio_metrics.get('avg_correlation', 0) > 0.6:
                rebalance.append('Portfolio shows high correlation - consider diversification')
            if len(sector_exposures) > 0 and sector_exposures[0].portfolio_percentage > 50.0:
                rebalance.append(f'High sector concentration in {sector_exposures[0].sector}')
            return {'reduce_exposure': reduce_exposure[:5], 'rebalance': rebalance}
        except COMMON_EXC as exc:
            self.logger.warning('_generate_recommendations failed: %s', exc)
            return {'reduce_exposure': [], 'rebalance': []}

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified implementation)."""
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        finance_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C']
        healthcare_symbols = ['JNJ', 'PFE', 'UNH', 'ABBV', 'BMY', 'MRK']
        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in finance_symbols:
            return 'Financials'
        elif symbol in healthcare_symbols:
            return 'Healthcare'
        else:
            return 'Other'

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        try:
            if self.ctx and hasattr(self.ctx, 'data_fetcher'):
                df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
                if df is not None and (not df.empty) and ('close' in df.columns):
                    return float(df['close'].iloc[-1])
                df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
                if df is not None and (not df.empty) and ('close' in df.columns):
                    return float(df['close'].iloc[-1])
            return 0.0
        except COMMON_EXC:
            return 0.0

    def _get_empty_analysis(self) -> PortfolioAnalysis:
        """Return empty analysis when no positions or analysis fails."""
        return PortfolioAnalysis(timestamp=datetime.now(UTC), total_positions=0, total_value=0.0, position_correlations=[], avg_portfolio_correlation=0.0, max_correlation=0.0, sector_exposures=[], largest_position_pct=0.0, concentration_level=ConcentrationLevel.LOW, diversification_ratio=1.0, effective_positions=0.0, reduce_exposure_symbols=[], rebalance_recommendations=[])
