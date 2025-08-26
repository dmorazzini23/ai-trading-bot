"""
Intelligent Position Manager - Main orchestrator for advanced position management.

Coordinates all position management strategies:
- Dynamic trailing stops
- Multi-tiered profit taking
- Market regime adaptation
- Technical signal integration
- Portfolio correlation optimization

AI-AGENT-REF: Main intelligent position management orchestrator
"""
from ai_trading.logging import get_logger
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from ai_trading.exc import COMMON_EXC
from .correlation_analyzer import PortfolioAnalysis, PortfolioCorrelationAnalyzer
from .market_regime import MarketRegime, MarketRegimeDetector, RegimeMetrics
from .profit_taking import ProfitTakingEngine
from .technical_analyzer import DivergenceType, SignalStrength, TechnicalSignalAnalyzer
from .trailing_stops import TrailingStopManager
logger = get_logger(__name__)

class PositionAction(Enum):
    """Recommended position actions."""
    HOLD = 'hold'
    PARTIAL_SELL = 'partial_sell'
    FULL_SELL = 'full_sell'
    REDUCE_SIZE = 'reduce_size'
    TRAIL_STOP = 'trail_stop'
    NO_ACTION = 'no_action'

@dataclass
class PositionRecommendation:
    """Comprehensive position management recommendation."""
    symbol: str
    action: PositionAction
    confidence: float
    urgency: float
    quantity_to_sell: int | None = None
    percentage_to_sell: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    primary_reason: str = ''
    contributing_factors: list[str] = None
    regime_influence: float = 0.0
    technical_influence: float = 0.0
    profit_taking_influence: float = 0.0
    correlation_influence: float = 0.0
    timestamp: datetime = None

class IntelligentPositionManager:
    """
    Main orchestrator for intelligent position management.

    Integrates multiple analysis components to make optimal position decisions:
    - Market regime detection and adaptation
    - Technical signal analysis
    - Dynamic trailing stops
    - Multi-tiered profit taking
    - Portfolio correlation optimization
    """

    def __init__(self, ctx=None):
        self.ctx = ctx
        self.logger = get_logger(__name__ + '.IntelligentPositionManager')
        self.regime_detector = MarketRegimeDetector(ctx)
        self.technical_analyzer = TechnicalSignalAnalyzer(ctx)
        self.trailing_stop_manager = TrailingStopManager(ctx)
        self.profit_taking_engine = ProfitTakingEngine(ctx)
        self.correlation_analyzer = PortfolioCorrelationAnalyzer(ctx)
        self.confidence_threshold = 0.6
        self.urgency_threshold = 0.7
        self.analysis_weights = {'regime': 0.25, 'technical': 0.3, 'profit_taking': 0.2, 'trailing_stops': 0.15, 'correlation': 0.1}
        self.last_portfolio_analysis: PortfolioAnalysis | None = None
        self.last_regime: RegimeMetrics | None = None

    def analyze_position(self, symbol: str, position_data: Any, current_positions: list[Any]) -> PositionRecommendation:
        """
        Perform comprehensive analysis and generate position recommendation.

        Args:
            symbol: Symbol to analyze
            position_data: Current position data
            current_positions: All current positions for portfolio analysis

        Returns:
            PositionRecommendation with detailed analysis and action
        """
        try:
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                return self._get_default_recommendation(symbol, 'No current price data')
            regime_analysis = self._analyze_market_regime()
            technical_analysis = self._analyze_technical_signals(symbol, position_data)
            profit_analysis = self._analyze_profit_opportunities(symbol, position_data, current_price)
            stop_analysis = self._analyze_trailing_stops(symbol, position_data, current_price)
            correlation_analysis = self._analyze_portfolio_context(symbol, current_positions)
            recommendation = self._generate_integrated_recommendation(symbol, position_data, current_price, regime_analysis, technical_analysis, profit_analysis, stop_analysis, correlation_analysis)
            self.logger.info('POSITION_ANALYSIS | %s action=%s confidence=%.2f urgency=%.2f reason=%s', symbol, recommendation.action.value, recommendation.confidence, recommendation.urgency, recommendation.primary_reason)
            return recommendation
        except COMMON_EXC as exc:
            self.logger.warning('analyze_position failed for %s: %s', symbol, exc)
            return self._get_default_recommendation(symbol, f'Analysis error: {exc}')

    def update_position_tracking(self, symbol: str, position_data: Any) -> None:
        """Update position tracking across all components."""
        try:
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                return
            self.trailing_stop_manager.update_trailing_stop(symbol, position_data, current_price)
            self.profit_taking_engine.update_profit_plan(symbol, current_price, position_data)
        except COMMON_EXC as exc:
            self.logger.warning('update_position_tracking failed for %s: %s', symbol, exc)

    def should_hold_position(self, symbol: str, current_position, unrealized_pnl_pct: float, days_held: int, current_positions: list[Any]=None) -> bool:
        """
        Enhanced position hold decision integrating all analysis components.

        This replaces the simple logic in the original position_manager.py
        """
        try:
            recommendation = self.analyze_position(symbol, current_position, current_positions or [])
            if recommendation.action in [PositionAction.HOLD, PositionAction.TRAIL_STOP]:
                return True
            elif recommendation.action in [PositionAction.FULL_SELL]:
                return False
            elif recommendation.action in [PositionAction.PARTIAL_SELL, PositionAction.REDUCE_SIZE]:
                return recommendation.confidence < 0.8
            else:
                return True
        except COMMON_EXC as exc:
            self.logger.warning('should_hold_position failed for %s: %s', symbol, exc)
            return unrealized_pnl_pct > 5.0 or days_held < 3

    def get_portfolio_recommendations(self, current_positions: list[Any]) -> list[PositionRecommendation]:
        """Get recommendations for entire portfolio."""
        recommendations = []
        try:
            portfolio_analysis = self.correlation_analyzer.analyze_portfolio(current_positions)
            self.last_portfolio_analysis = portfolio_analysis
            for position in current_positions:
                symbol = getattr(position, 'symbol', '')
                if symbol:
                    rec = self.analyze_position(symbol, position, current_positions)
                    recommendations.append(rec)
            recommendations.sort(key=lambda r: (r.urgency, r.confidence), reverse=True)
            return recommendations
        except COMMON_EXC as exc:
            self.logger.warning('get_portfolio_recommendations failed: %s', exc)
            return []

    def _analyze_market_regime(self) -> dict[str, Any]:
        """Analyze current market regime."""
        try:
            regime_metrics = self.regime_detector.detect_regime()
            self.last_regime = regime_metrics
            regime_params = self.regime_detector.get_regime_parameters(regime_metrics.regime)
            return {'regime': regime_metrics.regime, 'confidence': regime_metrics.confidence, 'parameters': regime_params, 'metrics': regime_metrics}
        except COMMON_EXC as exc:
            self.logger.warning('_analyze_market_regime failed: %s', exc)
            return {'regime': MarketRegime.RANGE_BOUND, 'confidence': 0.3, 'parameters': {}, 'metrics': None}

    def _analyze_technical_signals(self, symbol: str, position_data: Any) -> dict[str, Any]:
        """Analyze technical signals for position."""
        try:
            signals = self.technical_analyzer.analyze_signals(symbol, position_data)
            return {'signals': signals, 'hold_strength': signals.hold_recommendation, 'exit_urgency': signals.exit_urgency, 'divergence': signals.divergence_type, 'momentum': signals.momentum_score}
        except COMMON_EXC as exc:
            self.logger.warning('_analyze_technical_signals failed for %s: %s', symbol, exc)
            return {'signals': None, 'hold_strength': SignalStrength.NEUTRAL, 'exit_urgency': 0.0, 'divergence': DivergenceType.NONE, 'momentum': 0.5}

    def _analyze_profit_opportunities(self, symbol: str, position_data: Any, current_price: float) -> dict[str, Any]:
        """Analyze profit taking opportunities."""
        try:
            triggered_targets = self.profit_taking_engine.update_profit_plan(symbol, current_price, position_data)
            profit_plan = self.profit_taking_engine.get_profit_plan(symbol)
            velocity = self.profit_taking_engine.calculate_profit_velocity(symbol)
            return {'triggered_targets': triggered_targets, 'profit_plan': profit_plan, 'velocity': velocity, 'has_targets': len(triggered_targets) > 0}
        except COMMON_EXC as exc:
            self.logger.warning('_analyze_profit_opportunities failed for %s: %s', symbol, exc)
            return {'triggered_targets': [], 'profit_plan': None, 'velocity': 0.0, 'has_targets': False}

    def _analyze_trailing_stops(self, symbol: str, position_data: Any, current_price: float) -> dict[str, Any]:
        """Analyze trailing stop status."""
        try:
            stop_level = self.trailing_stop_manager.update_trailing_stop(symbol, position_data, current_price)
            return {'stop_level': stop_level, 'is_triggered': stop_level.is_triggered if stop_level else False, 'stop_price': stop_level.stop_price if stop_level else 0.0, 'trail_distance': stop_level.trail_distance if stop_level else 0.0}
        except COMMON_EXC as exc:
            self.logger.warning('_analyze_trailing_stops failed for %s: %s', symbol, exc)
            return {'stop_level': None, 'is_triggered': False, 'stop_price': 0.0, 'trail_distance': 0.0}

    def _analyze_portfolio_context(self, symbol: str, current_positions: list[Any]) -> dict[str, Any]:
        """Analyze portfolio context for position."""
        try:
            if self.last_portfolio_analysis and (datetime.now(UTC) - self.last_portfolio_analysis.timestamp).seconds < 300:
                portfolio_analysis = self.last_portfolio_analysis
            else:
                portfolio_analysis = self.correlation_analyzer.analyze_portfolio(current_positions)
                self.last_portfolio_analysis = portfolio_analysis
            should_reduce, reason = self.correlation_analyzer.should_reduce_position(symbol, current_positions)
            correlation_factor = self.correlation_analyzer.get_correlation_adjustment_factor(symbol)
            return {'portfolio_analysis': portfolio_analysis, 'should_reduce': should_reduce, 'reduce_reason': reason, 'correlation_factor': correlation_factor}
        except COMMON_EXC as exc:
            self.logger.warning('_analyze_portfolio_context failed for %s: %s', symbol, exc)
            return {'portfolio_analysis': None, 'should_reduce': False, 'reduce_reason': '', 'correlation_factor': 1.0}

    def _generate_integrated_recommendation(self, symbol: str, position_data: Any, current_price: float, regime_analysis: dict, technical_analysis: dict, profit_analysis: dict, stop_analysis: dict, correlation_analysis: dict) -> PositionRecommendation:
        """Generate integrated recommendation from all analyses."""
        try:
            hold_score = 0.0
            sell_score = 0.0
            partial_sell_score = 0.0
            contributing_factors = []
            regime = regime_analysis.get('regime', MarketRegime.RANGE_BOUND)
            regime_confidence = regime_analysis.get('confidence', 0.5)
            regime_params = regime_analysis.get('parameters', {})
            patience_multiplier = regime_params.get('profit_taking_patience', 1.0)
            if patience_multiplier > 1.2:
                hold_score += 0.3 * regime_confidence
                contributing_factors.append(f'Trending market favors holding ({regime.value})')
            elif patience_multiplier < 0.8:
                sell_score += 0.2 * regime_confidence
                contributing_factors.append(f'Range-bound market favors quick profits ({regime.value})')
            technical_signals = technical_analysis.get('signals')
            if technical_signals:
                hold_strength = technical_signals.hold_recommendation
                exit_urgency = technical_signals.exit_urgency
                divergence = technical_signals.divergence_type
                if hold_strength == SignalStrength.VERY_STRONG:
                    hold_score += 0.4
                    contributing_factors.append('Very strong technical signals')
                elif hold_strength == SignalStrength.STRONG:
                    hold_score += 0.2
                    contributing_factors.append('Strong technical signals')
                elif hold_strength == SignalStrength.WEAK:
                    sell_score += 0.2
                    contributing_factors.append('Weak technical signals')
                elif hold_strength == SignalStrength.VERY_WEAK:
                    sell_score += 0.4
                    contributing_factors.append('Very weak technical signals')
                if exit_urgency > 0.7:
                    sell_score += 0.3
                    contributing_factors.append(f'High exit urgency ({exit_urgency:.2f})')
                if divergence == DivergenceType.BEARISH:
                    sell_score += 0.2
                    contributing_factors.append('Bearish momentum divergence')
            triggered_targets = profit_analysis.get('triggered_targets', [])
            if triggered_targets:
                total_pct_to_sell = sum((target.quantity_pct for target in triggered_targets))
                if total_pct_to_sell >= 50:
                    sell_score += 0.3
                    contributing_factors.append(f'Major profit targets triggered ({total_pct_to_sell:.1f}%)')
                else:
                    partial_sell_score += 0.4
                    contributing_factors.append(f'Profit targets triggered ({total_pct_to_sell:.1f}%)')
            if stop_analysis.get('is_triggered', False):
                sell_score += 0.5
                contributing_factors.append('Trailing stop triggered')
            if correlation_analysis.get('should_reduce', False):
                reason = correlation_analysis.get('reduce_reason', '')
                sell_score += 0.2
                contributing_factors.append(f'Portfolio risk: {reason}')
            correlation_factor = correlation_analysis.get('correlation_factor', 1.0)
            if correlation_factor < 0.8:
                sell_score += 0.1
                contributing_factors.append('High portfolio correlation')
            elif correlation_factor > 1.2:
                hold_score += 0.1
                contributing_factors.append('Low portfolio correlation')
            action, confidence, urgency = self._determine_action_from_scores(hold_score, sell_score, partial_sell_score)
            quantity_to_sell, percentage_to_sell, target_price, stop_price = self._calculate_action_details(action, position_data, current_price, triggered_targets, stop_analysis)
            primary_reason = self._determine_primary_reason(action, regime_analysis, technical_analysis, profit_analysis, stop_analysis, correlation_analysis)
            recommendation = PositionRecommendation(symbol=symbol, action=action, confidence=confidence, urgency=urgency, quantity_to_sell=quantity_to_sell, percentage_to_sell=percentage_to_sell, target_price=target_price, stop_price=stop_price, primary_reason=primary_reason, contributing_factors=contributing_factors, regime_influence=hold_score * self.analysis_weights['regime'], technical_influence=max(hold_score, sell_score) * self.analysis_weights['technical'], profit_taking_influence=partial_sell_score * self.analysis_weights['profit_taking'], correlation_influence=correlation_factor, timestamp=datetime.now(UTC))
            return recommendation
        except COMMON_EXC as exc:
            self.logger.warning('_generate_integrated_recommendation failed for %s: %s', symbol, exc)
            return self._get_default_recommendation(symbol, f'Integration error: {exc}')

    def _determine_action_from_scores(self, hold_score: float, sell_score: float, partial_sell_score: float) -> tuple[PositionAction, float, float]:
        """Determine action from component scores."""
        max_score = max(hold_score, sell_score, partial_sell_score)
        if max_score < 0.2:
            return (PositionAction.NO_ACTION, 0.3, 0.0)
        confidence = min(1.0, max_score / 0.6)
        urgency = max(0.0, min(1.0, (max_score - 0.3) / 0.4))
        if sell_score == max_score and sell_score > 0.4:
            return (PositionAction.FULL_SELL, confidence, urgency)
        elif partial_sell_score == max_score and partial_sell_score > 0.3:
            return (PositionAction.PARTIAL_SELL, confidence, urgency)
        elif hold_score == max_score and hold_score > 0.3:
            return (PositionAction.HOLD, confidence, urgency)
        elif sell_score > hold_score and sell_score > 0.2:
            return (PositionAction.REDUCE_SIZE, confidence * 0.8, urgency)
        else:
            return (PositionAction.HOLD, confidence * 0.6, urgency * 0.5)

    def _calculate_action_details(self, action: PositionAction, position_data: Any, current_price: float, triggered_targets: list, stop_analysis: dict) -> tuple[int | None, float | None, float | None, float | None]:
        """Calculate specific details for the recommended action."""
        try:
            qty = int(getattr(position_data, 'qty', 0))
            quantity_to_sell = None
            percentage_to_sell = None
            target_price = None
            stop_price = None
            if action == PositionAction.FULL_SELL:
                quantity_to_sell = qty
                percentage_to_sell = 100.0
                target_price = current_price * 0.995
            elif action == PositionAction.PARTIAL_SELL:
                if triggered_targets:
                    total_pct = sum((target.quantity_pct for target in triggered_targets))
                    percentage_to_sell = min(50.0, total_pct)
                    quantity_to_sell = int(qty * percentage_to_sell / 100)
                    target_price = current_price * 0.998
                else:
                    percentage_to_sell = 25.0
                    quantity_to_sell = int(qty * 0.25)
                    target_price = current_price * 0.998
            elif action == PositionAction.REDUCE_SIZE:
                percentage_to_sell = 20.0
                quantity_to_sell = int(qty * 0.2)
                target_price = current_price * 0.998
            stop_level = stop_analysis.get('stop_level')
            if stop_level:
                stop_price = stop_level.stop_price
            return (quantity_to_sell, percentage_to_sell, target_price, stop_price)
        except COMMON_EXC:
            return (None, None, None, None)

    def _determine_primary_reason(self, action: PositionAction, regime_analysis: dict, technical_analysis: dict, profit_analysis: dict, stop_analysis: dict, correlation_analysis: dict) -> str:
        """Determine the primary reason for the recommendation."""
        if stop_analysis.get('is_triggered', False):
            return 'Trailing stop loss triggered'
        triggered_targets = profit_analysis.get('triggered_targets', [])
        if triggered_targets:
            total_pct = sum((target.quantity_pct for target in triggered_targets))
            return f'Profit targets triggered ({total_pct:.1f}%)'
        technical_signals = technical_analysis.get('signals')
        if technical_signals and technical_signals.exit_urgency > 0.7:
            return f'Technical exit signal (urgency: {technical_signals.exit_urgency:.2f})'
        if technical_signals and technical_signals.divergence_type == DivergenceType.BEARISH:
            return 'Bearish momentum divergence detected'
        if correlation_analysis.get('should_reduce', False):
            return f"Portfolio risk management: {correlation_analysis.get('reduce_reason', 'correlation')}"
        regime = regime_analysis.get('regime', MarketRegime.RANGE_BOUND)
        if action == PositionAction.HOLD:
            return f'Market regime supports holding ({regime.value})'
        elif action in [PositionAction.PARTIAL_SELL, PositionAction.FULL_SELL]:
            return f'Market regime favors profit taking ({regime.value})'
        return 'Comprehensive analysis recommendation'

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

    def _get_default_recommendation(self, symbol: str, reason: str) -> PositionRecommendation:
        """Get default recommendation when analysis fails."""
        return PositionRecommendation(symbol=symbol, action=PositionAction.NO_ACTION, confidence=0.3, urgency=0.0, primary_reason=reason, contributing_factors=[], timestamp=datetime.now(UTC))