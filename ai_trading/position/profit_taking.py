"""
Multi-Tiered Profit Taking Engine for intelligent position management.

Implements sophisticated profit taking strategies:
- Scale-out system with 25% increments at key levels
- Risk-multiple based exits (25% at 2R, 25% at 3R, 50% trail)
- Technical level-based partial exits at resistance/overbought
- Correlation-adjusted profit taking when similar positions win

AI-AGENT-REF: Advanced profit taking with multi-tiered scale-out strategies
"""
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import pandas as pd
from ai_trading.exc import COMMON_EXC
logger = logging.getLogger(__name__)

class ProfitTakingStrategy(Enum):
    """Profit taking strategy types."""
    RISK_MULTIPLE = 'risk_multiple'
    TECHNICAL_LEVELS = 'technical'
    PERCENTAGE_BASED = 'percentage'
    TIME_BASED = 'time_based'
    CORRELATION_BASED = 'correlation'

@dataclass
class ProfitTarget:
    """Individual profit taking target."""
    level: float
    quantity_pct: float
    strategy: ProfitTakingStrategy
    priority: int
    triggered: bool = False
    trigger_time: datetime | None = None
    reason: str = ''

@dataclass
class ProfitTakingPlan:
    """Complete profit taking plan for a position."""
    symbol: str
    entry_price: float
    current_price: float
    position_size: int
    risk_amount: float
    targets: list[ProfitTarget]
    remaining_quantity: int
    total_profit_realized: float
    created_time: datetime
    last_updated: datetime

class ProfitTakingEngine:
    """
    Manage multi-tiered profit taking for position optimization.

    Implements multiple profit taking strategies:
    - Risk-multiple based (2R, 3R scaling)
    - Technical level-based (resistance, overbought)
    - Percentage-based targets
    - Time-weighted optimization
    - Correlation-adjusted taking
    """

    def __init__(self, ctx=None):
        self.ctx = ctx
        self.logger = logging.getLogger(__name__ + '.ProfitTakingEngine')
        self.default_targets = [{'level': 2.0, 'quantity_pct': 25.0, 'strategy': ProfitTakingStrategy.RISK_MULTIPLE}, {'level': 3.0, 'quantity_pct': 25.0, 'strategy': ProfitTakingStrategy.RISK_MULTIPLE}, {'level': 5.0, 'quantity_pct': 25.0, 'strategy': ProfitTakingStrategy.RISK_MULTIPLE}]
        self.resistance_buffer = 0.5
        self.overbought_threshold = 75
        self.velocity_threshold = 5.0
        self.time_decay_days = 14
        self.correlation_threshold = 0.7
        self.correlation_adjustment = 0.2
        self.profit_plans: dict[str, ProfitTakingPlan] = {}

    def create_profit_plan(self, symbol: str, position_data: Any, entry_price: float, risk_amount: float) -> ProfitTakingPlan:
        """
        Create comprehensive profit taking plan for new position.

        Args:
            symbol: Symbol for the position
            position_data: Position data object
            entry_price: Entry price for the position
            risk_amount: Initial risk amount (for R multiple calculations)

        Returns:
            ProfitTakingPlan with all target levels
        """
        try:
            position_size = int(getattr(position_data, 'qty', 0))
            current_price = self._get_current_price(symbol)
            if position_size == 0 or current_price <= 0:
                return None
            targets = []
            targets.extend(self._create_risk_multiple_targets(entry_price, risk_amount, position_size))
            technical_targets = self._create_technical_targets(symbol, current_price, position_size)
            targets.extend(technical_targets)
            time_targets = self._create_time_based_targets(entry_price, current_price, position_size)
            targets.extend(time_targets)
            targets.sort(key=lambda t: (t.priority, t.level))
            plan = ProfitTakingPlan(symbol=symbol, entry_price=entry_price, current_price=current_price, position_size=position_size, risk_amount=risk_amount, targets=targets, remaining_quantity=position_size, total_profit_realized=0.0, created_time=datetime.now(UTC), last_updated=datetime.now(UTC))
            self.profit_plans[symbol] = plan
            self.logger.info('PROFIT_PLAN_CREATED | %s entry=%.2f size=%d targets=%d', symbol, entry_price, position_size, len(targets))
            return plan
        except COMMON_EXC as exc:
            self.logger.warning('create_profit_plan failed for %s: %s', symbol, exc)
            return None

    def update_profit_plan(self, symbol: str, current_price: float, position_data: Any=None) -> list[ProfitTarget]:
        """
        Update profit plan and return triggered targets.

        Args:
            symbol: Symbol to update
            current_price: Current market price
            position_data: Current position data

        Returns:
            List of triggered profit targets
        """
        try:
            if symbol not in self.profit_plans:
                return []
            plan = self.profit_plans[symbol]
            plan.current_price = current_price
            plan.last_updated = datetime.now(UTC)
            if position_data:
                current_qty = int(getattr(position_data, 'qty', 0))
                plan.remaining_quantity = current_qty
            triggered_targets = []
            for target in plan.targets:
                if not target.triggered and self._is_target_triggered(target, plan):
                    target.triggered = True
                    target.trigger_time = datetime.now(UTC)
                    triggered_targets.append(target)
                    self.logger.info('PROFIT_TARGET_TRIGGERED | %s level=%.2f qty_pct=%.1f%% strategy=%s', symbol, target.level, target.quantity_pct, target.strategy.value)
            correlation_targets = self._check_correlation_adjustments(symbol, plan)
            triggered_targets.extend(correlation_targets)
            return triggered_targets
        except COMMON_EXC as exc:
            self.logger.warning('update_profit_plan failed for %s: %s', symbol, exc)
            return []

    def get_profit_plan(self, symbol: str) -> ProfitTakingPlan | None:
        """Get current profit plan for symbol."""
        return self.profit_plans.get(symbol)

    def remove_profit_plan(self, symbol: str) -> None:
        """Remove profit plan for symbol."""
        if symbol in self.profit_plans:
            self.logger.info('PROFIT_PLAN_REMOVED | %s', symbol)
            del self.profit_plans[symbol]

    def calculate_profit_velocity(self, symbol: str) -> float:
        """Calculate profit velocity (gain per day) for position."""
        try:
            if symbol not in self.profit_plans:
                return 0.0
            plan = self.profit_plans[symbol]
            days_held = (datetime.now(UTC) - plan.created_time).days
            if days_held == 0:
                return 0.0
            current_gain_pct = (plan.current_price - plan.entry_price) / plan.entry_price * 100
            velocity = current_gain_pct / days_held
            return velocity
        except COMMON_EXC:
            return 0.0

    def _create_risk_multiple_targets(self, entry_price: float, risk_amount: float, position_size: int) -> list[ProfitTarget]:
        """Create risk-multiple based profit targets."""
        targets = []
        try:
            if risk_amount <= 0:
                return self._create_percentage_targets(entry_price, position_size)
            risk_per_share = risk_amount / position_size
            r_multiples = [{'r': 2.0, 'pct': 25.0, 'priority': 1}, {'r': 3.0, 'pct': 25.0, 'priority': 2}, {'r': 5.0, 'pct': 25.0, 'priority': 3}]
            for rm in r_multiples:
                target_price = entry_price + risk_per_share * rm['r']
                target = ProfitTarget(level=target_price, quantity_pct=rm['pct'], strategy=ProfitTakingStrategy.RISK_MULTIPLE, priority=rm['priority'], reason=f"{rm['r']}R target at ${target_price:.2f}")
                targets.append(target)
        except COMMON_EXC as exc:
            self.logger.warning('_create_risk_multiple_targets failed: %s', exc)
            return self._create_percentage_targets(entry_price, position_size)
        return targets

    def _create_percentage_targets(self, entry_price: float, position_size: int) -> list[ProfitTarget]:
        """Create percentage-based profit targets as fallback."""
        targets = []
        percentage_levels = [{'pct': 5.0, 'qty_pct': 25.0, 'priority': 1}, {'pct': 10.0, 'qty_pct': 25.0, 'priority': 2}, {'pct': 20.0, 'qty_pct': 25.0, 'priority': 3}]
        for level in percentage_levels:
            target_price = entry_price * (1 + level['pct'] / 100)
            target = ProfitTarget(level=target_price, quantity_pct=level['qty_pct'], strategy=ProfitTakingStrategy.PERCENTAGE_BASED, priority=level['priority'], reason=f"{level['pct']}% gain target at ${target_price:.2f}")
            targets.append(target)
        return targets

    def _create_technical_targets(self, symbol: str, current_price: float, position_size: int) -> list[ProfitTarget]:
        """Create technical level-based profit targets."""
        targets = []
        try:
            market_data = self._get_market_data(symbol)
            if market_data is None:
                return targets
            resistance_levels = self._find_resistance_levels(market_data, current_price)
            for i, level in enumerate(resistance_levels[:2]):
                target_price = level * (1 - self.resistance_buffer / 100)
                target = ProfitTarget(level=target_price, quantity_pct=15.0, strategy=ProfitTakingStrategy.TECHNICAL_LEVELS, priority=4 + i, reason=f'Resistance level at ${level:.2f}')
                targets.append(target)
            rsi_target = self._create_rsi_overbought_target(symbol, market_data, position_size)
            if rsi_target:
                targets.append(rsi_target)
        except COMMON_EXC as exc:
            self.logger.warning('_create_technical_targets failed for %s: %s', symbol, exc)
        return targets

    def _create_time_based_targets(self, entry_price: float, current_price: float, position_size: int) -> list[ProfitTarget]:
        """Create time-based profit targets for high-velocity moves."""
        targets = []
        try:
            current_gain_pct = (current_price - entry_price) / entry_price * 100
            if current_gain_pct > 3.0:
                velocity_target = ProfitTarget(level=current_price * 1.02, quantity_pct=20.0, strategy=ProfitTakingStrategy.TIME_BASED, priority=6, reason='Time-decay profit taking')
                targets.append(velocity_target)
        except COMMON_EXC as exc:
            self.logger.warning('_create_time_based_targets failed: %s', exc)
        return targets

    def _is_target_triggered(self, target: ProfitTarget, plan: ProfitTakingPlan) -> bool:
        """Check if a profit target has been triggered."""
        try:
            current_price = plan.current_price
            if target.strategy in (ProfitTakingStrategy.RISK_MULTIPLE, ProfitTakingStrategy.TECHNICAL_LEVELS, ProfitTakingStrategy.PERCENTAGE_BASED):
                return current_price >= target.level
            elif target.strategy == ProfitTakingStrategy.TIME_BASED:
                days_held = (datetime.now(UTC) - plan.created_time).days
                if days_held >= self.time_decay_days:
                    time_factor = min(1.0, days_held / 30.0)
                    adjusted_level = target.level * (1 - time_factor * 0.02)
                    return current_price >= adjusted_level
                else:
                    return current_price >= target.level
            elif target.strategy == ProfitTakingStrategy.CORRELATION_BASED:
                return False
            return False
        except COMMON_EXC:
            return False

    def _check_correlation_adjustments(self, symbol: str, plan: ProfitTakingPlan) -> list[ProfitTarget]:
        """Check for correlation-based profit taking adjustments."""
        triggered_targets = []
        try:
            current_gain_pct = (plan.current_price - plan.entry_price) / plan.entry_price * 100
            if current_gain_pct > 10.0:
                correlation_target = ProfitTarget(level=plan.current_price * 0.98, quantity_pct=15.0, strategy=ProfitTakingStrategy.CORRELATION_BASED, priority=7, reason='Portfolio correlation risk management')
                recent_correlation_targets = [t for t in plan.targets if t.strategy == ProfitTakingStrategy.CORRELATION_BASED and t.triggered and t.trigger_time and ((datetime.now(UTC) - t.trigger_time).days < 5)]
                if not recent_correlation_targets:
                    correlation_target.triggered = True
                    correlation_target.trigger_time = datetime.now(UTC)
                    plan.targets.append(correlation_target)
                    triggered_targets.append(correlation_target)
                    self.logger.info('CORRELATION_TARGET_TRIGGERED | %s gain=%.2f%%', symbol, current_gain_pct)
        except COMMON_EXC as exc:
            self.logger.warning('_check_correlation_adjustments failed for %s: %s', symbol, exc)
        return triggered_targets

    def _find_resistance_levels(self, data: pd.DataFrame, current_price: float) -> list[float]:
        """Find resistance levels from price data."""
        try:
            if 'high' not in data.columns or len(data) < 20:
                return []
            highs = data['high']
            resistance_levels = []
            periods = [20, 50]
            for period in periods:
                if len(highs) >= period:
                    recent_highs = highs.tail(period)
                    max_high = recent_highs.max()
                    if max_high > current_price * 1.01:
                        resistance_levels.append(max_high)
            resistance_levels = sorted(set(resistance_levels))
            return resistance_levels[:3]
        except COMMON_EXC:
            return []

    def _create_rsi_overbought_target(self, symbol: str, data: pd.DataFrame, position_size: int) -> ProfitTarget | None:
        """Create RSI overbought profit target."""
        try:
            if 'close' not in data.columns or len(data) < 20:
                return None
            closes = data['close']
            rsi = self._calculate_rsi(closes, 14)
            if pd.isna(rsi) or rsi < self.overbought_threshold:
                return None
            current_price = closes.iloc[-1]
            target = ProfitTarget(level=current_price * 0.995, quantity_pct=10.0, strategy=ProfitTakingStrategy.TECHNICAL_LEVELS, priority=5, reason=f'RSI overbought at {rsi:.1f}')
            return target
        except COMMON_EXC:
            return None

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

    def _get_market_data(self, symbol: str) -> pd.DataFrame | None:
        """Get market data for profit target analysis."""
        try:
            if self.ctx and hasattr(self.ctx, 'data_fetcher'):
                df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
                if df is not None and (not df.empty):
                    return df
                df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
                if df is not None and (not df.empty):
                    return df
            return None
        except COMMON_EXC:
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int=14) -> float:
        """Calculate RSI indicator."""
        try:
            if len(prices) < period + 1:
                return 50.0
            deltas = prices.diff()
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except COMMON_EXC:
            return 50.0