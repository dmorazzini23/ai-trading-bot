"""Position holding and management logic for reducing churn."""
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from ai_trading.exc import COMMON_EXC
HAS_NUMPY = True
HAS_PANDAS = True

def requires_pandas_position(func):
    """Decorator to ensure pandas is available for position functions."""

    def wrapper(*args, **kwargs):
        if not HAS_PANDAS:
            raise ImportError(f'pandas required for {func.__name__}')
        return func(*args, **kwargs)
    return wrapper
logger = logging.getLogger(__name__)
_position_lock = Lock()

@dataclass
class PositionInfo:
    """Track position details for hold/sell decisions."""
    symbol: str
    qty: int
    entry_price: float
    entry_time: datetime
    current_price: float
    last_update: datetime
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0
    momentum_score: float = 0.0
    volume_factor: float = 1.0
    sector: str = 'Unknown'

class PositionManager:
    """Manages position holding decisions to reduce churn - Enhanced with intelligent strategies."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.logger = logging.getLogger(__name__ + '.PositionManager')
        self.positions: dict[str, PositionInfo] = {}
        self.hold_threshold_pct = 5.0
        self.min_hold_days = 3
        self.momentum_threshold = 0.7
        self.intelligent_manager = None
        self.use_intelligent_system = True
        try:
            import os
            import sys
            position_path = os.path.join(os.path.dirname(__file__), 'ai_trading', 'position')
            if position_path not in sys.path:
                sys.path.insert(0, position_path)
            from intelligent_manager import IntelligentPositionManager
            self.intelligent_manager = IntelligentPositionManager(ctx)
            self.logger.info('INTELLIGENT_POSITION_MANAGER | Initialized advanced position management system')
        except COMMON_EXC as exc:
            self.logger.info('Intelligent position manager not available: %s; using legacy', exc)
            self.use_intelligent_system = False

    def should_hold_position(self, symbol: str, current_position, unrealized_pnl_pct: float, days_held: int) -> bool:
        """Determine if position should be held vs sold - Enhanced with intelligent analysis."""
        try:
            if self.use_intelligent_system and self.intelligent_manager:
                try:
                    current_positions = self._get_current_positions()
                    result = self.intelligent_manager.should_hold_position(symbol, current_position, unrealized_pnl_pct, days_held, current_positions)
                    self.logger.info('INTELLIGENT_HOLD_DECISION | %s hold=%s pnl=%.2f%% days=%d', symbol, result, unrealized_pnl_pct, days_held)
                    return result
                except COMMON_EXC as exc:
                    self.logger.warning('Intelligent system failed for %s, using fallback: %s', symbol, exc)
            return self._legacy_should_hold_position(symbol, current_position, unrealized_pnl_pct, days_held)
        except COMMON_EXC as exc:
            self.logger.warning('should_hold_position failed for %s: %s', symbol, exc)
            return False

    def _legacy_should_hold_position(self, symbol: str, current_position, unrealized_pnl_pct: float, days_held: int) -> bool:
        """Legacy position hold logic (original implementation)."""
        try:
            if unrealized_pnl_pct > self.hold_threshold_pct:
                self.logger.info('POSITION_HOLD_PROFIT | %s unrealized_pnl=%.2f%%', symbol, unrealized_pnl_pct)
                return True
            if days_held < self.min_hold_days:
                self.logger.info('POSITION_HOLD_AGE | %s days_held=%d', symbol, days_held)
                return True
            momentum_score = self.calculate_momentum_score(symbol)
            if momentum_score > self.momentum_threshold:
                self.logger.info('POSITION_HOLD_MOMENTUM | %s momentum=%.2f', symbol, momentum_score)
                return True
            return False
        except COMMON_EXC as exc:
            self.logger.warning('_legacy_should_hold_position failed for %s: %s', symbol, exc)
            return False

    def get_intelligent_recommendations(self, current_positions: list) -> list:
        """Get intelligent position recommendations for all positions."""
        try:
            if self.use_intelligent_system and self.intelligent_manager:
                recommendations = self.intelligent_manager.get_portfolio_recommendations(current_positions)
                self.logger.info('INTELLIGENT_RECOMMENDATIONS | Generated %d recommendations', len(recommendations))
                return recommendations
            else:
                self.logger.info('INTELLIGENT_RECOMMENDATIONS | Intelligent system not available')
                return []
        except COMMON_EXC as exc:
            self.logger.warning('get_intelligent_recommendations failed: %s', exc)
            return []

    def update_intelligent_tracking(self, symbol: str, position_data) -> None:
        """Update intelligent position tracking."""
        try:
            if self.use_intelligent_system and self.intelligent_manager:
                self.intelligent_manager.update_position_tracking(symbol, position_data)
        except COMMON_EXC as exc:
            self.logger.warning('update_intelligent_tracking failed for %s: %s', symbol, exc)

    def _get_current_positions(self) -> list:
        """Get current positions for portfolio context."""
        try:
            if self.ctx and hasattr(self.ctx, 'api') and hasattr(self.ctx.api, 'list_open_positions'):
                return self.ctx.api.list_open_positions()
            return []
        except COMMON_EXC:
            return []

    def calculate_momentum_score(self, symbol: str) -> float:
        """Calculate momentum score for position hold decision."""
        try:
            df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
            if df is None or df.empty or len(df) < 20:
                return 0.0
            closes = df['close'].tail(20)
            price_momentum = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
            volumes = df['volume'].tail(20)
            volume_momentum = volumes.tail(5).mean() / volumes.head(15).mean() - 1.0
            momentum_score = price_momentum * 0.7 + volume_momentum * 0.3
            momentum_score = max(0.0, min(1.0, momentum_score + 0.5))
            return momentum_score
        except COMMON_EXC as exc:
            self.logger.warning('calculate_momentum_score failed for %s: %s', symbol, exc)
            return 0.0

    def calculate_position_score(self, symbol: str, position_data) -> float:
        """Score existing positions for hold/sell decision."""
        try:
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                return 0.0
            entry_price = float(getattr(position_data, 'avg_entry_price', 0))
            if entry_price <= 0:
                return 0.0
            qty = int(getattr(position_data, 'qty', 0))
            if qty == 0:
                return 0.0
            pnl_pct = (current_price - entry_price) / entry_price * 100
            days_held = self._calculate_days_held(symbol)
            momentum_score = self.calculate_momentum_score(symbol)
            score = 0.0
            if pnl_pct > 5.0:
                score += 0.4 * min(1.0, pnl_pct / 20.0)
            elif pnl_pct < -10.0:
                score -= 0.2
            if days_held < self.min_hold_days:
                score += 0.2
            score += 0.3 * momentum_score
            sector_score = self._get_sector_strength(symbol)
            score += 0.1 * sector_score
            return max(0.0, min(1.0, score))
        except COMMON_EXC as exc:
            self.logger.warning('calculate_position_score failed for %s: %s', symbol, exc)
            return 0.0

    def update_position_tracking(self, symbol: str, position_data) -> None:
        """Update position tracking information."""
        try:
            with _position_lock:
                current_price = self._get_current_price(symbol)
                entry_price = float(getattr(position_data, 'avg_entry_price', 0))
                qty = int(getattr(position_data, 'qty', 0))
                if symbol not in self.positions:
                    self.positions[symbol] = PositionInfo(symbol=symbol, qty=qty, entry_price=entry_price, entry_time=datetime.now(UTC), current_price=current_price, last_update=datetime.now(UTC))
                else:
                    pos = self.positions[symbol]
                    pos.qty = qty
                    pos.current_price = current_price
                    pos.last_update = datetime.now(UTC)
                    if entry_price > 0:
                        pos.unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
                    pos.days_held = (datetime.now(UTC) - pos.entry_time).days
                    pos.momentum_score = self.calculate_momentum_score(symbol)
        except COMMON_EXC as exc:
            self.logger.warning('update_position_tracking failed for %s: %s', symbol, exc)

    def get_hold_signals(self, current_positions: list) -> dict[str, str]:
        """Generate hold signals for existing positions."""
        hold_signals = {}
        try:
            for position in current_positions:
                symbol = getattr(position, 'symbol', '')
                if not symbol:
                    continue
                self.update_position_tracking(symbol, position)
                score = self.calculate_position_score(symbol, position)
                if score >= 0.6:
                    hold_signals[symbol] = 'hold'
                    self.logger.info('POSITION_SIGNAL_HOLD | %s score=%.2f', symbol, score)
                elif score <= 0.3:
                    hold_signals[symbol] = 'sell'
                    self.logger.info('POSITION_SIGNAL_SELL | %s score=%.2f', symbol, score)
                else:
                    hold_signals[symbol] = 'neutral'
        except COMMON_EXC as exc:
            self.logger.warning('get_hold_signals failed: %s', exc)
        return hold_signals

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        try:
            df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
            if df is not None and (not df.empty) and ('close' in df.columns):
                return float(df['close'].iloc[-1])
            df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
            if df is not None and (not df.empty) and ('close' in df.columns):
                return float(df['close'].iloc[-1])
            return 0.0
        except COMMON_EXC:
            return 0.0

    def _calculate_days_held(self, symbol: str) -> int:
        """Calculate number of days position has been held."""
        try:
            with _position_lock:
                if symbol in self.positions:
                    entry_time = self.positions[symbol].entry_time
                    return (datetime.now(UTC) - entry_time).days
            return 0
        except COMMON_EXC:
            return 0

    def _get_sector_strength(self, symbol: str) -> float:
        """Get sector strength score (placeholder for sector rotation logic)."""
        return 0.5

    def cleanup_stale_positions(self) -> None:
        """Clean up position tracking for symbols no longer held."""
        try:
            current_positions = self.ctx.api.list_open_positions()
            current_symbols = {pos.symbol for pos in current_positions}
            with _position_lock:
                stale_symbols = set(self.positions.keys()) - current_symbols
                for symbol in stale_symbols:
                    self.logger.info('POSITION_CLEANUP | removing tracking for %s', symbol)
                    del self.positions[symbol]
        except COMMON_EXC as exc:
            self.logger.warning('cleanup_stale_positions failed: %s', exc)

def should_hold_position(symbol: str, current_position, unrealized_pnl_pct: float, days_held: int) -> bool:
    """Standalone function for position hold decision - used by problem statement example."""
    if unrealized_pnl_pct > 5.0:
        return True
    if days_held < 3:
        return True
    return False

def calculate_position_score(symbol: str, position_data) -> float:
    """Standalone function for position scoring - used by problem statement example."""
    try:
        qty = getattr(position_data, 'qty', 0)
        if qty == 0:
            return 0.0
        score = min(1.0, abs(qty) / 100.0)
        return score
    except COMMON_EXC:
        return 0.0