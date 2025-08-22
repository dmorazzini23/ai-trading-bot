"""Position holding and management logic for reducing churn."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

from ai_trading.exc import COMMON_EXC  # AI-AGENT-REF: narrow handler

# AI-AGENT-REF: numpy and pandas are hard dependencies
HAS_NUMPY = True
HAS_PANDAS = True


def requires_pandas_position(func):
    """Decorator to ensure pandas is available for position functions."""

    def wrapper(*args, **kwargs):
        if not HAS_PANDAS:
            raise ImportError(f"pandas required for {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


logger = logging.getLogger(__name__)

# Global position tracking with thread safety
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
    sector: str = "Unknown"


class PositionManager:
    """Manages position holding decisions to reduce churn - Enhanced with intelligent strategies."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.logger = logging.getLogger(__name__ + ".PositionManager")
        self.positions: dict[str, PositionInfo] = {}

        # Legacy parameters (maintained for compatibility)
        self.hold_threshold_pct = 5.0  # Hold positions with >5% gains
        self.min_hold_days = 3  # Hold new positions for at least 3 days
        self.momentum_threshold = 0.7  # Hold high momentum positions

        # AI-AGENT-REF: Initialize intelligent position management system
        self.intelligent_manager = None
        self.use_intelligent_system = True
        try:
            # Import the intelligent manager from the ai_trading package
            import os
            import sys

            # Add ai_trading position module to path
            position_path = os.path.join(
                os.path.dirname(__file__), "ai_trading", "position"
            )
            if position_path not in sys.path:
                sys.path.insert(0, position_path)

            from intelligent_manager import IntelligentPositionManager

            self.intelligent_manager = IntelligentPositionManager(ctx)

            self.logger.info(
                "INTELLIGENT_POSITION_MANAGER | Initialized advanced position management system"
            )

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.info(
                "Intelligent position manager not available: %s; using legacy",
                exc,
            )
            self.use_intelligent_system = False

    def should_hold_position(
        self, symbol: str, current_position, unrealized_pnl_pct: float, days_held: int
    ) -> bool:
        """Determine if position should be held vs sold - Enhanced with intelligent analysis."""
        try:
            # Use intelligent system if available
            if self.use_intelligent_system and self.intelligent_manager:
                try:
                    # Get current positions for portfolio context
                    current_positions = self._get_current_positions()

                    # Use intelligent position manager
                    result = self.intelligent_manager.should_hold_position(
                        symbol,
                        current_position,
                        unrealized_pnl_pct,
                        days_held,
                        current_positions,
                    )

                    self.logger.info(
                        "INTELLIGENT_HOLD_DECISION | %s hold=%s pnl=%.2f%% days=%d",
                        symbol,
                        result,
                        unrealized_pnl_pct,
                        days_held,
                    )
                    return result

                except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
                    self.logger.warning(
                        "Intelligent system failed for %s, using fallback: %s",
                        symbol,
                        exc,
                    )
                    # Fall through to legacy logic

            # Legacy logic (fallback)
            return self._legacy_should_hold_position(
                symbol, current_position, unrealized_pnl_pct, days_held
            )

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning("should_hold_position failed for %s: %s", symbol, exc)
            return False

    def _legacy_should_hold_position(
        self, symbol: str, current_position, unrealized_pnl_pct: float, days_held: int
    ) -> bool:
        """Legacy position hold logic (original implementation)."""
        try:
            # Hold winners with >5% gain
            if unrealized_pnl_pct > self.hold_threshold_pct:
                self.logger.info(
                    "POSITION_HOLD_PROFIT | %s unrealized_pnl=%.2f%%",
                    symbol,
                    unrealized_pnl_pct,
                )
                return True

            # Hold new positions for at least 3 days
            if days_held < self.min_hold_days:
                self.logger.info(
                    "POSITION_HOLD_AGE | %s days_held=%d", symbol, days_held
                )
                return True

            # Hold strong momentum positions
            momentum_score = self.calculate_momentum_score(symbol)
            if momentum_score > self.momentum_threshold:
                self.logger.info(
                    "POSITION_HOLD_MOMENTUM | %s momentum=%.2f", symbol, momentum_score
                )
                return True

            return False

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning(
                "_legacy_should_hold_position failed for %s: %s", symbol, exc
            )
            return False

    def get_intelligent_recommendations(self, current_positions: list) -> list:
        """Get intelligent position recommendations for all positions."""
        try:
            if self.use_intelligent_system and self.intelligent_manager:
                recommendations = (
                    self.intelligent_manager.get_portfolio_recommendations(
                        current_positions
                    )
                )
                self.logger.info(
                    "INTELLIGENT_RECOMMENDATIONS | Generated %d recommendations",
                    len(recommendations),
                )
                return recommendations
            else:
                self.logger.info(
                    "INTELLIGENT_RECOMMENDATIONS | Intelligent system not available"
                )
                return []

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning("get_intelligent_recommendations failed: %s", exc)
            return []

    def update_intelligent_tracking(self, symbol: str, position_data) -> None:
        """Update intelligent position tracking."""
        try:
            if self.use_intelligent_system and self.intelligent_manager:
                self.intelligent_manager.update_position_tracking(symbol, position_data)

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning(
                "update_intelligent_tracking failed for %s: %s", symbol, exc
            )

    def _get_current_positions(self) -> list:
        """Get current positions for portfolio context."""
        try:
            if (
                self.ctx
                and hasattr(self.ctx, "api")
                and hasattr(self.ctx.api, "list_open_positions")
            ):
                return self.ctx.api.list_open_positions()
            return []
        except COMMON_EXC:  # AI-AGENT-REF: narrow
            return []

    def calculate_momentum_score(self, symbol: str) -> float:
        """Calculate momentum score for position hold decision."""
        try:
            # Get minute data for momentum calculation
            df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
            if df is None or df.empty or len(df) < 20:
                return 0.0

            # Calculate price momentum (20-period)
            closes = df["close"].tail(20)
            price_momentum = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]

            # Calculate volume momentum
            volumes = df["volume"].tail(20)
            volume_momentum = (volumes.tail(5).mean() / volumes.head(15).mean()) - 1.0

            # Combine momentum signals
            momentum_score = (price_momentum * 0.7) + (volume_momentum * 0.3)

            # Normalize to 0-1 range
            momentum_score = max(0.0, min(1.0, momentum_score + 0.5))

            return momentum_score

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning(
                "calculate_momentum_score failed for %s: %s", symbol, exc
            )
            return 0.0

    def calculate_position_score(self, symbol: str, position_data) -> float:
        """Score existing positions for hold/sell decision."""
        try:
            # Get current price and calculate P&L
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                return 0.0

            entry_price = float(getattr(position_data, "avg_entry_price", 0))
            if entry_price <= 0:
                return 0.0

            # Calculate unrealized P&L percentage
            qty = int(getattr(position_data, "qty", 0))
            if qty == 0:
                return 0.0

            pnl_pct = ((current_price - entry_price) / entry_price) * 100

            # Calculate position age
            days_held = self._calculate_days_held(symbol)

            # Get momentum score
            momentum_score = self.calculate_momentum_score(symbol)

            # Calculate composite score
            # Higher scores = more likely to hold
            score = 0.0

            # P&L component (40% weight)
            if pnl_pct > 5.0:
                score += 0.4 * min(1.0, pnl_pct / 20.0)  # Cap at 20% gain
            elif pnl_pct < -10.0:
                score -= 0.2  # Penalize large losses

            # Age component (20% weight)
            if days_held < self.min_hold_days:
                score += 0.2

            # Momentum component (30% weight)
            score += 0.3 * momentum_score

            # Sector rotation component (10% weight)
            sector_score = self._get_sector_strength(symbol)
            score += 0.1 * sector_score

            return max(0.0, min(1.0, score))

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning(
                "calculate_position_score failed for %s: %s", symbol, exc
            )
            return 0.0

    def update_position_tracking(self, symbol: str, position_data) -> None:
        """Update position tracking information."""
        try:
            with _position_lock:
                current_price = self._get_current_price(symbol)
                entry_price = float(getattr(position_data, "avg_entry_price", 0))
                qty = int(getattr(position_data, "qty", 0))

                if symbol not in self.positions:
                    # New position
                    self.positions[symbol] = PositionInfo(
                        symbol=symbol,
                        qty=qty,
                        entry_price=entry_price,
                        entry_time=datetime.now(UTC),
                        current_price=current_price,
                        last_update=datetime.now(UTC),
                    )
                else:
                    # Update existing position
                    pos = self.positions[symbol]
                    pos.qty = qty
                    pos.current_price = current_price
                    pos.last_update = datetime.now(UTC)

                    # Calculate metrics
                    if entry_price > 0:
                        pos.unrealized_pnl_pct = (
                            (current_price - entry_price) / entry_price
                        ) * 100
                    pos.days_held = (datetime.now(UTC) - pos.entry_time).days
                    pos.momentum_score = self.calculate_momentum_score(symbol)

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning(
                "update_position_tracking failed for %s: %s", symbol, exc
            )

    def get_hold_signals(self, current_positions: list) -> dict[str, str]:
        """Generate hold signals for existing positions."""
        hold_signals = {}

        try:
            for position in current_positions:
                symbol = getattr(position, "symbol", "")
                if not symbol:
                    continue

                # Update position tracking
                self.update_position_tracking(symbol, position)

                # Calculate position score
                score = self.calculate_position_score(symbol, position)

                # Determine signal
                if score >= 0.6:  # High score = hold
                    hold_signals[symbol] = "hold"
                    self.logger.info(
                        "POSITION_SIGNAL_HOLD | %s score=%.2f", symbol, score
                    )
                elif score <= 0.3:  # Low score = sell
                    hold_signals[symbol] = "sell"
                    self.logger.info(
                        "POSITION_SIGNAL_SELL | %s score=%.2f", symbol, score
                    )
                else:
                    # Neutral - defer to other signals
                    hold_signals[symbol] = "neutral"

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning("get_hold_signals failed: %s", exc)

        return hold_signals

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        try:
            # Try to get latest minute data
            df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
            if df is not None and not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])

            # Fallback to daily data
            df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
            if df is not None and not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])

            return 0.0

        except COMMON_EXC:  # AI-AGENT-REF: narrow
            return 0.0

    def _calculate_days_held(self, symbol: str) -> int:
        """Calculate number of days position has been held."""
        try:
            with _position_lock:
                if symbol in self.positions:
                    entry_time = self.positions[symbol].entry_time
                    return (datetime.now(UTC) - entry_time).days
            return 0
        except COMMON_EXC:  # AI-AGENT-REF: narrow
            return 0

    def _get_sector_strength(self, symbol: str) -> float:
        """Get sector strength score (placeholder for sector rotation logic)."""
        # Placeholder implementation - could be enhanced with sector ETF performance
        return 0.5

    def cleanup_stale_positions(self) -> None:
        """Clean up position tracking for symbols no longer held."""
        try:
            # Get current positions from API
            current_positions = self.ctx.api.list_open_positions()
            current_symbols = {pos.symbol for pos in current_positions}

            with _position_lock:
                # Remove tracking for positions no longer held
                stale_symbols = set(self.positions.keys()) - current_symbols
                for symbol in stale_symbols:
                    self.logger.info(
                        "POSITION_CLEANUP | removing tracking for %s", symbol
                    )
                    del self.positions[symbol]

        except COMMON_EXC as exc:  # AI-AGENT-REF: narrow
            self.logger.warning("cleanup_stale_positions failed: %s", exc)


def should_hold_position(
    symbol: str, current_position, unrealized_pnl_pct: float, days_held: int
) -> bool:
    """Standalone function for position hold decision - used by problem statement example."""
    if unrealized_pnl_pct > 5.0:  # Hold winners with >5% gain
        return True
    if days_held < 3:  # Hold new positions for at least 3 days
        return True

    # For momentum scoring, we'd need context - return conservative default
    return False


def calculate_position_score(symbol: str, position_data) -> float:
    """Standalone function for position scoring - used by problem statement example."""
    try:
        # Simplified scoring without full context
        # In practice, this would need access to market data and context

        # Mock scoring based on available position data
        qty = getattr(position_data, "qty", 0)
        if qty == 0:
            return 0.0

        # Basic score based on position size (larger positions = higher priority)
        score = min(1.0, abs(qty) / 100.0)  # Normalize to typical position size
        return score

    except COMMON_EXC:  # AI-AGENT-REF: narrow
        return 0.0
