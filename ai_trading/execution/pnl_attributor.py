"""PnL attribution system to track and explain profit/loss sources.

This module provides detailed tracking and attribution of PnL changes,
including position-based PnL, market movement PnL, and fee/slippage costs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any

from ai_trading.logging import get_logger


def get_phase_logger(name: str, phase: str) -> logging.Logger:
    """Get a logger for a specific phase - fallback implementation."""
    logger_name = f"{name}.{phase}" if phase else name
    return get_logger(logger_name)


class PnLSource(Enum):
    """Sources of PnL attribution."""

    POSITION_CHANGE = "position_change"  # PnL from changing position size
    MARKET_MOVEMENT = "market_movement"  # PnL from market price changes
    FEES = "fees"  # Transaction fees and commissions
    SLIPPAGE = "slippage"  # Execution slippage costs
    DIVIDEND = "dividend"  # Dividend income
    INTEREST = "interest"  # Interest income/costs
    ADJUSTMENT = "adjustment"  # Manual adjustments or corrections
    UNKNOWN = "unknown"  # Unattributed PnL


class PnLEvent:
    """Represents a single PnL event with attribution."""

    def __init__(
        self,
        symbol: str,
        pnl_amount: float,
        source: PnLSource,
        description: str,
        position_qty: float | None = None,
        price: float | None = None,
        correlation_id: str | None = None,
    ):
        self.symbol = symbol
        self.pnl_amount = pnl_amount
        self.source = source
        self.description = description
        self.position_qty = position_qty
        self.price = price
        self.correlation_id = correlation_id
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging and storage."""
        return {
            "symbol": self.symbol,
            "pnl_amount": self.pnl_amount,
            "source": self.source.value,
            "description": self.description,
            "position_qty": self.position_qty,
            "price": self.price,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
        }


class PositionSnapshot:
    """Snapshot of a position at a point in time."""

    def __init__(
        self,
        symbol: str,
        quantity: float,
        avg_cost: float,
        market_price: float,
        market_value: float | None = None,
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.market_price = market_price
        self.market_value = market_value or (quantity * market_price)
        self.cost_basis = quantity * avg_cost
        self.unrealized_pnl = self.market_value - self.cost_basis
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "market_price": self.market_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "timestamp": self.timestamp.isoformat(),
        }


class PnLAttributor:
    """Tracks and attributes PnL changes to their sources."""

    def __init__(self):
        self.logger = get_phase_logger(__name__, "PNL_ATTRIBUTION")
        self._lock = Lock()

        # Track position snapshots by symbol
        self._position_snapshots: dict[str, PositionSnapshot] = {}

        # Track PnL events
        self._pnl_events: list[PnLEvent] = []

        # Track daily PnL by source
        self._daily_pnl: dict[str, dict[PnLSource, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Track cumulative PnL by symbol and source
        self._cumulative_pnl: dict[str, dict[PnLSource, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Track total portfolio PnL
        self._total_realized_pnl = 0.0
        self._total_unrealized_pnl = 0.0

        # Configuration
        self.max_events_history = 5000

    def update_position_snapshot(
        self,
        symbol: str,
        quantity: float,
        avg_cost: float,
        market_price: float,
        correlation_id: str | None = None,
    ) -> None:
        """Update position snapshot and calculate PnL attribution."""
        with self._lock:
            old_snapshot = self._position_snapshots.get(symbol)

            new_snapshot = PositionSnapshot(symbol, quantity, avg_cost, market_price)
            self._position_snapshots[symbol] = new_snapshot

            # Log the position update
            self.logger.info(
                "POSITION_SNAPSHOT_UPDATE",
                extra={
                    "symbol": symbol,
                    "new_snapshot": new_snapshot.to_dict(),
                    "correlation_id": correlation_id,
                },
            )

            # Calculate PnL attribution if we had a previous snapshot
            if old_snapshot:
                self._calculate_pnl_attribution(
                    old_snapshot, new_snapshot, correlation_id
                )

    def _calculate_pnl_attribution(
        self,
        old_snapshot: PositionSnapshot,
        new_snapshot: PositionSnapshot,
        correlation_id: str | None = None,
    ) -> None:
        """Calculate PnL attribution between two position snapshots."""
        symbol = new_snapshot.symbol

        # Calculate position change PnL (realized from position size changes)
        qty_change = new_snapshot.quantity - old_snapshot.quantity
        if abs(qty_change) > 1e-6:  # Position size changed
            # For position size changes, calculate realized PnL
            if qty_change < 0:  # Selling/reducing position
                avg_exit_price = new_snapshot.market_price  # Approximate
                position_change_pnl = abs(qty_change) * (
                    avg_exit_price - old_snapshot.avg_cost
                )

                self._add_pnl_event(
                    symbol=symbol,
                    pnl_amount=position_change_pnl,
                    source=PnLSource.POSITION_CHANGE,
                    description=f"Realized PnL from selling {abs(qty_change)} shares at ~${avg_exit_price:.2f}",
                    position_qty=qty_change,
                    price=avg_exit_price,
                    correlation_id=correlation_id,
                )

        # Calculate market movement PnL (unrealized from price changes)
        price_change = new_snapshot.market_price - old_snapshot.market_price
        if (
            abs(price_change) > 1e-6 and old_snapshot.quantity != 0
        ):  # Price changed and we had position
            # Use the position quantity before any changes for market movement calculation
            base_quantity = old_snapshot.quantity
            market_movement_pnl = base_quantity * price_change

            self._add_pnl_event(
                symbol=symbol,
                pnl_amount=market_movement_pnl,
                source=PnLSource.MARKET_MOVEMENT,
                description=f"Market movement PnL: ${price_change:.2f} price change on {base_quantity} shares",
                position_qty=base_quantity,
                price=price_change,
                correlation_id=correlation_id,
            )

    def add_trade_pnl(
        self,
        symbol: str,
        trade_qty: float,
        execution_price: float,
        avg_cost: float,
        fees: float = 0,
        slippage: float = 0,
        correlation_id: str | None = None,
    ) -> None:
        """Add PnL from a completed trade with detailed attribution."""

        # Calculate realized PnL from the trade
        if trade_qty != 0:
            realized_pnl = trade_qty * (execution_price - avg_cost)

            self._add_pnl_event(
                symbol=symbol,
                pnl_amount=realized_pnl,
                source=PnLSource.POSITION_CHANGE,
                description=f"Trade PnL: {trade_qty} shares @ ${execution_price:.2f} (cost basis ${avg_cost:.2f})",
                position_qty=trade_qty,
                price=execution_price,
                correlation_id=correlation_id,
            )

        # Add fees as separate PnL event
        if fees != 0:
            self._add_pnl_event(
                symbol=symbol,
                pnl_amount=-abs(fees),  # Fees are always negative PnL
                source=PnLSource.FEES,
                description=f"Transaction fees: ${fees:.2f}",
                position_qty=trade_qty,
                price=execution_price,
                correlation_id=correlation_id,
            )

        # Add slippage as separate PnL event
        if slippage != 0:
            self._add_pnl_event(
                symbol=symbol,
                pnl_amount=slippage,  # Can be positive or negative
                source=PnLSource.SLIPPAGE,
                description=f"Execution slippage: ${slippage:.2f}",
                position_qty=trade_qty,
                price=execution_price,
                correlation_id=correlation_id,
            )

    def add_dividend_pnl(
        self,
        symbol: str,
        dividend_amount: float,
        shares: float,
        correlation_id: str | None = None,
    ) -> None:
        """Add dividend income to PnL attribution."""
        total_dividend = dividend_amount * shares

        self._add_pnl_event(
            symbol=symbol,
            pnl_amount=total_dividend,
            source=PnLSource.DIVIDEND,
            description=f"Dividend income: ${dividend_amount:.4f} per share on {shares} shares",
            position_qty=shares,
            price=dividend_amount,
            correlation_id=correlation_id,
        )

    def add_manual_adjustment(
        self,
        symbol: str,
        adjustment_amount: float,
        reason: str,
        correlation_id: str | None = None,
    ) -> None:
        """Add manual PnL adjustment with explanation."""
        self._add_pnl_event(
            symbol=symbol,
            pnl_amount=adjustment_amount,
            source=PnLSource.ADJUSTMENT,
            description=f"Manual adjustment: {reason}",
            correlation_id=correlation_id,
        )

    def _add_pnl_event(
        self,
        symbol: str,
        pnl_amount: float,
        source: PnLSource,
        description: str,
        position_qty: float | None = None,
        price: float | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Add a PnL event and update tracking."""
        event = PnLEvent(
            symbol=symbol,
            pnl_amount=pnl_amount,
            source=source,
            description=description,
            position_qty=position_qty,
            price=price,
            correlation_id=correlation_id,
        )

        with self._lock:
            self._pnl_events.append(event)

            # Update cumulative tracking
            self._cumulative_pnl[symbol][source] += pnl_amount

            # Update daily tracking
            date_key = event.timestamp.date().isoformat()
            self._daily_pnl[date_key][source] += pnl_amount

            # Update total PnL tracking
            if source in [
                PnLSource.POSITION_CHANGE,
                PnLSource.FEES,
                PnLSource.SLIPPAGE,
                PnLSource.DIVIDEND,
                PnLSource.ADJUSTMENT,
            ]:
                self._total_realized_pnl += pnl_amount

            # Bound history size
            if len(self._pnl_events) > self.max_events_history:
                self._pnl_events = self._pnl_events[-self.max_events_history // 2 :]

        # Log the PnL event
        self.logger.info(f"PNL_EVENT_{source.value.upper()}", extra=event.to_dict())

    def get_pnl_by_symbol(
        self, symbol: str, sources: list[PnLSource] | None = None
    ) -> dict[str, float]:
        """Get cumulative PnL for a symbol, optionally filtered by sources."""
        with self._lock:
            symbol_pnl = self._cumulative_pnl[symbol].copy()

            if sources:
                symbol_pnl = {
                    source.value: symbol_pnl.get(source, 0) for source in sources
                }
            else:
                symbol_pnl = {
                    source.value: amount for source, amount in symbol_pnl.items()
                }

            # Add current unrealized PnL if we have position snapshot
            if symbol in self._position_snapshots:
                snapshot = self._position_snapshots[symbol]
                symbol_pnl["unrealized"] = snapshot.unrealized_pnl

            return symbol_pnl

    def get_daily_pnl(self, date: str | None = None) -> dict[str, float]:
        """Get PnL for a specific date, defaults to today."""
        if date is None:
            date = datetime.now(UTC).date().isoformat()

        with self._lock:
            daily_pnl = self._daily_pnl[date].copy()
            return {source.value: amount for source, amount in daily_pnl.items()}

    def get_total_pnl_summary(self) -> dict[str, Any]:
        """Get comprehensive PnL summary."""
        with self._lock:
            # Calculate total unrealized PnL from current positions
            total_unrealized = sum(
                snapshot.unrealized_pnl
                for snapshot in self._position_snapshots.values()
            )

            # Calculate total by source
            total_by_source = defaultdict(float)
            for symbol_pnl in self._cumulative_pnl.values():
                for source, amount in symbol_pnl.items():
                    total_by_source[source] += amount

            # Calculate today's PnL
            today = datetime.now(UTC).date().isoformat()
            today_pnl = self.get_daily_pnl(today)

            return {
                "total_realized_pnl": self._total_realized_pnl,
                "total_unrealized_pnl": total_unrealized,
                "total_pnl": self._total_realized_pnl + total_unrealized,
                "pnl_by_source": {
                    source.value: amount for source, amount in total_by_source.items()
                },
                "today_pnl": today_pnl,
                "positions_count": len(self._position_snapshots),
                "events_count": len(self._pnl_events),
            }

    def get_recent_pnl_events(
        self,
        symbol: str | None = None,
        sources: list[PnLSource] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recent PnL events, optionally filtered."""
        with self._lock:
            events = self._pnl_events[-limit:].copy()

            # Filter by symbol if specified
            if symbol:
                events = [e for e in events if e.symbol == symbol]

            # Filter by sources if specified
            if sources:
                events = [e for e in events if e.source in sources]

            return [event.to_dict() for event in events]

    def explain_pnl_change(
        self, symbol: str, time_window_minutes: int = 60
    ) -> dict[str, Any]:
        """Explain recent PnL changes for a symbol."""
        cutoff_time = datetime.now(UTC).timestamp() - (time_window_minutes * 60)

        with self._lock:
            recent_events = [
                e
                for e in self._pnl_events
                if (e.symbol == symbol and e.timestamp.timestamp() > cutoff_time)
            ]

        if not recent_events:
            return {
                "symbol": symbol,
                "time_window_minutes": time_window_minutes,
                "explanation": "No PnL events in the specified time window",
                "total_change": 0,
                "events": [],
            }

        # Aggregate by source
        pnl_by_source = defaultdict(float)
        for event in recent_events:
            pnl_by_source[event.source] += event.pnl_amount

        total_change = sum(pnl_by_source.values())

        # Generate human-readable explanation
        explanations = []
        for source, amount in pnl_by_source.items():
            if abs(amount) > 0.01:  # Only include significant amounts
                sign = "gained" if amount > 0 else "lost"
                explanations.append(
                    f"{sign} ${abs(amount):.2f} from {source.value.replace('_', ' ')}"
                )

        return {
            "symbol": symbol,
            "time_window_minutes": time_window_minutes,
            "total_change": total_change,
            "pnl_by_source": {
                source.value: amount for source, amount in pnl_by_source.items()
            },
            "explanation": (
                "; ".join(explanations)
                if explanations
                else "No significant PnL changes"
            ),
            "events_count": len(recent_events),
            "events": [event.to_dict() for event in recent_events],
        }

    def get_position_snapshots(self) -> dict[str, dict[str, Any]]:
        """Get current position snapshots."""
        with self._lock:
            return {
                symbol: snapshot.to_dict()
                for symbol, snapshot in self._position_snapshots.items()
            }

    def calculate_attribution_statistics(self) -> dict[str, Any]:
        """Calculate statistics about PnL attribution."""
        with self._lock:
            if not self._pnl_events:
                return {"message": "No PnL events to analyze"}

            # Count events by source
            source_counts = defaultdict(int)
            source_amounts = defaultdict(float)

            for event in self._pnl_events:
                source_counts[event.source] += 1
                source_amounts[event.source] += event.pnl_amount

            # Calculate average PnL per event by source
            avg_pnl_by_source = {
                source.value: source_amounts[source] / source_counts[source]
                for source in source_counts
            }

            # Find biggest winners and losers
            sorted_events = sorted(self._pnl_events, key=lambda e: e.pnl_amount)
            biggest_loss = sorted_events[0] if sorted_events else None
            biggest_gain = sorted_events[-1] if sorted_events else None

            return {
                "total_events": len(self._pnl_events),
                "events_by_source": {
                    source.value: count for source, count in source_counts.items()
                },
                "total_pnl_by_source": {
                    source.value: amount for source, amount in source_amounts.items()
                },
                "avg_pnl_by_source": avg_pnl_by_source,
                "biggest_gain": biggest_gain.to_dict() if biggest_gain else None,
                "biggest_loss": biggest_loss.to_dict() if biggest_loss else None,
                "symbols_tracked": len({e.symbol for e in self._pnl_events}),
            }


# Global PnL attributor instance
_pnl_attributor: PnLAttributor | None = None
_attributor_lock = Lock()


def get_pnl_attributor() -> PnLAttributor:
    """Get or create the global PnL attributor instance."""
    global _pnl_attributor
    with _attributor_lock:
        if _pnl_attributor is None:
            _pnl_attributor = PnLAttributor()
        return _pnl_attributor


def update_position_for_pnl(
    symbol: str,
    quantity: float,
    avg_cost: float,
    market_price: float,
    correlation_id: str | None = None,
) -> None:
    """Update position snapshot for PnL tracking."""
    attributor = get_pnl_attributor()
    attributor.update_position_snapshot(
        symbol, quantity, avg_cost, market_price, correlation_id
    )


def record_trade_pnl(
    symbol: str,
    trade_qty: float,
    execution_price: float,
    avg_cost: float,
    fees: float = 0,
    slippage: float = 0,
    correlation_id: str | None = None,
) -> None:
    """Record PnL from a completed trade."""
    attributor = get_pnl_attributor()
    attributor.add_trade_pnl(
        symbol, trade_qty, execution_price, avg_cost, fees, slippage, correlation_id
    )


def record_dividend_income(
    symbol: str,
    dividend_per_share: float,
    shares: float,
    correlation_id: str | None = None,
) -> None:
    """Record dividend income."""
    attributor = get_pnl_attributor()
    attributor.add_dividend_pnl(symbol, dividend_per_share, shares, correlation_id)


def get_symbol_pnl_breakdown(symbol: str) -> dict[str, float]:
    """Get PnL breakdown for a specific symbol."""
    attributor = get_pnl_attributor()
    return attributor.get_pnl_by_symbol(symbol)


def get_portfolio_pnl_summary() -> dict[str, Any]:
    """Get comprehensive portfolio PnL summary."""
    attributor = get_pnl_attributor()
    return attributor.get_total_pnl_summary()


def explain_recent_pnl_changes(symbol: str, minutes: int = 60) -> dict[str, Any]:
    """Explain recent PnL changes for a symbol."""
    attributor = get_pnl_attributor()
    return attributor.explain_pnl_change(symbol, minutes)


def get_pnl_attribution_stats() -> dict[str, Any]:
    """Get PnL attribution statistics."""
    attributor = get_pnl_attributor()
    return attributor.calculate_attribution_statistics()
