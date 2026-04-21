"""
Legacy SQLAlchemy models for non-OMS convenience tables.

These tables are not the authoritative OMS durability schema. They remain
available for legacy tools and reports, but their schema is defined here and
consumed directly by ``ai_trading.database.connection`` so the repo does not
maintain a second, conflicting table definition path.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class Base(DeclarativeBase):
    """Base class for legacy SQLAlchemy models."""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name, None)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, Decimal):
                value = float(value)
            result[column.name] = value
        return result


class Trade(Base):
    """Legacy trade record for non-OMS reporting paths."""

    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(128), primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    order_type: Mapped[str | None] = mapped_column(String(32), nullable=True, default="market")
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    price: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    executed_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True, default="pending")
    created_at: Mapped[Any | None] = mapped_column(String(64), nullable=True, default=lambda: datetime.now(UTC))
    executed_at: Mapped[Any | None] = mapped_column(String(64), nullable=True)
    commission: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    slippage: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    strategy_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    signal_strength: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, default="")
    market_data_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True, default="{}")

    @property
    def gross_pnl(self) -> float:
        if not self.executed_price or self.status != "filled":
            return 0.0
        if self.side == "buy":
            return _as_float(self.quantity) * (_as_float(self.executed_price) - _as_float(self.price))
        return _as_float(self.quantity) * (_as_float(self.price) - _as_float(self.executed_price))

    @property
    def net_pnl(self) -> float:
        return _as_float(self.gross_pnl) - _as_float(self.commission)

    @property
    def notional_value(self) -> float:
        price = self.executed_price or self.price
        return abs(_as_float(self.quantity) * _as_float(price))


class Portfolio(Base):
    """Legacy portfolio position record for non-OMS reporting paths."""

    __tablename__ = "portfolio"

    id: Mapped[str] = mapped_column(String(128), primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    average_cost: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    last_updated: Mapped[Any | None] = mapped_column(String(64), nullable=True, default=lambda: datetime.now(UTC))
    asset_class: Mapped[str | None] = mapped_column(String(32), nullable=True, default="equity")
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    market_value: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    unrealized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    day_change: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    day_change_percent: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)

    @property
    def total_cost(self) -> float:
        return abs(_as_float(self.quantity) * _as_float(self.average_cost))

    @property
    def current_market_value(self) -> float:
        return abs(_as_float(self.quantity) * _as_float(self.current_price))

    @property
    def position_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        return (_as_float(self.current_price) - _as_float(self.average_cost)) * _as_float(self.quantity)

    @property
    def position_pnl_percent(self) -> float:
        avg_cost = _as_float(self.average_cost)
        if avg_cost == 0.0:
            return 0.0
        return (_as_float(self.current_price) - avg_cost) / avg_cost * 100


class RiskMetric(Base):
    """Legacy risk metric record for non-OMS reporting paths."""

    __tablename__ = "risk_metrics"

    id: Mapped[str] = mapped_column(String(128), primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    calculation_date: Mapped[Any | None] = mapped_column(String(64), nullable=True, default=lambda: datetime.now(UTC))
    var_95: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    var_99: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    expected_shortfall: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    sortino_ratio: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    current_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    volatility: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    beta: Mapped[float | None] = mapped_column(Float, nullable=True, default=1.0)
    correlation_spy: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    concentration_risk: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    liquidity_risk: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)

    @property
    def risk_score(self) -> float:
        var_score = min(abs(_as_float(self.var_95)) * 1000, 50)
        drawdown_score = min(abs(_as_float(self.max_drawdown)) * 100, 30)
        vol_score = min(_as_float(self.volatility) * 100, 20)
        return var_score + drawdown_score + vol_score

    @property
    def risk_level(self) -> str:
        score = self.risk_score
        if score < 25:
            return "Low"
        if score < 50:
            return "Medium"
        if score < 75:
            return "High"
        return "Critical"


class PerformanceMetric(Base):
    """Legacy performance metric record for non-OMS reporting paths."""

    __tablename__ = "performance_metrics"

    id: Mapped[str] = mapped_column(String(128), primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    portfolio_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    period_start: Mapped[str | None] = mapped_column(String(64), nullable=True)
    period_end: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    annualized_return: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    benchmark_return: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    alpha: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    tracking_error: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    information_ratio: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    average_win: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    average_loss: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    largest_win: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    largest_loss: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    total_trades: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    winning_trades: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    losing_trades: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)


__all__ = [
    "Base",
    "PerformanceMetric",
    "Portfolio",
    "RiskMetric",
    "Trade",
]
