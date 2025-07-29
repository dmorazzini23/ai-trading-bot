"""Database models for institutional trading system using SQLAlchemy."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, Numeric, Integer, Boolean, 
    Text, JSON, ForeignKey, Index, CheckConstraint,
    create_engine, MetaData
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.sql import func

from ..core.enums import (
    TradingSide, OrderType, AssetClass, MarketRegime, 
    RiskLevel, StrategyType, TimeFrame, ExecutionStatus
)


Base = declarative_base()
metadata = MetaData()


class TimestampMixin:
    """Mixin for timestamp fields."""
    
    created_at = Column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )


class UUIDMixin:
    """Mixin for UUID primary key."""
    
    id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False
    )


class MarketDataModel(Base, UUIDMixin, TimestampMixin):
    """Market data storage model."""
    
    __tablename__ = 'market_data'
    
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    
    open_price = Column(Numeric(12, 4), nullable=False)
    high_price = Column(Numeric(12, 4), nullable=False)
    low_price = Column(Numeric(12, 4), nullable=False)
    close_price = Column(Numeric(12, 4), nullable=False)
    volume = Column(Integer, nullable=False)
    
    vwap = Column(Numeric(12, 4), nullable=True)
    bid = Column(Numeric(12, 4), nullable=True)
    ask = Column(Numeric(12, 4), nullable=True)
    spread = Column(Numeric(12, 4), nullable=True)
    
    # Metadata
    source = Column(String(50), nullable=False, default='alpaca')
    metadata_ = Column(JSONB, nullable=True, name='metadata')
    
    __table_args__ = (
        Index('ix_market_data_symbol_timestamp', 'symbol', 'timestamp'),
        Index('ix_market_data_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        CheckConstraint('high_price >= low_price', name='chk_high_low_price'),
        CheckConstraint('volume >= 0', name='chk_volume_positive'),
    )


class TradingSignalModel(Base, UUIDMixin, TimestampMixin):
    """Trading signal storage model."""
    
    __tablename__ = 'trading_signals'
    
    correlation_id = Column(PostgresUUID(as_uuid=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    
    signal_type = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    strength = Column(Numeric(5, 4), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    entry_price = Column(Numeric(12, 4), nullable=True)
    stop_loss = Column(Numeric(12, 4), nullable=True)
    take_profit = Column(Numeric(12, 4), nullable=True)
    
    # Signal metadata
    metadata_ = Column(JSONB, nullable=True, name='metadata')
    
    # Status tracking
    is_active = Column(Boolean, default=True, nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        CheckConstraint('strength >= 0 AND strength <= 1', name='chk_strength_range'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='chk_confidence_range'),
        Index('ix_signals_strategy_symbol_active', 'strategy_id', 'symbol', 'is_active'),
    )


class OrderModel(Base, UUIDMixin, TimestampMixin):
    """Order tracking model."""
    
    __tablename__ = 'orders'
    
    correlation_id = Column(PostgresUUID(as_uuid=True), nullable=False, index=True)
    signal_id = Column(
        PostgresUUID(as_uuid=True), 
        ForeignKey('trading_signals.id'),
        nullable=True,
        index=True
    )
    
    symbol = Column(String(20), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    
    side = Column(String(10), nullable=False)
    order_type = Column(String(20), nullable=False)
    quantity = Column(Numeric(12, 6), nullable=False)
    price = Column(Numeric(12, 4), nullable=True)
    stop_price = Column(Numeric(12, 4), nullable=True)
    
    # Execution details
    status = Column(String(20), nullable=False, default='pending')
    filled_quantity = Column(Numeric(12, 6), default=0, nullable=False)
    average_price = Column(Numeric(12, 4), nullable=True)
    commission = Column(Numeric(10, 4), default=0, nullable=False)
    
    # External references
    broker_order_id = Column(String(100), nullable=True, index=True)
    venue = Column(String(50), default='alpaca', nullable=False)
    
    # Timing
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    executed_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata_ = Column(JSONB, nullable=True, name='metadata')
    error_message = Column(Text, nullable=True)
    
    # Relationships
    signal = relationship('TradingSignalModel', backref='orders')
    
    __table_args__ = (
        CheckConstraint('quantity > 0', name='chk_quantity_positive'),
        CheckConstraint('filled_quantity >= 0', name='chk_filled_quantity_positive'),
        CheckConstraint('filled_quantity <= quantity', name='chk_filled_le_quantity'),
        Index('ix_orders_status_created', 'status', 'created_at'),
        Index('ix_orders_strategy_symbol', 'strategy_id', 'symbol'),
    )


class PositionModel(Base, UUIDMixin, TimestampMixin):
    """Position tracking model."""
    
    __tablename__ = 'positions'
    
    symbol = Column(String(20), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    asset_class = Column(String(20), nullable=False)
    
    side = Column(String(10), nullable=False)
    quantity = Column(Numeric(12, 6), nullable=False)
    entry_price = Column(Numeric(12, 4), nullable=False)
    current_price = Column(Numeric(12, 4), nullable=False)
    
    # P&L tracking
    unrealized_pnl = Column(Numeric(12, 2), nullable=False, default=0)
    realized_pnl = Column(Numeric(12, 2), nullable=False, default=0)
    cost_basis = Column(Numeric(12, 2), nullable=False)
    market_value = Column(Numeric(12, 2), nullable=False)
    
    # Risk management
    stop_loss = Column(Numeric(12, 4), nullable=True)
    take_profit = Column(Numeric(12, 4), nullable=True)
    max_drawdown = Column(Numeric(12, 2), default=0, nullable=False)
    max_runup = Column(Numeric(12, 2), default=0, nullable=False)
    
    # Status
    is_open = Column(Boolean, default=True, nullable=False, index=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata_ = Column(JSONB, nullable=True, name='metadata')
    
    __table_args__ = (
        CheckConstraint('quantity != 0', name='chk_quantity_nonzero'),
        CheckConstraint('cost_basis > 0', name='chk_cost_basis_positive'),
        Index('ix_positions_strategy_symbol_open', 'strategy_id', 'symbol', 'is_open'),
        Index('ix_positions_symbol_open', 'symbol', 'is_open'),
    )


class StrategyPerformanceModel(Base, UUIDMixin, TimestampMixin):
    """Strategy performance tracking model."""
    
    __tablename__ = 'strategy_performance'
    
    strategy_id = Column(String(100), nullable=False, index=True)
    strategy_type = Column(String(50), nullable=False)
    
    # Configuration
    allocation = Column(Numeric(5, 4), nullable=False)
    is_enabled = Column(Boolean, default=True, nullable=False)
    
    # Position tracking
    active_positions = Column(Integer, default=0, nullable=False)
    max_positions = Column(Integer, default=10, nullable=False)
    
    # Trade statistics
    total_trades = Column(Integer, default=0, nullable=False)
    winning_trades = Column(Integer, default=0, nullable=False)
    losing_trades = Column(Integer, default=0, nullable=False)
    
    # P&L tracking
    total_pnl = Column(Numeric(12, 2), default=0, nullable=False)
    daily_pnl = Column(Numeric(12, 2), default=0, nullable=False)
    weekly_pnl = Column(Numeric(12, 2), default=0, nullable=False)
    monthly_pnl = Column(Numeric(12, 2), default=0, nullable=False)
    
    # Risk metrics
    max_drawdown = Column(Numeric(12, 2), default=0, nullable=False)
    current_drawdown = Column(Numeric(12, 2), default=0, nullable=False)
    volatility = Column(Numeric(8, 6), nullable=True)
    
    # Performance ratios
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    sortino_ratio = Column(Numeric(8, 4), nullable=True)
    calmar_ratio = Column(Numeric(8, 4), nullable=True)
    profit_factor = Column(Numeric(8, 4), nullable=True)
    
    # Timing
    last_trade_at = Column(DateTime(timezone=True), nullable=True)
    last_signal_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    parameters = Column(JSONB, nullable=True)
    metadata_ = Column(JSONB, nullable=True, name='metadata')
    
    __table_args__ = (
        CheckConstraint('allocation > 0 AND allocation <= 1', name='chk_allocation_range'),
        CheckConstraint('total_trades >= 0', name='chk_total_trades_positive'),
        CheckConstraint('winning_trades >= 0', name='chk_winning_trades_positive'),
        CheckConstraint('losing_trades >= 0', name='chk_losing_trades_positive'),
        Index('ix_strategy_performance_enabled', 'is_enabled'),
    )


class PortfolioSnapshotModel(Base, UUIDMixin, TimestampMixin):
    """Portfolio snapshot for historical tracking."""
    
    __tablename__ = 'portfolio_snapshots'
    
    # Portfolio values
    total_value = Column(Numeric(15, 2), nullable=False)
    cash_balance = Column(Numeric(15, 2), nullable=False)
    equity_value = Column(Numeric(15, 2), nullable=False)
    buying_power = Column(Numeric(15, 2), nullable=False)
    
    # P&L
    total_pnl = Column(Numeric(12, 2), nullable=False)
    daily_pnl = Column(Numeric(12, 2), nullable=False)
    
    # Exposure
    gross_exposure = Column(Numeric(15, 2), nullable=False)
    net_exposure = Column(Numeric(15, 2), nullable=False)
    leverage = Column(Numeric(8, 4), nullable=False)
    
    # Performance metrics
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    sortino_ratio = Column(Numeric(8, 4), nullable=True)
    max_drawdown = Column(Numeric(8, 4), nullable=True)
    win_rate = Column(Numeric(5, 4), nullable=True)
    
    # Risk metrics
    var_95 = Column(Numeric(12, 2), nullable=True)
    var_99 = Column(Numeric(12, 2), nullable=True)
    beta = Column(Numeric(8, 4), nullable=True)
    alpha = Column(Numeric(8, 4), nullable=True)
    
    # Metadata
    snapshot_type = Column(String(20), default='scheduled', nullable=False)
    metadata_ = Column(JSONB, nullable=True, name='metadata')
    
    __table_args__ = (
        CheckConstraint('total_value > 0', name='chk_total_value_positive'),
        CheckConstraint('leverage >= 0', name='chk_leverage_positive'),
        Index('ix_portfolio_snapshots_created', 'created_at'),
        Index('ix_portfolio_snapshots_type_created', 'snapshot_type', 'created_at'),
    )


class AlertModel(Base, UUIDMixin, TimestampMixin):
    """Alert tracking model."""
    
    __tablename__ = 'alerts'
    
    correlation_id = Column(PostgresUUID(as_uuid=True), nullable=False, index=True)
    
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    source = Column(String(50), nullable=False)
    
    # Context
    symbol = Column(String(20), nullable=True, index=True)
    strategy_id = Column(String(100), nullable=True, index=True)
    
    # Values
    threshold_value = Column(Numeric(15, 6), nullable=True)
    current_value = Column(Numeric(15, 6), nullable=True)
    
    # Status
    acknowledged = Column(Boolean, default=False, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_note = Column(Text, nullable=True)
    
    # Metadata
    metadata_ = Column(JSONB, nullable=True, name='metadata')
    
    __table_args__ = (
        Index('ix_alerts_severity_created', 'severity', 'created_at'),
        Index('ix_alerts_type_resolved', 'alert_type', 'resolved'),
        Index('ix_alerts_unresolved', 'resolved', 'created_at'),
    )


class AuditLogModel(Base, UUIDMixin, TimestampMixin):
    """Audit log for regulatory compliance."""
    
    __tablename__ = 'audit_logs'
    
    correlation_id = Column(PostgresUUID(as_uuid=True), nullable=False, index=True)
    
    audit_type = Column(String(50), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=False)
    
    # Context
    user_id = Column(String(100), nullable=True, index=True)
    strategy_id = Column(String(100), nullable=True, index=True)
    symbol = Column(String(20), nullable=True, index=True)
    
    # Data
    before_data = Column(JSONB, nullable=True)
    after_data = Column(JSONB, nullable=True)
    metadata_ = Column(JSONB, nullable=True, name='metadata')
    
    # IP and session tracking
    ip_address = Column(String(45), nullable=True)
    session_id = Column(String(100), nullable=True)
    
    __table_args__ = (
        Index('ix_audit_logs_type_created', 'audit_type', 'created_at'),
        Index('ix_audit_logs_user_created', 'user_id', 'created_at'),
        Index('ix_audit_logs_correlation', 'correlation_id'),
    )


# Database utility functions

def create_database_engine(database_url: str, **kwargs):
    """Create database engine with optimizations."""
    return create_engine(
        database_url,
        pool_size=kwargs.get('pool_size', 20),
        max_overflow=kwargs.get('max_overflow', 30),
        pool_timeout=kwargs.get('pool_timeout', 30),
        pool_recycle=kwargs.get('pool_recycle', 3600),
        echo=kwargs.get('echo', False),
        # PostgreSQL specific optimizations
        connect_args={
            "options": "-c timezone=utc",
            "application_name": "ai_trading_bot"
        }
    )


def create_session_factory(engine):
    """Create session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_db_session(session_factory) -> Session:
    """Get database session."""
    return session_factory()