"""Repository pattern for database operations."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from .models import (
    MarketDataModel, TradingSignalModel, OrderModel, PositionModel,
    StrategyPerformanceModel, PortfolioSnapshotModel, AlertModel, AuditLogModel
)
from ..core.models import (
    MarketData, TradingSignal, OrderRequest, ExecutionReport,
    TradePosition, StrategyPerformance, PortfolioMetrics, AlertEvent
)
from ..core.exceptions import DatabaseError


class BaseRepository:
    """Base repository with common functionality."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def commit(self) -> None:
        """Commit current transaction."""
        try:
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError("commit", reason=str(e))
    
    def rollback(self) -> None:
        """Rollback current transaction."""
        self.session.rollback()
    
    def refresh(self, instance) -> None:
        """Refresh instance from database."""
        self.session.refresh(instance)


class MarketDataRepository(BaseRepository):
    """Repository for market data operations."""
    
    def save_market_data(self, market_data: MarketData) -> MarketDataModel:
        """Save market data to database."""
        try:
            db_data = MarketDataModel(
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                timeframe="1m",  # Default timeframe
                open_price=market_data.open_price,
                high_price=market_data.high_price,
                low_price=market_data.low_price,
                close_price=market_data.close_price,
                volume=market_data.volume,
                vwap=market_data.vwap,
                bid=market_data.bid,
                ask=market_data.ask,
                spread=market_data.spread,
                source="alpaca"
            )
            
            self.session.add(db_data)
            return db_data
            
        except SQLAlchemyError as e:
            raise DatabaseError("save_market_data", reason=str(e))
    
    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """Get latest price for a symbol."""
        try:
            result = (
                self.session.query(MarketDataModel.close_price)
                .filter(MarketDataModel.symbol == symbol)
                .order_by(desc(MarketDataModel.timestamp))
                .first()
            )
            return result[0] if result else None
            
        except SQLAlchemyError as e:
            raise DatabaseError("get_latest_price", reason=str(e))
    
    def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MarketDataModel]:
        """Get historical market data."""
        try:
            query = (
                self.session.query(MarketDataModel)
                .filter(MarketDataModel.symbol == symbol)
                .filter(MarketDataModel.timestamp >= start_time)
            )
            
            if end_time:
                query = query.filter(MarketDataModel.timestamp <= end_time)
            
            return (
                query.order_by(MarketDataModel.timestamp)
                .limit(limit)
                .all()
            )
            
        except SQLAlchemyError as e:
            raise DatabaseError("get_historical_data", reason=str(e))


class TradingSignalRepository(BaseRepository):
    """Repository for trading signal operations."""
    
    def save_signal(self, signal: TradingSignal) -> TradingSignalModel:
        """Save trading signal to database."""
        try:
            db_signal = TradingSignalModel(
                correlation_id=signal.correlation_id,
                symbol=signal.symbol,
                strategy_id=signal.metadata.get('strategy_id', 'unknown'),
                signal_type=signal.signal_type.value,
                side=signal.side.value,
                strength=signal.strength,
                confidence=signal.confidence,
                timeframe=signal.timeframe.value,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata_=signal.metadata
            )
            
            self.session.add(db_signal)
            return db_signal
            
        except SQLAlchemyError as e:
            raise DatabaseError("save_signal", reason=str(e))
    
    def get_active_signals(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[TradingSignalModel]:
        """Get active trading signals."""
        try:
            query = self.session.query(TradingSignalModel).filter(
                TradingSignalModel.is_active == True
            )
            
            if strategy_id:
                query = query.filter(TradingSignalModel.strategy_id == strategy_id)
            
            if symbol:
                query = query.filter(TradingSignalModel.symbol == symbol)
            
            return query.order_by(desc(TradingSignalModel.created_at)).all()
            
        except SQLAlchemyError as e:
            raise DatabaseError("get_active_signals", reason=str(e))
    
    def deactivate_signal(self, signal_id: UUID) -> bool:
        """Deactivate a trading signal."""
        try:
            rows_updated = (
                self.session.query(TradingSignalModel)
                .filter(TradingSignalModel.id == signal_id)
                .update({
                    'is_active': False,
                    'processed_at': datetime.now(timezone.utc)
                })
            )
            return rows_updated > 0
            
        except SQLAlchemyError as e:
            raise DatabaseError("deactivate_signal", reason=str(e))


class OrderRepository(BaseRepository):
    """Repository for order operations."""
    
    def save_order(
        self,
        order_request: OrderRequest,
        signal_id: Optional[UUID] = None
    ) -> OrderModel:
        """Save order to database."""
        try:
            db_order = OrderModel(
                correlation_id=order_request.correlation_id,
                signal_id=signal_id,
                symbol=order_request.symbol,
                strategy_id=order_request.strategy_id,
                side=order_request.side.value,
                order_type=order_request.order_type.value,
                quantity=order_request.quantity,
                price=order_request.price,
                stop_price=order_request.stop_price,
                metadata_=order_request.metadata,
                submitted_at=datetime.now(timezone.utc)
            )
            
            self.session.add(db_order)
            return db_order
            
        except SQLAlchemyError as e:
            raise DatabaseError("save_order", reason=str(e))
    
    def update_order_execution(
        self,
        order_id: UUID,
        execution_report: ExecutionReport
    ) -> bool:
        """Update order with execution details."""
        try:
            update_data = {
                'status': execution_report.status.value,
                'filled_quantity': execution_report.filled_quantity,
                'average_price': execution_report.average_price,
                'commission': execution_report.commission,
                'executed_at': execution_report.execution_time,
                'broker_order_id': str(execution_report.order_id),
                'venue': execution_report.venue
            }
            
            if execution_report.error_message:
                update_data['error_message'] = execution_report.error_message
            
            rows_updated = (
                self.session.query(OrderModel)
                .filter(OrderModel.id == order_id)
                .update(update_data)
            )
            
            return rows_updated > 0
            
        except SQLAlchemyError as e:
            raise DatabaseError("update_order_execution", reason=str(e))
    
    def get_pending_orders(self, strategy_id: Optional[str] = None) -> List[OrderModel]:
        """Get pending orders."""
        try:
            query = self.session.query(OrderModel).filter(
                OrderModel.status == 'pending'
            )
            
            if strategy_id:
                query = query.filter(OrderModel.strategy_id == strategy_id)
            
            return query.order_by(OrderModel.created_at).all()
            
        except SQLAlchemyError as e:
            raise DatabaseError("get_pending_orders", reason=str(e))


class PositionRepository(BaseRepository):
    """Repository for position operations."""
    
    def save_position(self, position: TradePosition) -> PositionModel:
        """Save position to database."""
        try:
            db_position = PositionModel(
                symbol=position.symbol,
                strategy_id=position.strategy_id,
                asset_class=position.asset_class.value,
                side=position.side.value,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl,
                realized_pnl=position.realized_pnl,
                cost_basis=position.cost_basis,
                market_value=position.market_value,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                max_drawdown=position.max_drawdown,
                max_runup=position.max_runup
            )
            
            self.session.add(db_position)
            return db_position
            
        except SQLAlchemyError as e:
            raise DatabaseError("save_position", reason=str(e))
    
    def update_position(
        self,
        position_id: UUID,
        current_price: Decimal,
        unrealized_pnl: Decimal,
        market_value: Decimal
    ) -> bool:
        """Update position with current values."""
        try:
            rows_updated = (
                self.session.query(PositionModel)
                .filter(PositionModel.id == position_id)
                .update({
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'market_value': market_value,
                    'updated_at': datetime.now(timezone.utc)
                })
            )
            
            return rows_updated > 0
            
        except SQLAlchemyError as e:
            raise DatabaseError("update_position", reason=str(e))
    
    def close_position(
        self,
        position_id: UUID,
        realized_pnl: Decimal
    ) -> bool:
        """Close a position."""
        try:
            rows_updated = (
                self.session.query(PositionModel)
                .filter(PositionModel.id == position_id)
                .update({
                    'is_open': False,
                    'realized_pnl': realized_pnl,
                    'closed_at': datetime.now(timezone.utc)
                })
            )
            
            return rows_updated > 0
            
        except SQLAlchemyError as e:
            raise DatabaseError("close_position", reason=str(e))
    
    def get_open_positions(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[PositionModel]:
        """Get open positions."""
        try:
            query = self.session.query(PositionModel).filter(
                PositionModel.is_open == True
            )
            
            if strategy_id:
                query = query.filter(PositionModel.strategy_id == strategy_id)
            
            if symbol:
                query = query.filter(PositionModel.symbol == symbol)
            
            return query.all()
            
        except SQLAlchemyError as e:
            raise DatabaseError("get_open_positions", reason=str(e))


class PerformanceRepository(BaseRepository):
    """Repository for performance tracking."""
    
    def save_portfolio_snapshot(self, metrics: PortfolioMetrics) -> PortfolioSnapshotModel:
        """Save portfolio snapshot."""
        try:
            db_snapshot = PortfolioSnapshotModel(
                total_value=metrics.total_value,
                cash_balance=metrics.cash_balance,
                equity_value=metrics.equity_value,
                buying_power=metrics.buying_power,
                total_pnl=metrics.total_pnl,
                daily_pnl=metrics.daily_pnl,
                gross_exposure=metrics.gross_exposure,
                net_exposure=metrics.net_exposure,
                leverage=metrics.leverage,
                sharpe_ratio=metrics.sharpe_ratio,
                sortino_ratio=metrics.sortino_ratio,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                var_95=metrics.var_95,
                var_99=metrics.var_99,
                beta=metrics.beta,
                alpha=metrics.alpha
            )
            
            self.session.add(db_snapshot)
            return db_snapshot
            
        except SQLAlchemyError as e:
            raise DatabaseError("save_portfolio_snapshot", reason=str(e))
    
    def update_strategy_performance(
        self,
        strategy_id: str,
        performance: StrategyPerformance
    ) -> StrategyPerformanceModel:
        """Update strategy performance."""
        try:
            # Try to find existing record
            db_performance = (
                self.session.query(StrategyPerformanceModel)
                .filter(StrategyPerformanceModel.strategy_id == strategy_id)
                .first()
            )
            
            if db_performance:
                # Update existing
                db_performance.total_trades = performance.total_trades
                db_performance.winning_trades = performance.winning_trades
                db_performance.total_pnl = performance.total_pnl
                db_performance.daily_pnl = performance.daily_pnl
                db_performance.max_drawdown = performance.max_drawdown
                db_performance.sharpe_ratio = performance.sharpe_ratio
                db_performance.last_trade_time = performance.last_trade_time
                db_performance.updated_at = datetime.now(timezone.utc)
            else:
                # Create new
                db_performance = StrategyPerformanceModel(
                    strategy_id=performance.strategy_id,
                    strategy_type=performance.strategy_type.value,
                    allocation=performance.allocation,
                    total_trades=performance.total_trades,
                    winning_trades=performance.winning_trades,
                    total_pnl=performance.total_pnl,
                    daily_pnl=performance.daily_pnl,
                    max_drawdown=performance.max_drawdown,
                    sharpe_ratio=performance.sharpe_ratio,
                    last_trade_time=performance.last_trade_time
                )
                self.session.add(db_performance)
            
            return db_performance
            
        except SQLAlchemyError as e:
            raise DatabaseError("update_strategy_performance", reason=str(e))
    
    def get_performance_history(
        self,
        days: int = 30,
        snapshot_type: str = 'scheduled'
    ) -> List[PortfolioSnapshotModel]:
        """Get portfolio performance history."""
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            return (
                self.session.query(PortfolioSnapshotModel)
                .filter(PortfolioSnapshotModel.created_at >= start_date)
                .filter(PortfolioSnapshotModel.snapshot_type == snapshot_type)
                .order_by(PortfolioSnapshotModel.created_at)
                .all()
            )
            
        except SQLAlchemyError as e:
            raise DatabaseError("get_performance_history", reason=str(e))


class AlertRepository(BaseRepository):
    """Repository for alert operations."""
    
    def save_alert(self, alert: AlertEvent) -> AlertModel:
        """Save alert to database."""
        try:
            db_alert = AlertModel(
                correlation_id=alert.correlation_id,
                alert_type=alert.alert_type,
                severity=alert.severity.name,
                message=alert.message,
                source=alert.source,
                symbol=alert.symbol,
                strategy_id=alert.strategy_id,
                threshold_value=alert.threshold_value,
                current_value=alert.current_value,
                acknowledged=alert.acknowledged,
                resolved=alert.resolved,
                resolution_note=alert.resolution_note
            )
            
            self.session.add(db_alert)
            return db_alert
            
        except SQLAlchemyError as e:
            raise DatabaseError("save_alert", reason=str(e))
    
    def acknowledge_alert(self, alert_id: UUID, note: Optional[str] = None) -> bool:
        """Acknowledge an alert."""
        try:
            update_data = {
                'acknowledged': True,
                'acknowledged_at': datetime.now(timezone.utc)
            }
            
            if note:
                update_data['resolution_note'] = note
            
            rows_updated = (
                self.session.query(AlertModel)
                .filter(AlertModel.id == alert_id)
                .update(update_data)
            )
            
            return rows_updated > 0
            
        except SQLAlchemyError as e:
            raise DatabaseError("acknowledge_alert", reason=str(e))
    
    def get_unresolved_alerts(
        self,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[AlertModel]:
        """Get unresolved alerts."""
        try:
            query = self.session.query(AlertModel).filter(
                AlertModel.resolved == False
            )
            
            if severity:
                query = query.filter(AlertModel.severity == severity)
            
            return (
                query.order_by(desc(AlertModel.created_at))
                .limit(limit)
                .all()
            )
            
        except SQLAlchemyError as e:
            raise DatabaseError("get_unresolved_alerts", reason=str(e))


class AuditRepository(BaseRepository):
    """Repository for audit logging."""
    
    def log_trade_decision(
        self,
        correlation_id: UUID,
        strategy_id: str,
        symbol: str,
        decision: str,
        reasoning: str,
        metadata: Dict[str, Any]
    ) -> AuditLogModel:
        """Log trade decision for audit."""
        try:
            audit_log = AuditLogModel(
                correlation_id=correlation_id,
                audit_type='trade_decision',
                event_type='decision_made',
                description=f"{decision} decision for {symbol}: {reasoning}",
                strategy_id=strategy_id,
                symbol=symbol,
                metadata_=metadata
            )
            
            self.session.add(audit_log)
            return audit_log
            
        except SQLAlchemyError as e:
            raise DatabaseError("log_trade_decision", reason=str(e))
    
    def log_risk_override(
        self,
        correlation_id: UUID,
        user_id: str,
        override_type: str,
        before_data: Dict[str, Any],
        after_data: Dict[str, Any],
        justification: str
    ) -> AuditLogModel:
        """Log risk override for compliance."""
        try:
            audit_log = AuditLogModel(
                correlation_id=correlation_id,
                audit_type='risk_override',
                event_type=override_type,
                description=f"Risk override: {justification}",
                user_id=user_id,
                before_data=before_data,
                after_data=after_data
            )
            
            self.session.add(audit_log)
            return audit_log
            
        except SQLAlchemyError as e:
            raise DatabaseError("log_risk_override", reason=str(e))