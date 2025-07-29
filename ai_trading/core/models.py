"""Pydantic models for institutional trading system."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, validator
import numpy as np

from .enums import (
    TradingSide, OrderType, AssetClass, MarketRegime, 
    RiskLevel, StrategyType, TimeFrame, ExecutionStatus
)


class BaseTradeModel(BaseModel):
    """Base model with common trading system fields."""
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
            np.ndarray: lambda v: v.tolist(),
        }
    )
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: UUID = Field(default_factory=uuid4)


class MarketData(BaseTradeModel):
    """Real-time and historical market data."""
    
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    open_price: Decimal = Field(..., gt=0, description="Opening price")
    high_price: Decimal = Field(..., gt=0, description="High price")
    low_price: Decimal = Field(..., gt=0, description="Low price")
    close_price: Decimal = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    vwap: Optional[Decimal] = Field(None, description="Volume weighted average price")
    bid: Optional[Decimal] = Field(None, gt=0, description="Bid price")
    ask: Optional[Decimal] = Field(None, gt=0, description="Ask price")
    spread: Optional[Decimal] = Field(None, ge=0, description="Bid-ask spread")
    
    @validator('high_price')
    def validate_high_price(cls, v, values):
        if 'low_price' in values and v < values['low_price']:
            raise ValueError('High price must be >= low price')
        return v
    
    @validator('spread')
    def validate_spread(cls, v, values):
        if v is not None and 'ask' in values and 'bid' in values:
            if values['ask'] and values['bid']:
                expected_spread = values['ask'] - values['bid']
                if abs(v - expected_spread) > Decimal('0.01'):
                    raise ValueError('Spread must equal ask - bid')
        return v


class TradingSignal(BaseTradeModel):
    """Trading signal from strategy or model."""
    
    symbol: str = Field(..., description="Trading symbol")
    signal_type: StrategyType = Field(..., description="Strategy generating signal")
    side: TradingSide = Field(..., description="Trading direction")
    strength: float = Field(..., ge=0, le=1, description="Signal strength (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence (0-1)")
    timeframe: TimeFrame = Field(..., description="Signal timeframe")
    entry_price: Optional[Decimal] = Field(None, gt=0, description="Suggested entry price")
    stop_loss: Optional[Decimal] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(None, gt=0, description="Take profit price")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional signal data")
    
    @validator('stop_loss')
    def validate_stop_loss(cls, v, values):
        if v is not None and 'entry_price' in values and values['entry_price']:
            side = values.get('side')
            entry = values['entry_price']
            if side == TradingSide.BUY and v >= entry:
                raise ValueError('Stop loss for buy must be below entry price')
            elif side == TradingSide.SELL and v <= entry:
                raise ValueError('Stop loss for sell must be above entry price')
        return v


class OrderRequest(BaseTradeModel):
    """Order request for execution."""
    
    symbol: str = Field(..., description="Trading symbol")
    side: TradingSide = Field(..., description="Trading direction")
    order_type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    price: Optional[Decimal] = Field(None, gt=0, description="Limit price")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop price")
    time_in_force: str = Field(default="DAY", description="Time in force")
    strategy_id: str = Field(..., description="Originating strategy ID")
    parent_order_id: Optional[UUID] = Field(None, description="Parent order reference")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Order metadata")


class ExecutionReport(BaseTradeModel):
    """Trade execution report."""
    
    order_id: UUID = Field(default_factory=uuid4, description="Unique order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: TradingSide = Field(..., description="Trading direction")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    filled_quantity: Decimal = Field(default=Decimal('0'), ge=0, description="Filled quantity")
    average_price: Optional[Decimal] = Field(None, gt=0, description="Average fill price")
    status: ExecutionStatus = Field(..., description="Execution status")
    commission: Decimal = Field(default=Decimal('0'), ge=0, description="Trading commission")
    slippage: Optional[Decimal] = Field(None, description="Price slippage")
    execution_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    venue: str = Field(default="ALPACA", description="Execution venue")
    error_message: Optional[str] = Field(None, description="Error details if failed")


class TradePosition(BaseTradeModel):
    """Active trading position."""
    
    symbol: str = Field(..., description="Trading symbol")
    asset_class: AssetClass = Field(..., description="Asset classification")
    side: TradingSide = Field(..., description="Position side")
    quantity: Decimal = Field(..., description="Position size")
    entry_price: Decimal = Field(..., gt=0, description="Average entry price")
    current_price: Decimal = Field(..., gt=0, description="Current market price")
    unrealized_pnl: Decimal = Field(..., description="Unrealized P&L")
    realized_pnl: Decimal = Field(default=Decimal('0'), description="Realized P&L")
    cost_basis: Decimal = Field(..., gt=0, description="Total cost basis")
    market_value: Decimal = Field(..., description="Current market value")
    stop_loss: Optional[Decimal] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(None, gt=0, description="Take profit price")
    max_drawdown: Decimal = Field(default=Decimal('0'), description="Maximum drawdown")
    max_runup: Decimal = Field(default=Decimal('0'), description="Maximum runup")
    strategy_id: str = Field(..., description="Originating strategy")
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L as percentage of cost basis."""
        if self.cost_basis > 0:
            return float((self.unrealized_pnl / self.cost_basis) * 100)
        return 0.0


class RiskLimits(BaseTradeModel):
    """Risk management limits."""
    
    max_position_size: Decimal = Field(..., gt=0, le=1, description="Max position size (% of portfolio)")
    max_sector_exposure: Decimal = Field(..., gt=0, le=1, description="Max sector exposure")
    max_single_loss: Decimal = Field(..., gt=0, le=1, description="Max single position loss")
    max_daily_loss: Decimal = Field(..., gt=0, le=1, description="Max daily portfolio loss")
    max_drawdown: Decimal = Field(..., gt=0, le=1, description="Max portfolio drawdown")
    var_limit_95: Decimal = Field(..., gt=0, le=1, description="95% VaR limit")
    var_limit_99: Decimal = Field(..., gt=0, le=1, description="99% VaR limit")
    correlation_limit: Decimal = Field(..., gt=0, le=1, description="Max position correlation")
    leverage_limit: Decimal = Field(..., gt=0, description="Maximum portfolio leverage")


class PortfolioMetrics(BaseTradeModel):
    """Portfolio performance metrics."""
    
    total_value: Decimal = Field(..., description="Total portfolio value")
    cash_balance: Decimal = Field(..., description="Available cash")
    equity_value: Decimal = Field(..., ge=0, description="Equity positions value")
    buying_power: Decimal = Field(..., ge=0, description="Available buying power")
    total_pnl: Decimal = Field(..., description="Total unrealized P&L")
    daily_pnl: Decimal = Field(..., description="Daily P&L")
    gross_exposure: Decimal = Field(..., ge=0, description="Gross position exposure")
    net_exposure: Decimal = Field(..., description="Net position exposure")
    leverage: Decimal = Field(default=Decimal('1'), ge=0, description="Portfolio leverage")
    
    # Performance metrics
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")
    win_rate: Optional[float] = Field(None, ge=0, le=1, description="Win rate")
    profit_factor: Optional[float] = Field(None, ge=0, description="Profit factor")
    
    # Risk metrics
    var_95: Optional[Decimal] = Field(None, description="95% Value at Risk")
    var_99: Optional[Decimal] = Field(None, description="99% Value at Risk")
    beta: Optional[float] = Field(None, description="Portfolio beta")
    alpha: Optional[float] = Field(None, description="Portfolio alpha")


class StrategyPerformance(BaseTradeModel):
    """Strategy-specific performance tracking."""
    
    strategy_id: str = Field(..., description="Strategy identifier")
    strategy_type: StrategyType = Field(..., description="Strategy type")
    allocation: Decimal = Field(..., gt=0, le=1, description="Portfolio allocation")
    active_positions: int = Field(default=0, ge=0, description="Number of active positions")
    total_trades: int = Field(default=0, ge=0, description="Total executed trades")
    winning_trades: int = Field(default=0, ge=0, description="Number of winning trades")
    total_pnl: Decimal = Field(default=Decimal('0'), description="Strategy total P&L")
    daily_pnl: Decimal = Field(default=Decimal('0'), description="Strategy daily P&L")
    max_drawdown: Decimal = Field(default=Decimal('0'), description="Strategy max drawdown")
    sharpe_ratio: Optional[float] = Field(None, description="Strategy Sharpe ratio")
    last_trade_time: Optional[datetime] = Field(None, description="Last trade timestamp")
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades > 0:
            return (self.winning_trades / self.total_trades) * 100
        return 0.0


class AlertEvent(BaseTradeModel):
    """System alert event."""
    
    alert_type: str = Field(..., description="Alert category")
    severity: RiskLevel = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    source: str = Field(..., description="Alert source system")
    symbol: Optional[str] = Field(None, description="Related symbol")
    strategy_id: Optional[str] = Field(None, description="Related strategy")
    threshold_value: Optional[Decimal] = Field(None, description="Threshold breached")
    current_value: Optional[Decimal] = Field(None, description="Current value")
    acknowledged: bool = Field(default=False, description="Alert acknowledged")
    resolved: bool = Field(default=False, description="Alert resolved")
    resolution_note: Optional[str] = Field(None, description="Resolution details")