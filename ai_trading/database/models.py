"""
SQLAlchemy database models for institutional trading platform.

Contains models for trades, portfolio positions, risk metrics,
and performance tracking with proper relationships and constraints.
"""

from datetime import datetime, timezone
from decimal import Decimal
import uuid

# Using built-in modules to avoid dependency issues
from datetime import datetime as dt_datetime


# Minimal SQLAlchemy-like base class for now
class DeclarativeBase:
    """Base class for database models."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert model to dictionary."""
        result = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                if isinstance(value, (datetime, dt_datetime)):
                    value = value.isoformat()
                elif isinstance(value, Decimal):
                    value = float(value)
                result[attr] = value
        return result


Base = DeclarativeBase


class Trade(Base):
    """
    Trade model for recording all trading activity.
    
    Stores execution details, pricing, and metadata for each trade
    with full audit trail and performance tracking.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # AI-AGENT-REF: Core trade tracking model
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.symbol = kwargs.get('symbol')
        self.side = kwargs.get('side')  # 'buy' or 'sell'
        self.order_type = kwargs.get('order_type', 'market')
        self.quantity = kwargs.get('quantity', 0)
        self.price = kwargs.get('price', 0.0)
        self.executed_price = kwargs.get('executed_price')
        self.status = kwargs.get('status', 'pending')
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.executed_at = kwargs.get('executed_at')
        self.commission = kwargs.get('commission', 0.0)
        self.slippage = kwargs.get('slippage', 0.0)
        self.strategy_id = kwargs.get('strategy_id')
        self.signal_strength = kwargs.get('signal_strength', 0.0)
        self.stop_loss = kwargs.get('stop_loss')
        self.take_profit = kwargs.get('take_profit')
        self.notes = kwargs.get('notes', '')
        self.market_data_snapshot = kwargs.get('market_data_snapshot', '{}')
    
    @property
    def gross_pnl(self) -> float:
        """Calculate gross P&L for the trade."""
        if not self.executed_price or self.status != 'filled':
            return 0.0
        
        if self.side == 'buy':
            # For buy orders, profit when current/exit price > entry price
            return self.quantity * (self.executed_price - self.price)
        else:
            # For sell orders, profit when entry price > current/exit price  
            return self.quantity * (self.price - self.executed_price)
    
    @property
    def net_pnl(self) -> float:
        """Calculate net P&L after commissions."""
        return self.gross_pnl - self.commission
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of the trade."""
        price = self.executed_price or self.price
        return abs(self.quantity * price)


class Portfolio(Base):
    """
    Portfolio model for tracking positions and valuations.
    
    Maintains current positions, cash balances, and portfolio-level
    metrics with real-time updates and historical tracking.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # AI-AGENT-REF: Portfolio position tracking
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.account_id = kwargs.get('account_id')
        self.symbol = kwargs.get('symbol')
        self.quantity = kwargs.get('quantity', 0)
        self.average_cost = kwargs.get('average_cost', 0.0)
        self.current_price = kwargs.get('current_price', 0.0)
        self.last_updated = kwargs.get('last_updated', datetime.now(timezone.utc))
        self.asset_class = kwargs.get('asset_class', 'equity')
        self.sector = kwargs.get('sector')
        self.market_value = kwargs.get('market_value', 0.0)
        self.unrealized_pnl = kwargs.get('unrealized_pnl', 0.0)
        self.realized_pnl = kwargs.get('realized_pnl', 0.0)
        self.day_change = kwargs.get('day_change', 0.0)
        self.day_change_percent = kwargs.get('day_change_percent', 0.0)
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost basis of position."""
        return abs(self.quantity * self.average_cost)
    
    @property
    def current_market_value(self) -> float:
        """Calculate current market value of position."""
        return abs(self.quantity * self.current_price)
    
    @property
    def position_pnl(self) -> float:
        """Calculate unrealized P&L of position."""
        if self.quantity == 0:
            return 0.0
        return (self.current_price - self.average_cost) * self.quantity
    
    @property
    def position_pnl_percent(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.average_cost == 0:
            return 0.0
        return ((self.current_price - self.average_cost) / self.average_cost) * 100


class RiskMetric(Base):
    """
    Risk metrics model for tracking portfolio risk measures.
    
    Stores calculated risk metrics including VaR, Sharpe ratio,
    drawdown, and other institutional risk measures.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # AI-AGENT-REF: Risk measurement and tracking
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.portfolio_id = kwargs.get('portfolio_id')
        self.calculation_date = kwargs.get('calculation_date', datetime.now(timezone.utc))
        self.var_95 = kwargs.get('var_95', 0.0)
        self.var_99 = kwargs.get('var_99', 0.0)
        self.expected_shortfall = kwargs.get('expected_shortfall', 0.0)
        self.sharpe_ratio = kwargs.get('sharpe_ratio', 0.0)
        self.sortino_ratio = kwargs.get('sortino_ratio', 0.0)
        self.max_drawdown = kwargs.get('max_drawdown', 0.0)
        self.current_drawdown = kwargs.get('current_drawdown', 0.0)
        self.volatility = kwargs.get('volatility', 0.0)
        self.beta = kwargs.get('beta', 1.0)
        self.correlation_spy = kwargs.get('correlation_spy', 0.0)
        self.concentration_risk = kwargs.get('concentration_risk', 0.0)
        self.liquidity_risk = kwargs.get('liquidity_risk', 0.0)
    
    @property
    def risk_score(self) -> float:
        """Calculate composite risk score (0-100)."""
        # Simple composite score based on key metrics
        var_score = min(abs(self.var_95) * 1000, 50)  # Scale VaR
        drawdown_score = min(abs(self.max_drawdown) * 100, 30)  # Scale drawdown
        vol_score = min(self.volatility * 100, 20)  # Scale volatility
        
        return var_score + drawdown_score + vol_score
    
    @property
    def risk_level(self) -> str:
        """Categorize risk level based on composite score."""
        score = self.risk_score
        if score < 25:
            return "Low"
        elif score < 50:
            return "Medium"
        elif score < 75:
            return "High"
        else:
            return "Critical"


class PerformanceMetric(Base):
    """
    Performance metrics model for tracking strategy and portfolio performance.
    
    Stores calculated performance metrics including returns, ratios,
    and comparative benchmarking over various time periods.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # AI-AGENT-REF: Performance measurement and analytics
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.strategy_id = kwargs.get('strategy_id')
        self.portfolio_id = kwargs.get('portfolio_id')
        self.period_start = kwargs.get('period_start')
        self.period_end = kwargs.get('period_end')
        self.total_return = kwargs.get('total_return', 0.0)
        self.annualized_return = kwargs.get('annualized_return', 0.0)
        self.benchmark_return = kwargs.get('benchmark_return', 0.0)
        self.alpha = kwargs.get('alpha', 0.0)
        self.tracking_error = kwargs.get('tracking_error', 0.0)
        self.information_ratio = kwargs.get('information_ratio', 0.0)
        self.win_rate = kwargs.get('win_rate', 0.0)
        self.profit_factor = kwargs.get('profit_factor', 0.0)
        self.average_win = kwargs.get('average_win', 0.0)
        self.average_loss = kwargs.get('average_loss', 0.0)
        self.largest_win = kwargs.get('largest_win', 0.0)
        self.largest_loss = kwargs.get('largest_loss', 0.0)
        self.total_trades = kwargs.get('total_trades', 0)
        self.winning_trades = kwargs.get('winning_trades', 0)
        self.losing_trades = kwargs.get('losing_trades', 0)
    
    @property
    def loss_rate(self) -> float:
        """Calculate loss rate."""
        return 1.0 - self.win_rate if self.win_rate else 0.0
    
    @property
    def avg_win_loss_ratio(self) -> float:
        """Calculate average win to loss ratio."""
        if self.average_loss == 0:
            return float('inf') if self.average_win > 0 else 0.0
        return abs(self.average_win / self.average_loss)
    
    @property
    def expectancy(self) -> float:
        """Calculate trade expectancy."""
        if self.total_trades == 0:
            return 0.0
        
        win_prob = self.win_rate
        loss_prob = self.loss_rate
        avg_win = self.average_win
        avg_loss = abs(self.average_loss)
        
        return (win_prob * avg_win) - (loss_prob * avg_loss)
    
    @property
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        # Need max_drawdown from risk metrics - simplified calculation
        if hasattr(self, 'max_drawdown') and self.max_drawdown != 0:
            return self.annualized_return / abs(self.max_drawdown)
        return 0.0