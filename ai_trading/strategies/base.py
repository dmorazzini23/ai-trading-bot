"""
Base strategy framework for institutional trading.

Provides abstract base strategy class and strategy registry
for implementing and managing institutional trading strategies.
"""
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any
from ai_trading.logging import logger, logger_once
from ..core.enums import RiskLevel
from ..order.types import OrderSide

class StrategySignal:
    """
    Trading signal representation.

    Encapsulates signal data including strength, confidence,
    and metadata for institutional decision making.
    """

    def __init__(
        self,
        symbol: str,
        side: str | OrderSide | None,
        strength: float | None = None,
        confidence: float | None = None,
        **kwargs,
    ):
        """Initialize trading signal.

        Parameters
        ----------
        symbol : str
            Asset ticker symbol.
        side : str | OrderSide
            Direction of the trade (``buy`` or ``sell``). ``hold`` is permitted
            for neutral signals.
        strength : float, optional
            Raw signal strength in the ``[0, 1]`` range.
        confidence : float, optional
            Confidence weight in the ``[0, 1]`` range.
        """
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self._side_is_enum = isinstance(side, OrderSide)
        self.side = self._normalize_side(side)
        self._strength_provided = strength is not None
        self._confidence_provided = confidence is not None
        strength_val = 0.0 if strength is None else float(strength)
        confidence_val = 0.0 if confidence is None else float(confidence)
        self.strength = max(0.0, min(1.0, strength_val))
        self.confidence = max(0.0, min(1.0, confidence_val))
        self.timestamp = datetime.now(UTC)
        self.strategy_id = kwargs.get('strategy_id')
        self.timeframe = kwargs.get('timeframe', '1d')
        self.price_target = kwargs.get('price_target')
        self.stop_loss = kwargs.get('stop_loss')
        self.expected_return = kwargs.get('expected_return', 0.0)
        self.risk_score = kwargs.get('risk_score', 0.5)
        self.signal_type = kwargs.get('signal_type', 'momentum')
        self.metadata = kwargs.get('metadata', {})

    @staticmethod
    def _normalize_side(side: str | OrderSide | None) -> str:
        if isinstance(side, OrderSide):
            side = side.value
        value = str(side or "").strip().lower()
        if value in {"buy", "long", "entry"}:
            return "buy"
        if value in {"sell", "short", "exit"}:
            return "sell"
        if value in {"hold", "neutral", "none", "flat", ""}:
            return "hold"
        logger.warning("SIGNAL_SIDE_INVALID", extra={"side": value})
        return "hold"

    @property
    def has_explicit_strength(self) -> bool:
        """Return True when strength was explicitly provided."""
        return bool(self._strength_provided)

    @property
    def has_explicit_confidence(self) -> bool:
        """Return True when confidence was explicitly provided."""
        return bool(self._confidence_provided)

    @property
    def weighted_strength(self) -> float:
        """Calculate confidence-weighted signal strength."""
        return self.strength * self.confidence

    @property
    def score(self) -> float:
        """Return signed strength based on side."""
        if self.side == "buy":
            return self.strength
        if self.side == "sell":
            return -self.strength
        return 0.0

    @property
    def is_buy(self) -> bool:
        """Check if signal is a buy signal."""
        return self.side == "buy"

    @property
    def is_sell(self) -> bool:
        """Check if signal is a sell signal."""
        return self.side == "sell"

    def to_dict(self) -> dict:
        """Convert signal to dictionary representation."""
        return {'id': self.id, 'symbol': self.symbol, 'side': self.side, 'strength': self.strength, 'confidence': self.confidence, 'weighted_strength': self.weighted_strength, 'score': self.score, 'timestamp': self.timestamp.isoformat(), 'strategy_id': self.strategy_id, 'timeframe': self.timeframe, 'price_target': self.price_target, 'stop_loss': self.stop_loss, 'expected_return': self.expected_return, 'risk_score': self.risk_score, 'signal_type': self.signal_type, 'metadata': self.metadata}


# Backwards-compatible alias used by tests.
TradeSignal = StrategySignal

class BaseStrategy(ABC):
    """
    Abstract base class for institutional trading strategies.

    Provides framework for strategy implementation including
    signal generation, risk management, and performance tracking.
    """

    def __init__(self, strategy_id: str, name: str, risk_level: RiskLevel=RiskLevel.MODERATE):
        """Initialize base strategy."""
        self.strategy_id = strategy_id
        self.name = name
        self.risk_level = risk_level
        self.created_at = datetime.now(UTC)
        self.is_active = False
        self.parameters = {}
        self.symbols = []
        self.timeframes = ['1d']
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.max_position_size = risk_level.max_position_size
        self.max_drawdown_threshold = risk_level.max_drawdown_threshold
        logger_once.info(f'Strategy {self.name} ({self.strategy_id}) initialized with risk level {risk_level}')

    @abstractmethod
    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        """
        Generate trading signals based on market data.

        Args:
            market_data: Current market data and indicators

        Returns:
            List of trading signals
        """

    def generate_signal(
        self,
        symbol: str,
        side: str | OrderSide = "buy",
        strength: float | None = None,
        confidence: float | None = None,
        **kwargs: Any,
    ) -> StrategySignal:
        """Generate a placeholder single trading signal.

        This base implementation returns a neutral :class:`StrategySignal` and
        is intended to be overridden by concrete strategy implementations.

        Parameters
        ----------
        symbol:
            Asset ticker symbol.
        side:
            Trade direction, defaults to ``buy``.
        strength:
            Raw signal strength in ``[0, 1]``.
        confidence:
            Confidence weight in ``[0, 1]``.
        **kwargs:
            Additional metadata forwarded to :class:`StrategySignal`.

        Returns
        -------
        StrategySignal
            Placeholder signal object.
        """
        return StrategySignal(
            symbol=symbol,
            side=side,
            strength=strength,
            confidence=confidence,
            **kwargs,
        )

    def generate(self, ctx: Any) -> list[StrategySignal]:
        """Return list of signals from market context (dict-compat)."""
        if isinstance(ctx, dict):
            market_data = ctx
        else:
            md = getattr(ctx, 'market_data', None)
            if isinstance(md, dict):
                market_data = md
            else:
                market_data = {'symbols': getattr(ctx, 'symbols', []), 'data_by_symbol': getattr(ctx, 'data_by_symbol', {})}
        return self.generate_signals(market_data)

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_position: float = 0,
    ) -> int:
        """Calculate position size for a signal.

        The base implementation returns ``0`` as a placeholder and should be
        overridden by subclasses that implement concrete risk management
        logic.

        Parameters
        ----------
        signal:
            Trading signal for which to determine size.
        portfolio_value:
            Current portfolio value in dollars.
        current_position:
            Existing position size, if any. Defaults to ``0``.

        Returns
        -------
        int
            Placeholder position size ``0``.
        """
        return 0

    def validate_signal(self, signal: StrategySignal) -> bool:
        """
        Validate trading signal against strategy rules.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        try:
            if not getattr(signal, "has_explicit_strength", False) or not getattr(signal, "has_explicit_confidence", False):
                logger.warning("SIGNAL_MISSING_STRENGTH_CONFIDENCE", extra={"symbol": getattr(signal, "symbol", None)})
                return False
            if not signal.symbol or signal.strength <= 0:
                return False
            if signal.risk_score > 0.8:
                logger.warning(f'High risk signal for {signal.symbol}: {signal.risk_score}')
                return False
            if signal.confidence < 0.3:
                logger.debug(f'Low confidence signal for {signal.symbol}: {signal.confidence}')
                return False
            return True
        except (ValueError, TypeError) as e:
            logger.error(f'Error validating signal: {e}')
            return False

    def update_parameters(self, new_parameters: dict):
        """Update strategy parameters."""
        try:
            self.parameters.update(new_parameters)
            logger.info(f'Strategy {self.name} parameters updated: {new_parameters}')
        except (ValueError, TypeError) as e:
            logger.error(f'Error updating strategy parameters: {e}')

    def add_symbol(self, symbol: str):
        """Add symbol to strategy universe."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.debug(f'Symbol {symbol} added to strategy {self.name}')

    def remove_symbol(self, symbol: str):
        """Remove symbol from strategy universe."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            logger.debug(f'Symbol {symbol} removed from strategy {self.name}')

    def activate(self):
        """Activate strategy for live trading."""
        self.is_active = True
        logger.info(f'Strategy {self.name} activated')

    def deactivate(self):
        """Deactivate strategy."""
        self.is_active = False
        logger.info(f'Strategy {self.name} deactivated')

    def update_performance(self, return_pct: float, is_winner: bool):
        """Update strategy performance metrics."""
        try:
            self.trades_executed += 1
            self.total_return += return_pct
            if is_winner:
                self.win_rate = (self.win_rate * (self.trades_executed - 1) + 1) / self.trades_executed
            else:
                self.win_rate = self.win_rate * (self.trades_executed - 1) / self.trades_executed
            if return_pct < 0:
                self.max_drawdown = min(self.max_drawdown, return_pct)
        except (ValueError, TypeError) as e:
            logger.error(f'Error updating performance: {e}')

    def get_performance_summary(self) -> dict:
        """Get strategy performance summary."""
        return {'strategy_id': self.strategy_id, 'name': self.name, 'is_active': self.is_active, 'signals_generated': self.signals_generated, 'trades_executed': self.trades_executed, 'total_return': self.total_return, 'average_return': self.total_return / self.trades_executed if self.trades_executed > 0 else 0, 'max_drawdown': self.max_drawdown, 'win_rate': self.win_rate, 'risk_level': self.risk_level.value, 'symbols_count': len(self.symbols), 'created_at': self.created_at.isoformat()}

    def to_dict(self) -> dict:
        """Convert strategy to dictionary representation."""
        return {'strategy_id': self.strategy_id, 'name': self.name, 'risk_level': self.risk_level.value, 'is_active': self.is_active, 'parameters': self.parameters, 'symbols': self.symbols, 'timeframes': self.timeframes, 'performance': self.get_performance_summary(), 'created_at': self.created_at.isoformat()}

class Strategy(BaseStrategy):
    """Minimal concrete strategy used for tests and fallbacks."""

    def __init__(self) -> None:
        super().__init__(strategy_id="base", name="Base Strategy")

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        """Return an empty signal list for any market data."""
        return []


class StrategyRegistry:
    """
    Registry for managing multiple trading strategies.

    Provides centralized strategy management, activation control,
    and strategy performance monitoring.
    """

    def __init__(self):
        """Initialize strategy registry."""
        self.strategies: dict[str, BaseStrategy] = {}
        self.active_strategies: dict[str, BaseStrategy] = {}
        self.strategy_performance = {}
        logger.info('StrategyRegistry initialized')

    def register_strategy(self, strategy: BaseStrategy) -> bool:
        """
        Register a new strategy.

        Args:
            strategy: Strategy to register

        Returns:
            True if successfully registered
        """
        try:
            if strategy.strategy_id in self.strategies:
                logger.warning(f'Strategy {strategy.strategy_id} already registered')
                return False
            self.strategies[strategy.strategy_id] = strategy
            pos_size = getattr(strategy, 'max_position_size', 0) or 0
            if pos_size <= 0:
                pos_size = 1
            self.strategy_performance[strategy.strategy_id] = {'position_size': pos_size}
            logger.info(f'Strategy registered: {strategy.name} ({strategy.strategy_id})')
            return True
        except (ValueError, TypeError) as e:
            logger.error(f'Error registering strategy: {e}')
            return False

    def unregister_strategy(self, strategy_id: str) -> bool:
        """
        Unregister a strategy.

        Args:
            strategy_id: ID of strategy to unregister

        Returns:
            True if successfully unregistered
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f'Strategy {strategy_id} not found')
                return False
            if strategy_id in self.active_strategies:
                self.deactivate_strategy(strategy_id)
            strategy = self.strategies.pop(strategy_id)
            logger.info(f'Strategy unregistered: {strategy.name} ({strategy_id})')
            return True
        except (ValueError, TypeError) as e:
            logger.error(f'Error unregistering strategy: {e}')
            return False

    def activate_strategy(self, strategy_id: str) -> bool:
        """
        Activate a strategy for live trading.

        Args:
            strategy_id: ID of strategy to activate

        Returns:
            True if successfully activated
        """
        try:
            if strategy_id not in self.strategies:
                logger.error(f'Cannot activate unknown strategy: {strategy_id}')
                return False
            strategy = self.strategies[strategy_id]
            strategy.activate()
            self.active_strategies[strategy_id] = strategy
            logger.info(f'Strategy activated: {strategy.name}')
            return True
        except (ValueError, TypeError) as e:
            logger.error(f'Error activating strategy {strategy_id}: {e}')
            return False

    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Deactivate a strategy.

        Args:
            strategy_id: ID of strategy to deactivate

        Returns:
            True if successfully deactivated
        """
        try:
            if strategy_id in self.active_strategies:
                strategy = self.active_strategies.pop(strategy_id)
                strategy.deactivate()
                logger.info(f'Strategy deactivated: {strategy.name}')
                return True
            else:
                logger.warning(f'Strategy {strategy_id} not active')
                return False
        except (ValueError, TypeError) as e:
            logger.error(f'Error deactivating strategy {strategy_id}: {e}')
            return False

    def get_strategy(self, strategy_id: str) -> BaseStrategy | None:
        """Get strategy by ID."""
        return self.strategies.get(strategy_id)

    def get_active_strategies(self) -> list[BaseStrategy]:
        """Get list of active strategies."""
        return list(self.active_strategies.values())

    def get_all_strategies(self) -> list[BaseStrategy]:
        """Get list of all registered strategies."""
        return list(self.strategies.values())

    def generate_signals_from_active_strategies(self, market_data: dict) -> list[StrategySignal]:
        """
        Generate signals from all active strategies.

        Args:
            market_data: Current market data

        Returns:
            List of all signals from active strategies
        """
        all_signals = []
        for strategy in self.active_strategies.values():
            try:
                signals = strategy.generate_signals(market_data)
                for signal in signals:
                    if strategy.validate_signal(signal):
                        signal.strategy_id = strategy.strategy_id
                        all_signals.append(signal)
                        strategy.signals_generated += 1
            except (ValueError, TypeError) as e:
                logger.error(f'Error generating signals from strategy {strategy.name}: {e}')
        logger.debug(f'Generated {len(all_signals)} signals from {len(self.active_strategies)} active strategies')
        return all_signals

    def get_registry_summary(self) -> dict:
        """Get summary of strategy registry."""
        return {'total_strategies': len(self.strategies), 'active_strategies': len(self.active_strategies), 'strategy_list': [{'id': strategy.strategy_id, 'name': strategy.name, 'is_active': strategy.is_active, 'signals_generated': strategy.signals_generated, 'trades_executed': strategy.trades_executed} for strategy in self.strategies.values()]}
