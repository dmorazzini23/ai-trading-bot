"""Abstract base classes for institutional trading strategies."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import logging

from ..core.models import (
    TradingSignal, MarketData, PortfolioMetrics, 
    TradePosition, StrategyPerformance
)
from ..core.enums import StrategyType, TimeFrame, RiskLevel
from ..core.exceptions import StrategyError


logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(
        self,
        strategy_id: str,
        strategy_type: StrategyType,
        timeframe: TimeFrame,
        symbols: List[str],
        allocation: Decimal = Decimal('0.1'),
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        max_positions: int = 10,
        **kwargs
    ):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.timeframe = timeframe
        self.symbols = symbols
        self.allocation = allocation
        self.risk_level = risk_level
        self.max_positions = max_positions
        self.is_active = True
        self.created_at = datetime.now(timezone.utc)
        self.last_update = datetime.now(timezone.utc)
        
        # Performance tracking
        self.performance = StrategyPerformance(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            allocation=allocation
        )
        
        # Strategy-specific parameters
        self.parameters = kwargs
        
        # Internal state
        self._market_data: Dict[str, List[MarketData]] = {}
        self._positions: Dict[str, TradePosition] = {}
        self._signals_history: List[TradingSignal] = []
        
        logger.info(f"Initialized strategy {strategy_id} ({strategy_type}) for {len(symbols)} symbols")
    
    @abstractmethod
    def generate_signals(
        self, 
        market_data: Dict[str, List[MarketData]],
        portfolio_metrics: PortfolioMetrics,
        current_positions: Dict[str, TradePosition]
    ) -> List[TradingSignal]:
        """Generate trading signals based on market data and portfolio state.
        
        Args:
            market_data: Current market data for symbols
            portfolio_metrics: Current portfolio metrics
            current_positions: Current open positions
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        signal: TradingSignal,
        portfolio_metrics: PortfolioMetrics,
        current_positions: Dict[str, TradePosition]
    ) -> Decimal:
        """Calculate position size for a trading signal.
        
        Args:
            signal: Trading signal
            portfolio_metrics: Current portfolio metrics
            current_positions: Current open positions
            
        Returns:
            Position size in units
        """
        pass
    
    @abstractmethod
    def should_exit_position(
        self, 
        position: TradePosition,
        current_data: MarketData,
        portfolio_metrics: PortfolioMetrics
    ) -> bool:
        """Determine if a position should be exited.
        
        Args:
            position: Current position
            current_data: Latest market data
            portfolio_metrics: Current portfolio metrics
            
        Returns:
            True if position should be exited
        """
        pass
    
    def update_market_data(self, data: Dict[str, List[MarketData]]) -> None:
        """Update internal market data cache.
        
        Args:
            data: New market data by symbol
        """
        self._market_data.update(data)
        self.last_update = datetime.now(timezone.utc)
    
    def update_positions(self, positions: Dict[str, TradePosition]) -> None:
        """Update internal positions tracking.
        
        Args:
            positions: Current positions by symbol
        """
        self._positions = positions
        self.performance.active_positions = len(positions)
    
    def add_signal(self, signal: TradingSignal) -> None:
        """Add signal to history for analysis.
        
        Args:
            signal: Trading signal to record
        """
        self._signals_history.append(signal)
        
        # Keep only recent signals (last 1000)
        if len(self._signals_history) > 1000:
            self._signals_history = self._signals_history[-1000:]
    
    def get_recent_signals(self, hours: int = 24) -> List[TradingSignal]:
        """Get recent signals within specified timeframe.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent signals
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [s for s in self._signals_history if s.created_at >= cutoff]
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal before execution.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
            
        Raises:
            StrategyError: If signal validation fails
        """
        try:
            # Basic validation
            if signal.symbol not in self.symbols:
                raise StrategyError(
                    self.strategy_id, 
                    self.strategy_type.value,
                    f"Signal for unknown symbol: {signal.symbol}"
                )
            
            if not (0 <= signal.strength <= 1):
                raise StrategyError(
                    self.strategy_id,
                    self.strategy_type.value, 
                    f"Invalid signal strength: {signal.strength}"
                )
            
            if not (0 <= signal.confidence <= 1):
                raise StrategyError(
                    self.strategy_id,
                    self.strategy_type.value,
                    f"Invalid signal confidence: {signal.confidence}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validation failed for {self.strategy_id}: {e}")
            raise
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get strategy parameter value.
        
        Args:
            key: Parameter name
            default: Default value if not found
            
        Returns:
            Parameter value
        """
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set strategy parameter value.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        self.parameters[key] = value
        logger.info(f"Updated parameter {key}={value} for strategy {self.strategy_id}")
    
    def reset(self) -> None:
        """Reset strategy state."""
        self._market_data.clear()
        self._positions.clear()
        self._signals_history.clear()
        self.last_update = datetime.now(timezone.utc)
        logger.info(f"Reset strategy {self.strategy_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary representation.
        
        Returns:
            Strategy data as dictionary
        """
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type.value,
            "timeframe": self.timeframe.value,
            "symbols": self.symbols,
            "allocation": str(self.allocation),
            "risk_level": self.risk_level.name,
            "max_positions": self.max_positions,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat(),
            "parameters": self.parameters,
            "performance": self.performance.model_dump(),
        }


class TechnicalStrategy(BaseStrategy):
    """Base class for technical analysis strategies."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('strategy_type', StrategyType.MOMENTUM)
        super().__init__(**kwargs)
        
        # Technical indicators cache
        self._indicators: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    def calculate_indicators(
        self, 
        data: List[MarketData]
    ) -> Dict[str, Any]:
        """Calculate technical indicators for given data.
        
        Args:
            data: Historical market data
            
        Returns:
            Dictionary of calculated indicators
        """
        pass
    
    def update_indicators(self, symbol: str, data: List[MarketData]) -> None:
        """Update technical indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            data: Market data for indicator calculation
        """
        try:
            indicators = self.calculate_indicators(data)
            self._indicators[symbol] = indicators
            logger.debug(f"Updated indicators for {symbol}: {list(indicators.keys())}")
        except Exception as e:
            logger.error(f"Failed to update indicators for {symbol}: {e}")
            raise StrategyError(
                self.strategy_id,
                self.strategy_type.value,
                f"Indicator calculation failed for {symbol}: {e}"
            )
    
    def get_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get current indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicators
        """
        return self._indicators.get(symbol, {})


class MachineLearningStrategy(BaseStrategy):
    """Base class for machine learning strategies."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('strategy_type', StrategyType.MACHINE_LEARNING)
        super().__init__(**kwargs)
        
        # ML model state
        self._model = None
        self._feature_columns: List[str] = []
        self._last_training = None
        self._prediction_cache: Dict[str, Any] = {}
    
    @abstractmethod
    def prepare_features(
        self, 
        data: Dict[str, List[MarketData]]
    ) -> Dict[str, Any]:
        """Prepare features for ML model.
        
        Args:
            data: Market data by symbol
            
        Returns:
            Feature dictionary
        """
        pass
    
    @abstractmethod
    def train_model(
        self, 
        training_data: Dict[str, Any]
    ) -> None:
        """Train the ML model.
        
        Args:
            training_data: Prepared training data
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Make predictions using the ML model.
        
        Args:
            features: Input features
            
        Returns:
            Prediction scores by symbol
        """
        pass
    
    def needs_retraining(self) -> bool:
        """Check if model needs retraining.
        
        Returns:
            True if model should be retrained
        """
        if self._model is None or self._last_training is None:
            return True
        
        # Retrain weekly
        retrain_interval = timedelta(days=7)
        return datetime.now(timezone.utc) - self._last_training > retrain_interval


class EnsembleStrategy(BaseStrategy):
    """Base class for ensemble strategies combining multiple strategies."""
    
    def __init__(
        self,
        sub_strategies: List[BaseStrategy],
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        kwargs.setdefault('strategy_type', StrategyType.ENSEMBLE)
        super().__init__(**kwargs)
        
        self.sub_strategies = sub_strategies
        self.weights = weights or [1.0 / len(sub_strategies)] * len(sub_strategies)
        
        if len(self.weights) != len(sub_strategies):
            raise StrategyError(
                self.strategy_id,
                self.strategy_type.value,
                "Number of weights must match number of sub-strategies"
            )
    
    def generate_signals(
        self,
        market_data: Dict[str, List[MarketData]],
        portfolio_metrics: PortfolioMetrics,
        current_positions: Dict[str, TradePosition]
    ) -> List[TradingSignal]:
        """Generate ensemble signals by combining sub-strategy signals.
        
        Args:
            market_data: Current market data
            portfolio_metrics: Portfolio metrics
            current_positions: Current positions
            
        Returns:
            Combined trading signals
        """
        all_signals: Dict[str, List[TradingSignal]] = {}
        
        # Collect signals from all sub-strategies
        for strategy in self.sub_strategies:
            if strategy.is_active:
                try:
                    signals = strategy.generate_signals(
                        market_data, portfolio_metrics, current_positions
                    )
                    for signal in signals:
                        if signal.symbol not in all_signals:
                            all_signals[signal.symbol] = []
                        all_signals[signal.symbol].append(signal)
                except Exception as e:
                    logger.error(f"Sub-strategy {strategy.strategy_id} failed: {e}")
        
        # Combine signals for each symbol
        combined_signals = []
        for symbol, signals in all_signals.items():
            if signals:
                combined_signal = self._combine_signals(signals)
                if combined_signal:
                    combined_signals.append(combined_signal)
        
        return combined_signals
    
    def _combine_signals(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Combine multiple signals for the same symbol.
        
        Args:
            signals: List of signals for same symbol
            
        Returns:
            Combined signal or None
        """
        if not signals:
            return None
        
        # Weighted average of signal strengths
        weighted_strength = sum(
            signal.strength * weight 
            for signal, weight in zip(signals, self.weights[:len(signals)])
        ) / sum(self.weights[:len(signals)])
        
        # Weighted average of confidence
        weighted_confidence = sum(
            signal.confidence * weight
            for signal, weight in zip(signals, self.weights[:len(signals)])
        ) / sum(self.weights[:len(signals)])
        
        # Use the most common side
        sides = [signal.side for signal in signals]
        most_common_side = max(set(sides), key=sides.count)
        
        # Create combined signal
        return TradingSignal(
            symbol=signals[0].symbol,
            signal_type=self.strategy_type,
            side=most_common_side,
            strength=weighted_strength,
            confidence=weighted_confidence,
            timeframe=self.timeframe,
            metadata={
                "sub_signals": len(signals),
                "contributing_strategies": [s.metadata.get("strategy_id") for s in signals],
            }
        )