"""
Signal aggregation and processing for institutional strategies.

Provides signal combination, filtering, and processing
capabilities for institutional trading strategies.
"""

from typing import List, Dict, Optional
from datetime import datetime
import statistics
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .base import StrategySignal


class SignalAggregator:
    """
    Signal aggregation and combination engine.
    
    Combines multiple trading signals using various
    aggregation methods for institutional decision making.
    """
    
    def __init__(self):
        """Initialize signal aggregator."""
        # AI-AGENT-REF: Signal aggregation engine
        logger.info("SignalAggregator initialized")
    
    def aggregate_signals(self, signals: List[StrategySignal], method: str = "weighted_average") -> Optional[StrategySignal]:
        """
        Aggregate multiple signals into a single signal.
        
        Args:
            signals: List of signals to aggregate
            method: Aggregation method
            
        Returns:
            Aggregated signal or None
        """
        try:
            if not signals:
                return None
            
            if method == "weighted_average":
                return self._weighted_average_aggregation(signals)
            elif method == "consensus":
                return self._consensus_aggregation(signals)
            elif method == "strongest":
                return self._strongest_signal_aggregation(signals)
            else:
                logger.warning(f"Unknown aggregation method: {method}")
                return None
            
        except Exception as e:
            logger.error(f"Error aggregating signals: {e}")
            return None
    
    def _weighted_average_aggregation(self, signals: List[StrategySignal]) -> StrategySignal:
        """Aggregate signals using weighted average."""
        # Group by symbol and side
        signal_groups = {}
        for signal in signals:
            key = (signal.symbol, signal.side)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        # Create aggregated signals
        aggregated_signals = []
        for (symbol, side), group_signals in signal_groups.items():
            if not group_signals:
                continue
            
            # Calculate weighted averages
            total_weight = sum(s.confidence for s in group_signals)
            if total_weight == 0:
                continue
            
            weighted_strength = sum(s.strength * s.confidence for s in group_signals) / total_weight
            avg_confidence = statistics.mean([s.confidence for s in group_signals])
            
            # Create aggregated signal
            aggregated_signal = StrategySignal(
                symbol=symbol,
                side=side,
                strength=weighted_strength,
                confidence=avg_confidence,
                signal_type="aggregated_weighted",
                metadata={
                    "source_signals": len(group_signals),
                    "aggregation_method": "weighted_average"
                }
            )
            
            aggregated_signals.append(aggregated_signal)
        
        # Return strongest aggregated signal
        if aggregated_signals:
            return max(aggregated_signals, key=lambda s: s.weighted_strength)
        return None
    
    def _consensus_aggregation(self, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Aggregate signals using consensus method."""
        # Count signals by symbol and side
        signal_counts = {}
        for signal in signals:
            key = (signal.symbol, signal.side)
            if key not in signal_counts:
                signal_counts[key] = []
            signal_counts[key].append(signal)
        
        # Find consensus (majority agreement)
        consensus_threshold = len(signals) * 0.6  # 60% consensus
        
        for (symbol, side), group_signals in signal_counts.items():
            if len(group_signals) >= consensus_threshold:
                avg_strength = statistics.mean([s.strength for s in group_signals])
                avg_confidence = statistics.mean([s.confidence for s in group_signals])
                
                return StrategySignal(
                    symbol=symbol,
                    side=side,
                    strength=avg_strength,
                    confidence=avg_confidence,
                    signal_type="consensus",
                    metadata={
                        "consensus_count": len(group_signals),
                        "total_signals": len(signals)
                    }
                )
        
        return None
    
    def _strongest_signal_aggregation(self, signals: List[StrategySignal]) -> StrategySignal:
        """Return the strongest signal."""
        return max(signals, key=lambda s: s.weighted_strength)


class SignalProcessor:
    """
    Signal processing and filtering engine.
    
    Processes signals with filtering, validation,
    and enhancement for institutional trading.
    """
    
    def __init__(self):
        """Initialize signal processor."""
        # AI-AGENT-REF: Signal processing engine
        self.min_confidence = 0.3
        self.min_strength = 0.1
        logger.info("SignalProcessor initialized")
    
    def process_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """
        Process and filter signals.
        
        Args:
            signals: Raw signals to process
            
        Returns:
            Processed and filtered signals
        """
        try:
            processed_signals = []
            
            for signal in signals:
                # Apply filters
                if self._passes_filters(signal):
                    # Enhance signal
                    enhanced_signal = self._enhance_signal(signal)
                    processed_signals.append(enhanced_signal)
            
            # Sort by weighted strength
            processed_signals.sort(key=lambda s: s.weighted_strength, reverse=True)
            
            logger.debug(f"Processed {len(processed_signals)} signals from {len(signals)} input signals")
            return processed_signals
            
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
            return []
    
    def _passes_filters(self, signal: StrategySignal) -> bool:
        """Check if signal passes filtering criteria."""
        # Confidence filter
        if signal.confidence < self.min_confidence:
            return False
        
        # Strength filter
        if signal.strength < self.min_strength:
            return False
        
        # Additional filters can be added here
        return True
    
    def _enhance_signal(self, signal: StrategySignal) -> StrategySignal:
        """Enhance signal with additional processing."""
        # For now, return the signal as-is
        # In a real implementation, this could add technical indicators,
        # risk adjustments, or other enhancements
        return signal