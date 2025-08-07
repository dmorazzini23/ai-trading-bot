"""
Signal aggregation and processing for institutional strategies.

Provides signal combination, filtering, and processing
capabilities for institutional trading strategies with
meta-learning, stacking, and turnover management.
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import statistics
import logging
import numpy as np
import pandas as pd

# Use the centralized logger as per AGENTS.md
try:
    from ai_trading.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    sklearn_available = True
except ImportError:
    sklearn_available = False
    logger.warning("sklearn not available - some meta-learning features disabled")

from .base import StrategySignal


class SignalAggregator:
    """
    Signal aggregation and combination engine with meta-learning.
    
    Combines multiple trading signals using various aggregation methods
    including stacking meta-learners with decay and turnover penalties.
    """
    
    def __init__(
        self,
        enable_stacking: bool = True,
        decay_window: int = 30,
        min_performance_window: int = 10,
        turnover_penalty: float = 0.1,
        conflict_resolution: str = "majority"
    ):
        """
        Initialize signal aggregator with meta-learning capabilities.
        
        Args:
            enable_stacking: Whether to use meta-learning stacking
            decay_window: Window for performance decay (days)
            min_performance_window: Minimum observations for meta-model
            turnover_penalty: Penalty for signal turnover
            conflict_resolution: How to resolve signal conflicts
        """
        # AI-AGENT-REF: Signal aggregation engine with meta-learning
        self.enable_stacking = enable_stacking
        self.decay_window = decay_window
        self.min_performance_window = min_performance_window
        self.turnover_penalty = turnover_penalty
        self.conflict_resolution = conflict_resolution
        
        # Meta-learning components
        self.meta_model = None
        self.signal_performance_history = []
        self.recent_weights = {}
        self.signal_decay_factors = {}
        
        # Performance tracking
        self.signal_metrics = {}
        self.ensemble_history = []
        
        logger.info("SignalAggregator initialized with meta-learning capabilities")
    
    def aggregate_signals(
        self, 
        signals: List[StrategySignal], 
        method: str = "stacking",
        market_data: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[StrategySignal]:
        """
        Aggregate multiple signals into a single signal.
        
        Args:
            signals: List of signals to aggregate
            method: Aggregation method ('stacking', 'weighted_average', 'consensus', 'strongest')
            market_data: Current market data for context
            timestamp: Timestamp for performance tracking
            
        Returns:
            Aggregated signal or None
        """
        try:
            if not signals:
                return None
            
            # Apply signal decay
            decayed_signals = self._apply_signal_decay(signals, timestamp)
            
            # Resolve conflicts if needed
            resolved_signals = self._resolve_signal_conflicts(decayed_signals)
            
            # Choose aggregation method
            if method == "stacking" and self.enable_stacking:
                aggregated_signal = self._stacking_aggregation(resolved_signals, market_data, timestamp)
            elif method == "weighted_average":
                aggregated_signal = self._weighted_average_aggregation(resolved_signals)
            elif method == "consensus":
                aggregated_signal = self._consensus_aggregation(resolved_signals)
            elif method == "strongest":
                aggregated_signal = self._strongest_signal_aggregation(resolved_signals)
            else:
                logger.warning(f"Unknown aggregation method: {method}, using weighted average")
                aggregated_signal = self._weighted_average_aggregation(resolved_signals)
            
            # Apply turnover penalty
            if aggregated_signal:
                aggregated_signal = self._apply_turnover_penalty(aggregated_signal, timestamp)
            
            # Update performance tracking
            self._update_performance_tracking(signals, aggregated_signal, timestamp)
            
            return aggregated_signal
            
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
    
    def _stacking_aggregation(
        self,
        signals: List[StrategySignal],
        market_data: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[StrategySignal]:
        """
        Aggregate signals using stacking meta-learner.
        
        Uses recent performance to learn optimal combination weights.
        """
        try:
            if not sklearn_available:
                logger.warning("sklearn not available, falling back to weighted average")
                return self._weighted_average_aggregation(signals)
            
            # Check if we have enough data for meta-learning
            if len(self.signal_performance_history) < self.min_performance_window:
                logger.debug("Insufficient data for stacking, using weighted average")
                return self._weighted_average_aggregation(signals)
            
            # Prepare features for meta-model
            features = self._prepare_meta_features(signals, market_data)
            
            if features is None:
                return self._weighted_average_aggregation(signals)
            
            # Train or update meta-model
            if self.meta_model is None:
                self._train_meta_model()
            
            # Predict optimal weights
            if self.meta_model is not None:
                try:
                    predicted_weight = self.meta_model.predict([features])[0]
                    # Ensure reasonable bounds
                    predicted_weight = max(-1.0, min(1.0, predicted_weight))
                except Exception as e:
                    logger.warning(f"Meta-model prediction failed: {e}")
                    predicted_weight = 0.0
            else:
                predicted_weight = 0.0
            
            # Create stacked signal
            if signals:
                base_signal = signals[0]  # Use first signal as template
                
                # Combine signals based on meta-model prediction
                combined_strength = 0.0
                combined_confidence = 0.0
                total_weight = 0.0
                
                for signal in signals:
                    # Weight based on recent performance and meta-model
                    signal_id = signal.metadata.get('source', 'unknown')
                    recent_performance = self.signal_metrics.get(signal_id, {}).get('recent_performance', 0.5)
                    
                    # Combine recent performance with meta-model prediction
                    weight = 0.7 * recent_performance + 0.3 * abs(predicted_weight)
                    
                    combined_strength += signal.strength * weight
                    combined_confidence += signal.confidence * weight
                    total_weight += weight
                
                if total_weight > 0:
                    combined_strength /= total_weight
                    combined_confidence /= total_weight
                
                # Create aggregated signal
                stacked_signal = StrategySignal(
                    symbol=base_signal.symbol,
                    side=base_signal.side if combined_strength > 0 else base_signal.side,
                    strength=abs(combined_strength),
                    confidence=combined_confidence,
                    signal_type="stacked_meta",
                    metadata={
                        "source_signals": len(signals),
                        "meta_prediction": predicted_weight,
                        "aggregation_method": "stacking",
                        "timestamp": timestamp.isoformat() if timestamp else datetime.now().isoformat()
                    }
                )
                
                return stacked_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in stacking aggregation: {e}")
            return self._weighted_average_aggregation(signals)
    
    def _apply_signal_decay(
        self,
        signals: List[StrategySignal],
        timestamp: Optional[datetime] = None
    ) -> List[StrategySignal]:
        """Apply time-based decay to stale signals."""
        try:
            if not timestamp:
                return signals
            
            decayed_signals = []
            
            for signal in signals:
                # Extract signal generation time
                signal_time_str = signal.metadata.get('timestamp')
                if signal_time_str:
                    try:
                        signal_time = datetime.fromisoformat(signal_time_str.replace('Z', '+00:00'))
                        age_hours = (timestamp - signal_time).total_seconds() / 3600
                        
                        # Apply exponential decay (half-life of 24 hours)
                        decay_factor = np.exp(-age_hours / 24)
                        decay_factor = max(0.1, decay_factor)  # Minimum 10% weight
                        
                        # Create decayed signal
                        decayed_signal = StrategySignal(
                            symbol=signal.symbol,
                            side=signal.side,
                            strength=signal.strength * decay_factor,
                            confidence=signal.confidence * decay_factor,
                            signal_type=signal.signal_type,
                            metadata={**signal.metadata, 'decay_factor': decay_factor}
                        )
                        
                        decayed_signals.append(decayed_signal)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing signal timestamp: {e}")
                        decayed_signals.append(signal)
                else:
                    # No timestamp, assume fresh signal
                    decayed_signals.append(signal)
            
            return decayed_signals
            
        except Exception as e:
            logger.error(f"Error applying signal decay: {e}")
            return signals
    
    def _resolve_signal_conflicts(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Resolve conflicts between opposing signals."""
        try:
            if len(signals) <= 1:
                return signals
            
            # Group signals by symbol and side
            signal_groups = {}
            for signal in signals:
                key = signal.symbol
                if key not in signal_groups:
                    signal_groups[key] = {'buy': [], 'sell': []}
                
                side_key = 'buy' if signal.is_buy else 'sell'
                signal_groups[key][side_key].append(signal)
            
            resolved_signals = []
            
            for symbol, sides in signal_groups.items():
                buy_signals = sides['buy']
                sell_signals = sides['sell']
                
                # If no conflict, add all signals
                if not buy_signals or not sell_signals:
                    resolved_signals.extend(buy_signals + sell_signals)
                    continue
                
                # Resolve conflict based on method
                if self.conflict_resolution == "majority":
                    if len(buy_signals) > len(sell_signals):
                        resolved_signals.extend(buy_signals)
                    elif len(sell_signals) > len(buy_signals):
                        resolved_signals.extend(sell_signals)
                    else:
                        # Tie: use strongest signal
                        all_signals = buy_signals + sell_signals
                        strongest = max(all_signals, key=lambda s: s.weighted_strength)
                        resolved_signals.append(strongest)
                
                elif self.conflict_resolution == "strongest":
                    all_signals = buy_signals + sell_signals
                    strongest = max(all_signals, key=lambda s: s.weighted_strength)
                    resolved_signals.append(strongest)
                
                elif self.conflict_resolution == "veto":
                    # If there's a conflict, veto all signals for this symbol
                    logger.debug(f"Veto applied for conflicting signals on {symbol}")
                    continue
                
                else:
                    # Default: take strongest from each side and net them
                    if buy_signals:
                        strongest_buy = max(buy_signals, key=lambda s: s.weighted_strength)
                        resolved_signals.append(strongest_buy)
                    if sell_signals:
                        strongest_sell = max(sell_signals, key=lambda s: s.weighted_strength)
                        resolved_signals.append(strongest_sell)
            
            return resolved_signals
            
        except Exception as e:
            logger.error(f"Error resolving signal conflicts: {e}")
            return signals
    
    def _apply_turnover_penalty(
        self,
        signal: StrategySignal,
        timestamp: Optional[datetime] = None
    ) -> StrategySignal:
        """Apply turnover penalty to reduce excessive trading."""
        try:
            # Track recent signals for turnover calculation
            signal_id = f"{signal.symbol}_{signal.side.value}"
            
            # Get recent signal history
            recent_signals = [
                entry for entry in self.ensemble_history[-10:]  # Last 10 signals
                if entry.get('signal_id') == signal_id
            ]
            
            # Calculate turnover penalty
            if len(recent_signals) > 3:  # If trading too frequently
                penalty_factor = 1.0 - (self.turnover_penalty * (len(recent_signals) - 3))
                penalty_factor = max(0.1, penalty_factor)  # Minimum 10% strength
                
                # Apply penalty to signal strength
                penalized_signal = StrategySignal(
                    symbol=signal.symbol,
                    side=signal.side,
                    strength=signal.strength * penalty_factor,
                    confidence=signal.confidence,
                    signal_type=signal.signal_type,
                    metadata={
                        **signal.metadata,
                        'turnover_penalty': penalty_factor,
                        'recent_signals_count': len(recent_signals)
                    }
                )
                
                return penalized_signal
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying turnover penalty: {e}")
            return signal
    
    def _prepare_meta_features(
        self,
        signals: List[StrategySignal],
        market_data: Optional[Dict] = None
    ) -> Optional[List[float]]:
        """Prepare features for meta-learning model."""
        try:
            features = []
            
            # Signal-based features
            if signals:
                strengths = [s.strength for s in signals]
                confidences = [s.confidence for s in signals]
                
                features.extend([
                    len(signals),                    # Number of signals
                    np.mean(strengths),             # Average strength
                    np.std(strengths) if len(strengths) > 1 else 0,  # Strength dispersion
                    np.mean(confidences),           # Average confidence
                    np.max(strengths),              # Max strength
                    np.min(strengths),              # Min strength
                ])
                
                # Signal agreement (how many point same direction)
                buy_signals = sum(1 for s in signals if s.is_buy)
                sell_signals = len(signals) - buy_signals
                agreement = abs(buy_signals - sell_signals) / len(signals)
                features.append(agreement)
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
            
            # Market context features
            if market_data:
                features.extend([
                    market_data.get('volatility', 0.0),
                    market_data.get('volume_ratio', 1.0),
                    market_data.get('spread_bps', 10.0),
                    market_data.get('momentum', 0.0)
                ])
            else:
                features.extend([0.0, 1.0, 10.0, 0.0])
            
            # Recent performance features
            if self.signal_metrics:
                avg_performance = np.mean([
                    metrics.get('recent_performance', 0.5)
                    for metrics in self.signal_metrics.values()
                ])
                features.append(avg_performance)
            else:
                features.append(0.5)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing meta features: {e}")
            return None
    
    def _train_meta_model(self) -> None:
        """Train the meta-learning model."""
        try:
            if not sklearn_available or len(self.signal_performance_history) < self.min_performance_window:
                return
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 5:  # Need minimum samples
                return
            
            # Train model (using Ridge regression for stability)
            self.meta_model = Ridge(alpha=1.0, random_state=42)
            self.meta_model.fit(X, y)
            
            logger.debug(f"Meta-model trained with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training meta-model: {e}")
            self.meta_model = None
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[float]]:
        """Prepare training data for meta-model."""
        try:
            X, y = [], []
            
            for entry in self.signal_performance_history[-100:]:  # Use recent history
                features = entry.get('features')
                performance = entry.get('performance')
                
                if features is not None and performance is not None:
                    X.append(features)
                    y.append(performance)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return [], []
    
    def _update_performance_tracking(
        self,
        input_signals: List[StrategySignal],
        output_signal: Optional[StrategySignal],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update performance tracking for meta-learning."""
        try:
            if not output_signal or not timestamp:
                return
            
            # Store for future performance evaluation
            entry = {
                'timestamp': timestamp,
                'input_signals': len(input_signals),
                'output_signal': {
                    'symbol': output_signal.symbol,
                    'strength': output_signal.strength,
                    'confidence': output_signal.confidence,
                    'side': output_signal.side.value
                },
                'signal_id': f"{output_signal.symbol}_{output_signal.side.value}"
            }
            
            self.ensemble_history.append(entry)
            
            # Keep recent history only
            if len(self.ensemble_history) > 1000:
                self.ensemble_history = self.ensemble_history[-500:]
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def update_signal_performance(
        self,
        signal_id: str,
        actual_return: float,
        prediction_horizon: int = 1
    ) -> None:
        """
        Update signal performance metrics for meta-learning.
        
        Args:
            signal_id: Identifier for the signal source
            actual_return: Actual return achieved
            prediction_horizon: Horizon of the prediction (in periods)
        """
        try:
            if signal_id not in self.signal_metrics:
                self.signal_metrics[signal_id] = {
                    'returns': [],
                    'sharpe_ratio': 0.0,
                    'hit_rate': 0.5,
                    'recent_performance': 0.5
                }
            
            # Add return to history
            self.signal_metrics[signal_id]['returns'].append(actual_return)
            
            # Keep recent history only
            if len(self.signal_metrics[signal_id]['returns']) > 100:
                self.signal_metrics[signal_id]['returns'] = self.signal_metrics[signal_id]['returns'][-50:]
            
            # Update metrics
            returns = self.signal_metrics[signal_id]['returns']
            if len(returns) > 5:
                # Sharpe ratio
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                    self.signal_metrics[signal_id]['sharpe_ratio'] = sharpe
                
                # Hit rate
                hit_rate = np.mean([1 if r > 0 else 0 for r in returns])
                self.signal_metrics[signal_id]['hit_rate'] = hit_rate
                
                # Recent performance (last 20 observations)
                recent_returns = returns[-20:]
                recent_perf = np.mean(recent_returns) if recent_returns else 0.0
                # Normalize to 0-1 scale
                recent_perf = max(0.0, min(1.0, (recent_perf + 0.02) / 0.04))  # Assume ±2% range
                self.signal_metrics[signal_id]['recent_performance'] = recent_perf
            
            # Update meta-model training data
            self._add_performance_observation(signal_id, actual_return)
            
        except Exception as e:
            logger.error(f"Error updating signal performance: {e}")
    
    def _add_performance_observation(self, signal_id: str, actual_return: float) -> None:
        """Add performance observation for meta-learning."""
        try:
            # Find corresponding prediction in ensemble history
            for entry in reversed(self.ensemble_history[-50:]):  # Search recent history
                if entry.get('signal_id') == signal_id:
                    # Add performance to training data
                    observation = {
                        'signal_id': signal_id,
                        'performance': actual_return,
                        'timestamp': entry['timestamp'],
                        'features': None  # Will be populated when needed
                    }
                    
                    self.signal_performance_history.append(observation)
                    
                    # Keep reasonable history size
                    if len(self.signal_performance_history) > 500:
                        self.signal_performance_history = self.signal_performance_history[-250:]
                    
                    break
            
        except Exception as e:
            logger.error(f"Error adding performance observation: {e}")
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive signal statistics."""
        try:
            stats = {
                'total_signals_processed': len(self.ensemble_history),
                'meta_model_trained': self.meta_model is not None,
                'signal_performance_samples': len(self.signal_performance_history),
                'tracked_signal_sources': len(self.signal_metrics),
                'recent_turnover': 0.0
            }
            
            # Calculate average recent performance
            if self.signal_metrics:
                recent_performances = [
                    metrics['recent_performance'] 
                    for metrics in self.signal_metrics.values()
                ]
                stats['avg_recent_performance'] = np.mean(recent_performances)
                stats['signal_performance_std'] = np.std(recent_performances)
            
            # Calculate recent turnover
            if len(self.ensemble_history) > 10:
                recent_symbols = set()
                for entry in self.ensemble_history[-10:]:
                    recent_symbols.add(entry.get('signal_id', ''))
                stats['recent_turnover'] = len(recent_symbols) / 10.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {}


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