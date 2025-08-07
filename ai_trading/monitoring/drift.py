"""
Feature drift monitoring and signal attribution module.

Monitors feature drift using Population Stability Index (PSI),
tracks per-signal attribution, and provides shadow mode for
experimental model validation.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Feature drift metrics."""
    feature_name: str
    psi_score: float
    drift_level: str  # 'low', 'medium', 'high'
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float
    sample_size: int
    calculated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['calculated_at'] = self.calculated_at.isoformat()
        return result


@dataclass
class SignalAttribution:
    """Per-signal performance attribution."""
    signal_name: str
    period_return: float
    hit_ratio: float
    sharpe_ratio: float
    turnover: float
    max_drawdown: float
    trade_count: int
    avg_hold_period: float
    total_pnl: float
    period_start: datetime
    period_end: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['period_start'] = self.period_start.isoformat()
        result['period_end'] = self.period_end.isoformat()
        return result


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    psi_low: float = 0.1     # PSI < 0.1: no drift
    psi_medium: float = 0.2  # 0.1 < PSI < 0.2: medium drift
    psi_high: float = 0.2    # PSI > 0.2: high drift
    
    hit_ratio_min: float = 0.45      # Minimum hit ratio
    sharpe_ratio_min: float = 0.5    # Minimum Sharpe ratio
    drawdown_max: float = 0.15       # Maximum drawdown tolerance


class DriftMonitor:
    """
    Feature drift and signal attribution monitoring system.
    """
    
    def __init__(
        self,
        baseline_data_path: str = "artifacts/monitoring/baseline",
        alert_thresholds: Optional[AlertThreshold] = None
    ):
        """
        Initialize drift monitor.
        
        Args:
            baseline_data_path: Path to baseline data storage
            alert_thresholds: Alert threshold configuration
        """
        self.baseline_path = Path(baseline_data_path)
        self.baseline_path.mkdir(parents=True, exist_ok=True)
        
        self.alert_thresholds = alert_thresholds or AlertThreshold()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Baseline feature statistics
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._load_baseline_stats()
        
        # Signal attribution history
        self._signal_history: Dict[str, List[SignalAttribution]] = {}
    
    def _load_baseline_stats(self) -> None:
        """Load baseline feature statistics."""
        baseline_file = self.baseline_path / "feature_baseline.json"
        
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    self._baseline_stats = json.load(f)
                    
                self.logger.info(f"Loaded baseline stats for {len(self._baseline_stats)} features")
                
            except Exception as e:
                self.logger.error(f"Failed to load baseline stats: {e}")
    
    def _save_baseline_stats(self) -> None:
        """Save baseline feature statistics."""
        baseline_file = self.baseline_path / "feature_baseline.json"
        
        try:
            with open(baseline_file, 'w') as f:
                json.dump(self._baseline_stats, f, indent=2)
                
            self.logger.info(f"Saved baseline stats for {len(self._baseline_stats)} features")
            
        except Exception as e:
            self.logger.error(f"Failed to save baseline stats: {e}")
    
    def update_baseline(self, feature_data: pd.DataFrame) -> None:
        """
        Update baseline feature statistics.
        
        Args:
            feature_data: Baseline feature dataset
        """
        for column in feature_data.columns:
            feature_values = feature_data[column].dropna()
            
            if len(feature_values) > 0:
                self._baseline_stats[column] = {
                    'mean': float(feature_values.mean()),
                    'std': float(feature_values.std()),
                    'min': float(feature_values.min()),
                    'max': float(feature_values.max()),
                    'count': len(feature_values),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
        
        self._save_baseline_stats()
        self.logger.info(f"Updated baseline for {len(feature_data.columns)} features")
    
    def calculate_psi(
        self,
        baseline_data: np.ndarray,
        current_data: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            baseline_data: Baseline feature values
            current_data: Current feature values
            n_bins: Number of bins for distribution comparison
            
        Returns:
            PSI score
        """
        if len(baseline_data) == 0 or len(current_data) == 0:
            return 0.0
        
        try:
            # Create bins based on baseline data quantiles
            bins = np.quantile(baseline_data, np.linspace(0, 1, n_bins + 1))
            bins = np.unique(bins)  # Remove duplicates
            
            if len(bins) < 2:
                return 0.0
            
            # Calculate distributions
            baseline_dist, _ = np.histogram(baseline_data, bins=bins, density=True)
            current_dist, _ = np.histogram(current_data, bins=bins, density=True)
            
            # Normalize to probabilities
            baseline_dist = baseline_dist / baseline_dist.sum()
            current_dist = current_dist / current_dist.sum()
            
            # Avoid division by zero
            baseline_dist = np.where(baseline_dist == 0, 0.0001, baseline_dist)
            current_dist = np.where(current_dist == 0, 0.0001, current_dist)
            
            # Calculate PSI
            psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
            
            return float(psi)
            
        except Exception as e:
            self.logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def monitor_feature_drift(self, current_features: pd.DataFrame) -> List[DriftMetrics]:
        """
        Monitor feature drift against baseline.
        
        Args:
            current_features: Current feature dataset
            
        Returns:
            List of drift metrics for each feature
        """
        drift_metrics = []
        
        for feature_name in current_features.columns:
            if feature_name not in self._baseline_stats:
                self.logger.warning(f"No baseline stats for feature: {feature_name}")
                continue
            
            current_values = current_features[feature_name].dropna()
            baseline_stats = self._baseline_stats[feature_name]
            
            if len(current_values) == 0:
                continue
            
            # Create synthetic baseline data for PSI calculation
            # (In practice, you'd store actual baseline data)
            baseline_mean = baseline_stats['mean']
            baseline_std = baseline_stats['std']
            baseline_count = min(baseline_stats['count'], 10000)  # Limit size
            
            # Generate baseline sample assuming normal distribution
            np.random.seed(42)  # For reproducibility
            baseline_sample = np.random.normal(baseline_mean, baseline_std, baseline_count)
            
            # Calculate PSI
            psi_score = self.calculate_psi(baseline_sample, current_values.values)
            
            # Determine drift level
            if psi_score < self.alert_thresholds.psi_low:
                drift_level = "low"
            elif psi_score < self.alert_thresholds.psi_medium:
                drift_level = "medium"
            else:
                drift_level = "high"
            
            # Create drift metrics
            metrics = DriftMetrics(
                feature_name=feature_name,
                psi_score=psi_score,
                drift_level=drift_level,
                baseline_mean=baseline_mean,
                current_mean=float(current_values.mean()),
                baseline_std=baseline_std,
                current_std=float(current_values.std()),
                sample_size=len(current_values),
                calculated_at=datetime.now(timezone.utc)
            )
            
            drift_metrics.append(metrics)
            
            # Log alerts for high drift
            if drift_level in ["medium", "high"]:
                self.logger.warning(
                    f"Feature drift detected for {feature_name}: "
                    f"PSI={psi_score:.4f} ({drift_level})"
                )
        
        return drift_metrics
    
    def calculate_signal_attribution(
        self,
        signal_name: str,
        signal_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> SignalAttribution:
        """
        Calculate performance attribution for a signal.
        
        Args:
            signal_name: Name of the signal
            signal_returns: Signal return series
            benchmark_returns: Benchmark return series
            period_start: Attribution period start
            period_end: Attribution period end
            
        Returns:
            SignalAttribution metrics
        """
        if period_start is None:
            period_start = signal_returns.index[0] if len(signal_returns) > 0 else datetime.now(timezone.utc)
        
        if period_end is None:
            period_end = signal_returns.index[-1] if len(signal_returns) > 0 else datetime.now(timezone.utc)
        
        if len(signal_returns) == 0:
            return SignalAttribution(
                signal_name=signal_name,
                period_return=0.0,
                hit_ratio=0.0,
                sharpe_ratio=0.0,
                turnover=0.0,
                max_drawdown=0.0,
                trade_count=0,
                avg_hold_period=0.0,
                total_pnl=0.0,
                period_start=period_start,
                period_end=period_end
            )
        
        # Calculate metrics
        total_return = (1 + signal_returns).prod() - 1
        hit_ratio = (signal_returns > 0).mean()
        
        # Sharpe ratio
        if signal_returns.std() > 0:
            sharpe_ratio = signal_returns.mean() / signal_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = (1 + signal_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Trade count (rough estimate based on sign changes)
        signal_positions = np.sign(signal_returns)
        position_changes = signal_positions.diff().abs()
        trade_count = int(position_changes.sum() / 2)  # Round trips
        
        # Average holding period
        if trade_count > 0:
            avg_hold_period = len(signal_returns) / trade_count
        else:
            avg_hold_period = len(signal_returns)
        
        # Turnover (simplified as position change frequency)
        turnover = position_changes.mean()
        
        # Total P&L (cumulative return)
        total_pnl = total_return
        
        attribution = SignalAttribution(
            signal_name=signal_name,
            period_return=total_return,
            hit_ratio=hit_ratio,
            sharpe_ratio=sharpe_ratio,
            turnover=turnover,
            max_drawdown=max_drawdown,
            trade_count=trade_count,
            avg_hold_period=avg_hold_period,
            total_pnl=total_pnl,
            period_start=period_start,
            period_end=period_end
        )
        
        # Check for alerts
        self._check_attribution_alerts(attribution)
        
        return attribution
    
    def _check_attribution_alerts(self, attribution: SignalAttribution) -> None:
        """Check for performance alerts."""
        alerts = []
        
        if attribution.hit_ratio < self.alert_thresholds.hit_ratio_min:
            alerts.append(f"Low hit ratio: {attribution.hit_ratio:.2%}")
        
        if attribution.sharpe_ratio < self.alert_thresholds.sharpe_ratio_min:
            alerts.append(f"Low Sharpe ratio: {attribution.sharpe_ratio:.2f}")
        
        if attribution.max_drawdown > self.alert_thresholds.drawdown_max:
            alerts.append(f"High drawdown: {attribution.max_drawdown:.2%}")
        
        if alerts:
            self.logger.warning(
                f"Performance alerts for {attribution.signal_name}: {'; '.join(alerts)}"
            )
    
    def save_attribution_history(self, attribution: SignalAttribution) -> None:
        """Save signal attribution to history."""
        signal_name = attribution.signal_name
        
        if signal_name not in self._signal_history:
            self._signal_history[signal_name] = []
        
        self._signal_history[signal_name].append(attribution)
        
        # Save to file
        history_file = self.baseline_path / f"signal_attribution_{signal_name}.json"
        
        try:
            attribution_dicts = [attr.to_dict() for attr in self._signal_history[signal_name]]
            
            with open(history_file, 'w') as f:
                json.dump(attribution_dicts, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save attribution history: {e}")
    
    def get_drift_summary(self, drift_metrics: List[DriftMetrics]) -> Dict[str, Any]:
        """Get summary of drift monitoring results."""
        if not drift_metrics:
            return {}
        
        drift_levels = [m.drift_level for m in drift_metrics]
        psi_scores = [m.psi_score for m in drift_metrics]
        
        return {
            'total_features': len(drift_metrics),
            'high_drift_count': drift_levels.count('high'),
            'medium_drift_count': drift_levels.count('medium'),
            'low_drift_count': drift_levels.count('low'),
            'avg_psi': np.mean(psi_scores),
            'max_psi': np.max(psi_scores),
            'features_with_alerts': [m.feature_name for m in drift_metrics if m.drift_level in ['medium', 'high']]
        }


class ShadowMode:
    """
    Shadow mode for experimental model evaluation.
    
    Runs experimental models side-by-side with production models
    without affecting live trading.
    """
    
    def __init__(self, log_path: str = "artifacts/monitoring/shadow"):
        """
        Initialize shadow mode.
        
        Args:
            log_path: Path to shadow mode logs
        """
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def evaluate_shadow_model(
        self,
        model_name: str,
        shadow_predictions: Dict[str, float],
        production_predictions: Dict[str, float],
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate shadow model against production model.
        
        Args:
            model_name: Name of shadow model
            shadow_predictions: Shadow model predictions
            production_predictions: Production model predictions
            market_data: Market context data
            
        Returns:
            Evaluation results
        """
        timestamp = datetime.now(timezone.utc)
        
        # Calculate prediction differences
        common_symbols = set(shadow_predictions.keys()) & set(production_predictions.keys())
        
        if not common_symbols:
            self.logger.warning("No common symbols between shadow and production predictions")
            return {}
        
        differences = {}
        for symbol in common_symbols:
            diff = shadow_predictions[symbol] - production_predictions[symbol]
            differences[symbol] = diff
        
        # Calculate metrics
        diff_values = list(differences.values())
        evaluation = {
            'model_name': model_name,
            'timestamp': timestamp.isoformat(),
            'num_predictions': len(common_symbols),
            'avg_difference': np.mean(diff_values),
            'max_difference': np.max(np.abs(diff_values)),
            'std_difference': np.std(diff_values),
            'correlation': np.corrcoef(
                [shadow_predictions[s] for s in common_symbols],
                [production_predictions[s] for s in common_symbols]
            )[0, 1] if len(common_symbols) > 1 else 0.0,
            'shadow_predictions': shadow_predictions,
            'production_predictions': production_predictions,
            'differences': differences
        }
        
        # Add market context if provided
        if market_data:
            evaluation['market_context'] = market_data
        
        # Log evaluation
        self._log_shadow_evaluation(evaluation)
        
        return evaluation
    
    def _log_shadow_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """Log shadow model evaluation."""
        log_file = self.log_path / f"shadow_{evaluation['model_name']}.jsonl"
        
        try:
            # Append to JSONL file
            with open(log_file, 'a') as f:
                f.write(json.dumps(evaluation) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log shadow evaluation: {e}")


# Global instances
_global_drift_monitor: Optional[DriftMonitor] = None
_global_shadow_mode: Optional[ShadowMode] = None


def get_drift_monitor() -> DriftMonitor:
    """Get or create global drift monitor instance."""
    global _global_drift_monitor
    if _global_drift_monitor is None:
        _global_drift_monitor = DriftMonitor()
    return _global_drift_monitor


def get_shadow_mode() -> ShadowMode:
    """Get or create global shadow mode instance."""
    global _global_shadow_mode
    if _global_shadow_mode is None:
        _global_shadow_mode = ShadowMode()
    return _global_shadow_mode


def monitor_drift(current_features: pd.DataFrame) -> List[DriftMetrics]:
    """
    Convenience function to monitor feature drift.
    
    Args:
        current_features: Current feature dataset
        
    Returns:
        List of drift metrics
    """
    monitor = get_drift_monitor()
    return monitor.monitor_feature_drift(current_features)