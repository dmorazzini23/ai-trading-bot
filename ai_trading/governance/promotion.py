"""
Model promotion pipeline for shadow-to-production governance.

Manages the promotion process from shadow testing to production deployment
with performance validation and safety checks.
"""
import json
from ai_trading.logging import get_logger
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from ..model_registry import ModelRegistry
logger = get_logger(__name__)

@dataclass
class PromotionCriteria:
    """Criteria for model promotion from shadow to production."""
    min_shadow_sessions: int = 5
    max_turnover_ratio: float = 1.5
    min_live_sharpe: float = 0.5
    max_drift_psi: float = 0.25
    min_shadow_days: int = 3
    max_drawdown_threshold: float = 0.05
    min_trade_count: int = 10

@dataclass
class PromotionMetrics:
    """Metrics collected during shadow testing."""
    sessions_completed: int = 0
    total_trades: int = 0
    turnover_ratio: float = 0.0
    live_sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    drift_psi: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime | None = None

class ModelPromotion:
    """
    Model promotion manager for shadow-to-production workflow.

    Handles shadow testing, performance validation, and automatic
    promotion based on defined criteria.
    """

    def __init__(self, model_registry: ModelRegistry | None=None, criteria: PromotionCriteria | None=None, base_path: str='artifacts/governance'):
        """
        Initialize model promotion manager.

        Args:
            model_registry: Model registry instance
            criteria: Promotion criteria
            base_path: Base path for governance artifacts
        """
        self.registry = model_registry or ModelRegistry()
        self.criteria = criteria or PromotionCriteria()
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self.active_dir = self.base_path / 'active'
        self.active_dir.mkdir(exist_ok=True)

    def start_shadow_testing(self, model_id: str, benchmark_model_id: str | None=None) -> bool:
        """
        Start shadow testing for a model.

        Args:
            model_id: Model ID to put in shadow mode
            benchmark_model_id: Optional benchmark model for comparison

        Returns:
            True if shadow testing started successfully
        """
        try:
            model_info = self.registry.model_index.get(model_id)
            if model_info is None:
                raise ValueError(f'Model {model_id} not found')
            strategy = model_info['strategy']
            shadow_models = self.registry.get_shadow_models(strategy)
            if shadow_models:
                self.logger.warning(f'Strategy {strategy} already has shadow models: {[m[0] for m in shadow_models]}')
            self.registry.update_governance_status(model_id, 'shadow')
            shadow_metrics = PromotionMetrics(last_updated=datetime.now(UTC))
            self._save_shadow_metrics(model_id, shadow_metrics)
            self.logger.info(f'Started shadow testing for model {model_id} (strategy: {strategy})')
            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error starting shadow testing for {model_id}: {e}')
            return False

    def update_shadow_metrics(self, model_id: str, session_stats: dict[str, Any]) -> None:
        """
        Update shadow testing metrics.

        Args:
            model_id: Model ID in shadow testing
            session_stats: Statistics from latest trading session
        """
        try:
            current_metrics = self._load_shadow_metrics(model_id)
            if current_metrics is None:
                current_metrics = PromotionMetrics()
            current_metrics.sessions_completed += 1
            current_metrics.total_trades += session_stats.get('trade_count', 0)
            current_metrics.last_updated = datetime.now(UTC)
            alpha = 0.1
            new_turnover = session_stats.get('turnover_ratio', 0.0)
            current_metrics.turnover_ratio = alpha * new_turnover + (1 - alpha) * current_metrics.turnover_ratio
            new_sharpe = session_stats.get('sharpe_ratio', 0.0)
            current_metrics.live_sharpe_ratio = alpha * new_sharpe + (1 - alpha) * current_metrics.live_sharpe_ratio
            new_drawdown = session_stats.get('max_drawdown', 0.0)
            current_metrics.max_drawdown = max(current_metrics.max_drawdown, new_drawdown)
            new_psi = session_stats.get('drift_psi', 0.0)
            current_metrics.drift_psi = alpha * new_psi + (1 - alpha) * current_metrics.drift_psi
            new_latency = session_stats.get('avg_latency_ms', 0.0)
            current_metrics.avg_latency_ms = alpha * new_latency + (1 - alpha) * current_metrics.avg_latency_ms
            new_error_rate = session_stats.get('error_rate', 0.0)
            current_metrics.error_rate = alpha * new_error_rate + (1 - alpha) * current_metrics.error_rate
            self._save_shadow_metrics(model_id, current_metrics)
            metrics_dict = {'sessions_completed': current_metrics.sessions_completed, 'total_trades': current_metrics.total_trades, 'turnover_ratio': current_metrics.turnover_ratio, 'live_sharpe_ratio': current_metrics.live_sharpe_ratio, 'max_drawdown': current_metrics.max_drawdown, 'drift_psi': current_metrics.drift_psi, 'last_updated': current_metrics.last_updated.isoformat() if current_metrics.last_updated else None}
            self.registry.update_governance_status(model_id, 'shadow', metrics_dict)
            self.logger.debug(f'Updated shadow metrics for model {model_id}: {current_metrics.sessions_completed} sessions')
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error updating shadow metrics for {model_id}: {e}')

    def check_promotion_eligibility(self, model_id: str) -> tuple[bool, dict[str, Any]]:
        """
        Check if model is eligible for promotion.

        Args:
            model_id: Model ID to check

        Returns:
            Tuple of (eligible, evaluation_details)
        """
        try:
            metrics = self._load_shadow_metrics(model_id)
            if metrics is None:
                return (False, {'error': 'No shadow metrics found'})
            model_data = self.registry.load_model(model_id, verify_dataset_hash=False)[1]
            governance = model_data.get('governance', {})
            shadow_start = governance.get('shadow_start_time')
            if shadow_start:
                shadow_start_dt = datetime.fromisoformat(shadow_start.replace('Z', '+00:00'))
                days_in_shadow = (datetime.now(UTC) - shadow_start_dt).days
            else:
                days_in_shadow = 0
            checks = {'min_sessions': metrics.sessions_completed >= self.criteria.min_shadow_sessions, 'min_days': days_in_shadow >= self.criteria.min_shadow_days, 'min_trades': metrics.total_trades >= self.criteria.min_trade_count, 'turnover_check': metrics.turnover_ratio <= self.criteria.max_turnover_ratio, 'sharpe_check': metrics.live_sharpe_ratio >= self.criteria.min_live_sharpe, 'drawdown_check': metrics.max_drawdown <= self.criteria.max_drawdown_threshold, 'drift_check': metrics.drift_psi <= self.criteria.max_drift_psi}
            eligible = all(checks.values())
            evaluation = {'eligible': eligible, 'checks': checks, 'metrics': {'sessions_completed': metrics.sessions_completed, 'days_in_shadow': days_in_shadow, 'total_trades': metrics.total_trades, 'turnover_ratio': metrics.turnover_ratio, 'live_sharpe_ratio': metrics.live_sharpe_ratio, 'max_drawdown': metrics.max_drawdown, 'drift_psi': metrics.drift_psi}, 'criteria': {'min_sessions': self.criteria.min_shadow_sessions, 'min_days': self.criteria.min_shadow_days, 'min_trades': self.criteria.min_trade_count, 'max_turnover': self.criteria.max_turnover_ratio, 'min_sharpe': self.criteria.min_live_sharpe, 'max_drawdown': self.criteria.max_drawdown_threshold, 'max_drift': self.criteria.max_drift_psi}}
            return (eligible, evaluation)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error checking promotion eligibility for {model_id}: {e}')
            return (False, {'error': str(e)})

    def promote_to_production(self, model_id: str, force: bool=False) -> bool:
        """
        Promote model to production.

        Args:
            model_id: Model ID to promote
            force: Force promotion without eligibility check

        Returns:
            True if promotion successful
        """
        try:
            if not force:
                eligible, details = self.check_promotion_eligibility(model_id)
                if not eligible:
                    self.logger.warning(f'Model {model_id} not eligible for promotion: {details}')
                    return False
            model_info = self.registry.model_index.get(model_id)
            if model_info is None:
                raise ValueError(f'Model {model_id} not found')
            strategy = model_info['strategy']
            current_production = self.registry.get_production_model(strategy)
            if current_production:
                old_model_id, _ = current_production
                self.registry.update_governance_status(old_model_id, 'registered')
                self._remove_active_symlink(strategy)
                self.logger.info(f'Demoted previous production model {old_model_id} for strategy {strategy}')
            self.registry.update_governance_status(model_id, 'production')
            self._create_active_symlink(strategy, model_id)
            self.logger.info(f'Promoted model {model_id} to production for strategy {strategy}')
            return True
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error promoting model {model_id}: {e}')
            return False

    def _save_shadow_metrics(self, model_id: str, metrics: PromotionMetrics) -> None:
        """Save shadow metrics to disk."""
        metrics_file = self.base_path / f'{model_id}_shadow_metrics.json'
        metrics_dict = {'sessions_completed': metrics.sessions_completed, 'total_trades': metrics.total_trades, 'turnover_ratio': metrics.turnover_ratio, 'live_sharpe_ratio': metrics.live_sharpe_ratio, 'max_drawdown': metrics.max_drawdown, 'drift_psi': metrics.drift_psi, 'avg_latency_ms': metrics.avg_latency_ms, 'error_rate': metrics.error_rate, 'last_updated': metrics.last_updated.isoformat() if metrics.last_updated else None}
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def _load_shadow_metrics(self, model_id: str) -> PromotionMetrics | None:
        """Load shadow metrics from disk."""
        metrics_file = self.base_path / f'{model_id}_shadow_metrics.json'
        if not metrics_file.exists():
            return None
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            last_updated = None
            if data.get('last_updated'):
                last_updated = datetime.fromisoformat(data['last_updated'].replace('Z', '+00:00'))
            return PromotionMetrics(sessions_completed=data.get('sessions_completed', 0), total_trades=data.get('total_trades', 0), turnover_ratio=data.get('turnover_ratio', 0.0), live_sharpe_ratio=data.get('live_sharpe_ratio', 0.0), max_drawdown=data.get('max_drawdown', 0.0), drift_psi=data.get('drift_psi', 0.0), avg_latency_ms=data.get('avg_latency_ms', 0.0), error_rate=data.get('error_rate', 0.0), last_updated=last_updated)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error loading shadow metrics for {model_id}: {e}')
            return None

    def _create_active_symlink(self, strategy: str, model_id: str) -> None:
        """Create symlink to active production model."""
        try:
            model_info = self.registry.model_index[model_id]
            model_path = Path(model_info['path'])
            symlink_path = self.active_dir / f'{strategy}_active'
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(model_path.absolute())
            self.logger.debug(f'Created active symlink for {strategy}: {symlink_path} -> {model_path}')
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error creating active symlink for {strategy}: {e}')

    def _remove_active_symlink(self, strategy: str) -> None:
        """Remove active symlink for strategy."""
        try:
            symlink_path = self.active_dir / f'{strategy}_active'
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
                self.logger.debug(f'Removed active symlink for {strategy}')
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error removing active symlink for {strategy}: {e}')

    def get_active_model_path(self, strategy: str) -> str | None:
        """Get path to active production model."""
        symlink_path = self.active_dir / f'{strategy}_active'
        if symlink_path.exists() and symlink_path.is_symlink():
            return str(symlink_path.resolve())
        return None

    def list_shadow_models(self) -> list[dict[str, Any]]:
        """List all models currently in shadow testing."""
        shadow_models = []
        for model_id, info in self.registry.model_index.items():
            if not info.get('active', True):
                continue
            try:
                metadata_file = Path(info['path']) / 'meta.json'
                with open(metadata_file) as f:
                    metadata = json.load(f)
                governance = metadata.get('governance', {})
                if governance.get('status') == 'shadow':
                    shadow_metrics = self._load_shadow_metrics(model_id)
                    shadow_info = {'model_id': model_id, 'strategy': info['strategy'], 'shadow_start': governance.get('shadow_start_time'), 'metrics': shadow_metrics}
                    eligible, details = self.check_promotion_eligibility(model_id)
                    shadow_info['promotion_eligible'] = eligible
                    shadow_info['promotion_details'] = details
                    shadow_models.append(shadow_info)
            except (ValueError, TypeError) as e:
                self.logger.debug(f'Error checking shadow status for model {model_id}: {e}')
                continue
        return shadow_models
_global_promotion_manager: ModelPromotion | None = None

def get_promotion_manager() -> ModelPromotion:
    """Get or create global promotion manager."""
    global _global_promotion_manager
    if _global_promotion_manager is None:
        _global_promotion_manager = ModelPromotion()
    return _global_promotion_manager