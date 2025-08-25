"""
Walk-forward analysis for trading model evaluation.

Provides rolling and anchored walk-forward evaluation with proper
time series validation and comprehensive performance reporting.
"""
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any, TYPE_CHECKING
import numpy as np
from ai_trading.logging import logger
if TYPE_CHECKING:  # pragma: no cover - only for type hints
    import pandas as pd
matplotlib_available = False
plt = None
mdates = None

def _ensure_matplotlib() -> None:
    """Lazy import matplotlib if plotting is enabled."""
    global matplotlib_available, plt, mdates
    if matplotlib_available:
        return
    from ai_trading.config import get_settings
    if getattr(get_settings(), 'enable_plotting', False):
        import matplotlib.dates as mdates_module
        import matplotlib.pyplot as plt_module
        mdates = mdates_module
        plt = plt_module
        matplotlib_available = True
    else:
        logger.debug('Matplotlib plotting disabled by configuration')
from ..data.splits import walkforward_splits
from ..features.pipeline import create_feature_pipeline

def _get_ml_trainer():
    from ..training.train_ml import MLTrainer
    return MLTrainer

class WalkForwardEvaluator:
    """
    Walk-forward analysis evaluator for trading models.

    Implements both rolling and anchored walk-forward validation
    with comprehensive performance tracking and visualization.
    """

    def __init__(self, mode: str='rolling', train_span: int | timedelta=252, test_span: int | timedelta=21, embargo_pct: float=0.01, output_dir: str | None=None):
        """
        Initialize walk-forward evaluator.

        Args:
            mode: 'rolling' or 'anchored'
            train_span: Training window size
            test_span: Testing window size
            embargo_pct: Percentage embargo between train/test
            output_dir: Output directory (overridable via ARTIFACTS_DIR env var)
        """
        self.mode = mode
        self.train_span = train_span
        self.test_span = test_span
        self.embargo_pct = embargo_pct
        if output_dir is None:
            base = os.getenv('ARTIFACTS_DIR', 'artifacts')
            self.output_dir = os.path.join(base, 'walkforward')
        else:
            self.output_dir = output_dir
        import pandas as pd

        self.fold_results = []
        self.aggregate_results = {}
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_series = pd.Series(dtype=float)
        os.makedirs(self.output_dir, exist_ok=True)

    def run_walkforward(self, data: 'pd.DataFrame', target_col: str, feature_cols: list[str] | None=None, model_type: str='lightgbm', feature_pipeline_params: dict[str, Any] | None=None, save_results: bool=True) -> dict[str, Any]:
        """
        Run walk-forward analysis.

        Args:
            data: Complete dataset with features and target
            target_col: Name of target column
            feature_cols: List of feature columns (all others if None)
            model_type: Type of model to train
            feature_pipeline_params: Parameters for feature pipeline
            save_results: Whether to save results to disk

        Returns:
            Comprehensive walk-forward results
        """
        try:
            import pandas as pd

            logger.info(f'Starting {self.mode} walk-forward analysis')
            if feature_cols is None:
                feature_cols = [col for col in data.columns if col != target_col]
            X = data[feature_cols]
            y = data[target_col]
            splits = walkforward_splits(dates=data.index, mode=self.mode, train_span=self.train_span, test_span=self.test_span, embargo_pct=self.embargo_pct)
            logger.info(f'Generated {len(splits)} walk-forward periods')
            self.fold_results = []
            predictions_all = []
            actual_all = []
            equity_values = [100.0]
            for fold_idx, split_info in enumerate(splits):
                logger.debug(f'Processing fold {fold_idx + 1}/{len(splits)}')
                fold_result = self._run_single_fold(X, y, split_info, model_type, feature_pipeline_params, fold_idx)
                self.fold_results.append(fold_result)
                if 'predictions' in fold_result and 'actual' in fold_result:
                    predictions_all.extend(fold_result['predictions'])
                    actual_all.extend(fold_result['actual'])
                    if len(fold_result['predictions']) > 0:
                        fold_return = np.mean(fold_result['predictions']) * 0.01
                        new_equity = equity_values[-1] * (1 + fold_return)
                        equity_values.append(new_equity)
            self.aggregate_results = self._calculate_aggregate_metrics(predictions_all, actual_all, splits)
            equity_dates = [split['test_start'] for split in splits] + [splits[-1]['test_end']]
            self.equity_curve = pd.Series(equity_values, index=equity_dates[:len(equity_values)])
            self.drawdown_series = self._calculate_drawdown(self.equity_curve)
            if save_results:
                self._save_results()
            results = {'mode': self.mode, 'n_folds': len(splits), 'aggregate_metrics': self.aggregate_results, 'fold_results': self.fold_results, 'equity_curve': self.equity_curve.to_dict(), 'max_drawdown': self.drawdown_series.max(), 'final_equity': equity_values[-1], 'total_return': equity_values[-1] / equity_values[0] - 1}
            logger.info(f"Walk-forward analysis completed. Final return: {results['total_return']:.2%}")
            return results
        except (ValueError, TypeError) as e:
            logger.error(f'Error in walk-forward analysis: {e}')
            raise

    def _run_single_fold(self, X: 'pd.DataFrame', y: 'pd.Series', split_info: dict[str, Any], model_type: str, feature_pipeline_params: dict[str, Any] | None, fold_idx: int) -> dict[str, Any]:
        """Run single fold of walk-forward analysis."""
        try:
            train_start = split_info['train_start']
            train_end = split_info['train_end']
            test_start = split_info['test_start']
            test_end = split_info['test_end']
            train_mask = (X.index >= train_start) & (X.index < train_end)
            X_train = X[train_mask]
            y_train = y[train_mask]
            test_mask = (X.index >= test_start) & (X.index < test_end)
            X_test = X[test_mask]
            y_test = y[test_mask]
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f'Fold {fold_idx}: insufficient data')
                return {'error': 'insufficient_data'}
            if feature_pipeline_params:
                feature_pipeline = create_feature_pipeline(**feature_pipeline_params)
            else:
                feature_pipeline = None
            MLTrainer = _get_ml_trainer()
            trainer = MLTrainer(model_type=model_type, cv_splits=3, random_state=42 + fold_idx)
            trainer.train(X_train, y_train, optimize_hyperparams=False, feature_pipeline=feature_pipeline)
            if feature_pipeline:
                X_test_processed = feature_pipeline.transform(X_test)
            else:
                X_test_processed = X_test
            predictions = trainer.model.predict(X_test_processed)
            fold_metrics = self._calculate_fold_metrics(y_test, predictions, test_start, test_end)
            fold_result = {'fold': fold_idx, 'train_start': train_start.isoformat(), 'train_end': train_end.isoformat(), 'test_start': test_start.isoformat(), 'test_end': test_end.isoformat(), 'train_samples': len(X_train), 'test_samples': len(X_test), 'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions), 'actual': y_test.values.tolist(), 'metrics': fold_metrics, 'model_params': trainer.best_params}
            return fold_result
        except (ValueError, TypeError) as e:
            logger.error(f'Error in fold {fold_idx}: {e}')
            return {'fold': fold_idx, 'error': str(e)}

    def _calculate_fold_metrics(self, y_true: 'pd.Series', y_pred: np.ndarray, test_start: datetime, test_end: datetime) -> dict[str, float]:
        """Calculate performance metrics for a single fold."""
        try:
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_pred) > 1 else 0.0
            directional_accuracy = np.mean(np.sign(y_pred) == np.sign(y_true))
            if np.std(y_pred) > 0:
                information_ratio = np.mean(y_pred) / np.std(y_pred)
            else:
                information_ratio = 0.0
            long_signals = y_pred > 0
            short_signals = y_pred < 0
            hit_rate_long = np.mean(y_true[long_signals] > 0) if np.any(long_signals) else 0.0
            hit_rate_short = np.mean(y_true[short_signals] < 0) if np.any(short_signals) else 0.0
            return {'mse': float(mse), 'mae': float(mae), 'correlation': float(correlation) if not np.isnan(correlation) else 0.0, 'directional_accuracy': float(directional_accuracy), 'information_ratio': float(information_ratio), 'hit_rate_long': float(hit_rate_long), 'hit_rate_short': float(hit_rate_short), 'n_long_signals': int(np.sum(long_signals)), 'n_short_signals': int(np.sum(short_signals)), 'period_days': (test_end - test_start).days}
        except (ValueError, TypeError) as e:
            logger.error(f'Error calculating fold metrics: {e}')
            return {}

    def _calculate_aggregate_metrics(self, predictions_all: list[float], actual_all: list[float], splits: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate aggregate metrics across all folds."""
        try:
            if not predictions_all or not actual_all:
                return {}
            predictions = np.array(predictions_all)
            actual = np.array(actual_all)
            mse = np.mean((actual - predictions) ** 2)
            mae = np.mean(np.abs(actual - predictions))
            correlation = np.corrcoef(actual, predictions)[0, 1] if len(predictions) > 1 else 0.0
            directional_accuracy = np.mean(np.sign(predictions) == np.sign(actual))
            if np.std(predictions) > 0:
                sharpe_like = np.mean(predictions) / np.std(predictions) * np.sqrt(252)
            else:
                sharpe_like = 0.0
            downside_returns = predictions[predictions < 0]
            if len(downside_returns) > 0:
                sortino_ratio = np.mean(predictions) / np.std(downside_returns) * np.sqrt(252)
            else:
                sortino_ratio = sharpe_like
            max_drawdown = self.drawdown_series.max() if len(self.drawdown_series) > 0 else 0.0
            annual_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (252 / len(splits)) - 1
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
            n_predictions = len(predictions)
            total_days = sum((split['period_days'] for split in self.fold_results if 'metrics' in split))
            turnover = n_predictions / max(1, total_days) * 252
            aggregate_metrics = {'net_sharpe': float(sharpe_like), 'sortino_ratio': float(sortino_ratio), 'calmar_ratio': float(calmar_ratio), 'max_drawdown': float(max_drawdown), 'turnover_annual': float(turnover), 'correlation': float(correlation) if not np.isnan(correlation) else 0.0, 'directional_accuracy': float(directional_accuracy), 'mse': float(mse), 'mae': float(mae), 'total_predictions': int(n_predictions), 'evaluation_period_days': int(total_days)}
            return aggregate_metrics
        except (ValueError, TypeError) as e:
            logger.error(f'Error calculating aggregate metrics: {e}')
            return {}

    def _calculate_drawdown(self, equity_curve: 'pd.Series') -> 'pd.Series':
        """Calculate drawdown series from equity curve."""
        try:
            import pandas as pd

            if len(equity_curve) == 0:
                return pd.Series(dtype=float)
            running_max = equity_curve.expanding().max()
            drawdown = (running_max - equity_curve) / running_max
            return drawdown
        except (ValueError, TypeError) as e:
            logger.error(f'Error calculating drawdown: {e}')
            import pandas as pd

            return pd.Series(dtype=float)

    def _save_results(self) -> None:
        """Save walk-forward results to disk."""
        try:
            import pandas as pd

            timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
            if self.fold_results:
                fold_df = pd.DataFrame([
                    {
                        'fold': result.get('fold', 0),
                        'train_start': result.get('train_start', ''),
                        'test_start': result.get('test_start', ''),
                        'train_samples': result.get('train_samples', 0),
                        'test_samples': result.get('test_samples', 0),
                        **result.get('metrics', {}),
                    }
                    for result in self.fold_results
                    if 'metrics' in result
                ])
                fold_file = os.path.join(self.output_dir, f'walkforward_folds_{timestamp}.csv')
                fold_df.to_csv(fold_file, index=False)
                logger.info(f'Fold results saved to {fold_file}')
            if self.aggregate_results:
                agg_file = os.path.join(self.output_dir, f'walkforward_aggregate_{timestamp}.json')
                with open(agg_file, 'w') as f:
                    json.dump(self.aggregate_results, f, indent=2, default=str)
                logger.info(f'Aggregate results saved to {agg_file}')
            if len(self.equity_curve) > 0:
                equity_file = os.path.join(self.output_dir, f'equity_curve_{timestamp}.csv')
                self.equity_curve.to_csv(equity_file, header=['equity'])
                logger.info(f'Equity curve saved to {equity_file}')
            self._create_plots(timestamp)
        except (ValueError, TypeError) as e:
            logger.error(f'Error saving results: {e}')

    def _create_plots(self, timestamp: str) -> None:
        """Create and save visualization plots."""
        try:
            _ensure_matplotlib()
            if not matplotlib_available:
                logger.warning('Matplotlib not available - skipping plots')
                return
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Walk-Forward Analysis Results ({self.mode.title()})', fontsize=16)
            if len(self.equity_curve) > 0:
                ax1.plot(self.equity_curve.index, self.equity_curve.values, 'b-', linewidth=2)
                ax1.set_title('Equity Curve')
                ax1.set_ylabel('Portfolio Value')
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            if len(self.drawdown_series) > 0:
                ax2.fill_between(self.drawdown_series.index, 0, -self.drawdown_series.values, color='red', alpha=0.6)
                ax2.set_title('Drawdown')
                ax2.set_ylabel('Drawdown %')
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            if self.fold_results:
                fold_metrics = [r.get('metrics', {}) for r in self.fold_results if 'metrics' in r]
                if fold_metrics:
                    correlations = [m.get('correlation', 0) for m in fold_metrics]
                    ax3.plot(range(len(correlations)), correlations, 'g-o', markersize=4)
                    ax3.set_title('Correlation by Fold')
                    ax3.set_ylabel('Correlation')
                    ax3.set_xlabel('Fold')
                    ax3.grid(True, alpha=0.3)
                    dir_acc = [m.get('directional_accuracy', 0) for m in fold_metrics]
                    ax4.plot(range(len(dir_acc)), dir_acc, 'orange', marker='s', markersize=4)
                    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
                    ax4.set_title('Directional Accuracy by Fold')
                    ax4.set_ylabel('Accuracy')
                    ax4.set_xlabel('Fold')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = os.path.join(self.output_dir, f'walkforward_plots_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f'Plots saved to {plot_file}')
        except (ValueError, TypeError) as e:
            logger.error(f'Error creating plots: {e}')

def run_walkforward_smoke_test() -> None:
    """Run a quick smoke test of walk-forward analysis."""
    try:
        import pandas as pd

        logger.info('Running walk-forward smoke test')
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)
        price_trend = np.linspace(100, 150, n_samples)
        price_noise = np.random.normal(0, 5, n_samples)
        prices = price_trend + price_noise
        data = pd.DataFrame({'price': prices, 'feature_1': np.random.randn(n_samples), 'feature_2': np.random.randn(n_samples), 'feature_3': prices * 0.01 + np.random.randn(n_samples) * 0.1, 'target': np.random.randn(n_samples) * 0.02}, index=dates)
        data['target'] = data['target'] + data['feature_3'] * 0.5
        evaluator = WalkForwardEvaluator(mode='rolling', train_span=timedelta(days=180), test_span=timedelta(days=30))
        results = evaluator.run_walkforward(data=data, target_col='target', feature_cols=['feature_1', 'feature_2', 'feature_3'], model_type='ridge', save_results=True)
        logger.info('Smoke test completed successfully:')
        logger.info(f"  - Folds: {results['n_folds']}")
        logger.info(f"  - Final return: {results['total_return']:.2%}")
        logger.info(f"  - Max drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"  - Net Sharpe: {results['aggregate_metrics'].get('net_sharpe', 'N/A')}")
        return results
    except (ValueError, TypeError) as e:
        logger.error(f'Error in smoke test: {e}')
        raise
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Walk-forward evaluation')
    parser.add_argument('--smoke', action='store_true', help='Run smoke test')
    args = parser.parse_args()
    if args.smoke:
        run_walkforward_smoke_test()
    else:
        logger.info('Use --smoke to run smoke test')