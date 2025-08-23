from importlib.util import find_spec
from ai_trading.config import get_settings
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
'Utility helpers for meta-learning weight management.'
import csv
import json
import logging
import os
import pickle
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
import pandas as pd
from json import JSONDecodeError
try:
    import requests
    RequestException = requests.exceptions.RequestException
except ImportError:

    class RequestException(Exception):
        pass
COMMON_EXC = (TypeError, ValueError, KeyError, JSONDecodeError, RequestException, TimeoutError, ImportError)
sys.modules.setdefault('meta_learning', sys.modules[__name__])
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    nn = None
    DataLoader = TensorDataset = None
    TORCH_AVAILABLE = False
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
open = open
logger = logging.getLogger(__name__)

def get_device() -> str:
    """
    Honor CPU_ONLY and handle environments without torch.
    """
    if os.getenv('CPU_ONLY') == '1' or not TORCH_AVAILABLE:
        return 'cpu'
    try:
        return 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    except COMMON_EXC:
        return 'cpu'

class SimpleMetaLearner:
    """
    Lightweight wrapper so the module can import without torch.
    When TORCH_AVAILABLE is False, construction raises on actual use,
    but mere import of the module succeeds.
    """

    def __init__(self, input_dim: int, hidden: int=32):
        if not TORCH_AVAILABLE:
            raise RuntimeError('Meta-learner requires PyTorch; TORCH_AVAILABLE=False')

        class _Net(nn.Module):

            def __init__(self, d: int, h: int):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1), nn.Sigmoid())

            def forward(self, x):
                return self.net(x)
        self.impl = _Net(input_dim, hidden)

    def forward(self, x):
        return self.impl.forward(x)

def train_meta_learner(X, y, *, epochs: int=20, lr: float=0.001):
    """
    Train only if TORCH_AVAILABLE; otherwise raise a clear error on call.
    Importing the module remains safe.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError('train_meta_learner requires PyTorch; not installed')
    device = get_device()
    model = SimpleMetaLearner(X.shape[1]).impl.to(device)
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = (batch_X.to(device), batch_y.to(device))
            preds = model(batch_X).view(-1)
            loss = loss_fn(preds, batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model

class MetaLearning:
    """Meta-learning class for trading strategy optimization."""

    def __init__(self):
        """Initialize the MetaLearning instance."""
        self.logger = logging.getLogger(__name__ + '.MetaLearning')
        self.logger.debug('MetaLearning instance initialized')

def validate_trade_data_quality(trade_log_path: str) -> dict:
    """Perform comprehensive data quality checks on trade log before meta-learning."""
    quality_report = {'file_exists': False, 'file_readable': False, 'has_valid_format': False, 'row_count': 0, 'valid_price_rows': 0, 'data_quality_score': 0.0, 'issues': [], 'recommendations': [], 'mixed_format_detected': False, 'audit_format_rows': 0, 'meta_format_rows': 0}
    if not isinstance(trade_log_path, str | os.PathLike):
        error_msg = f'trade_log_path must be a string or PathLike object, got {type(trade_log_path).__name__}'
        quality_report['issues'].append(error_msg)
        quality_report['recommendations'].append('Pass a valid file path string instead of dictionary or other object')
        logger.error('META_LEARNING_INPUT_VALIDATION_ERROR: %s', error_msg)
        return quality_report
    try:
        if not Path(trade_log_path).exists():
            quality_report['issues'].append(f'Trade log file does not exist: {trade_log_path}')
            quality_report['recommendations'].append('Initialize trade logging system')
            return quality_report
        quality_report['file_exists'] = True
        try:
            file_size = Path(trade_log_path).stat().st_size
            if file_size == 0:
                quality_report['issues'].append('Trade log file is empty')
                quality_report['recommendations'].append('Ensure trade logging is actively writing data')
                return quality_report
        except COMMON_EXC as e:
            quality_report['issues'].append(f'Cannot access file stats: {e}')
            return quality_report
        quality_report['file_readable'] = True
        if pd is None:
            quality_report['issues'].append('pandas not available for data validation')
            try:
                with open(trade_log_path, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    quality_report['issues'].append('Trade log file is empty')
                    return quality_report
                header_line = lines[0].strip()
                if 'price' in header_line.lower():
                    quality_report['has_valid_format'] = True
                    quality_report['meta_format_rows'] = len(lines) - 1
                    for line in lines[1:]:
                        try:
                            import csv
                            import io
                            csv_reader = csv.reader(io.StringIO(line))
                            row = next(csv_reader)
                            for col in row:
                                try:
                                    val = float(col)
                                    if val > 0:
                                        quality_report['valid_price_rows'] += 1
                                        break
                                except (ValueError, TypeError):
                                    continue
                        except COMMON_EXC:
                            continue
                    quality_report['row_count'] = len(lines) - 1
                    if quality_report['row_count'] > 0:
                        quality_report['data_quality_score'] = quality_report['valid_price_rows'] / quality_report['row_count']
                    return quality_report
            except COMMON_EXC as e:
                quality_report['issues'].append(f'Failed basic CSV validation: {e}')
            return quality_report
        try:
            with open(trade_log_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                quality_report['issues'].append('Trade log file is empty')
                quality_report['recommendations'].append('Ensure trade logging is actively writing data')
                return quality_report
            audit_format_rows = 0
            meta_format_rows = 0
            valid_price_rows = 0
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    import csv
                    import io
                    csv_reader = csv.reader(io.StringIO(line))
                    row = next(csv_reader)
                    if len(row) == 0:
                        continue
                    first_col = str(row[0]).strip()
                    if len(first_col) > 20 and '-' in first_col:
                        audit_format_rows += 1
                        if len(row) >= 6:
                            try:
                                price = float(row[5])
                                if price > 0:
                                    valid_price_rows += 1
                            except (ValueError, IndexError) as e:
                                logger.debug('Invalid price in audit format row: %s', e)
                    elif len(first_col) <= 10 and first_col.isalpha() and (len(first_col) >= 2):
                        meta_format_rows += 1
                        if len(row) >= 3:
                            try:
                                price = float(row[2])
                                if price > 0:
                                    valid_price_rows += 1
                            except (ValueError, IndexError) as e:
                                logger.debug('Invalid price in meta format row: %s', e)
                    elif 'price' in first_col.lower() or any(('price' in str(col).lower() for col in row[:3])):
                        meta_format_rows += 1
                    else:
                        found_numeric = False
                        for _col_idx, col_val in enumerate(row[:5]):
                            try:
                                price = float(col_val)
                                if price > 0:
                                    found_numeric = True
                                    break
                            except (ValueError, TypeError):
                                continue
                        if found_numeric:
                            valid_price_rows += 1
                except COMMON_EXC as e:
                    logger.debug(f'Failed to parse line {line_num}: {e}')
                    continue
            quality_report['row_count'] = len([l for l in lines if l.strip()])
            quality_report['audit_format_rows'] = audit_format_rows
            quality_report['meta_format_rows'] = meta_format_rows
            quality_report['valid_price_rows'] = valid_price_rows
            if audit_format_rows > 0 and meta_format_rows > 0:
                quality_report['mixed_format_detected'] = True
                quality_report['issues'].append(f'Mixed log formats detected: {audit_format_rows} audit rows, {meta_format_rows} meta rows')
                quality_report['recommendations'].append('Separate audit and meta-learning logs or implement unified parsing')
                quality_report['has_valid_format'] = True
            elif audit_format_rows > 0:
                quality_report['issues'].append('Only audit format detected - conversion needed for meta-learning')
                quality_report['recommendations'].append('Convert audit format to meta-learning format')
                quality_report['has_valid_format'] = True
            elif meta_format_rows > 0:
                quality_report['has_valid_format'] = True
            elif quality_report['row_count'] > 0:
                for line_num, line in enumerate(lines[:5]):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        import csv
                        import io
                        csv_reader = csv.reader(io.StringIO(line))
                        row = next(csv_reader)
                        found_numeric = False
                        for col in row:
                            try:
                                val = float(col)
                                if val > 0:
                                    found_numeric = True
                                    valid_price_rows += 1
                                    break
                            except (ValueError, TypeError):
                                continue
                        if found_numeric:
                            quality_report['has_valid_format'] = True
                            quality_report['meta_format_rows'] += 1
                    except COMMON_EXC:
                        continue
                if not quality_report['has_valid_format']:
                    quality_report['issues'].append('No recognizable format found')
                    quality_report['recommendations'].append('Check log format and data integrity')
            else:
                quality_report['issues'].append('No recognizable format found')
                quality_report['recommendations'].append('Check log format and data integrity')
        except COMMON_EXC as e:
            quality_report['issues'].append(f'Failed to read file: {e}')
            return quality_report
        if quality_report['row_count'] > 0:
            quality_score = quality_report['valid_price_rows'] / quality_report['row_count']
            quality_report['data_quality_score'] = quality_score
            if quality_score < 0.5:
                quality_report['issues'].append(f'Low data quality score: {quality_score:.2%}')
                quality_report['recommendations'].append('Investigate and fix data quality issues')
        if quality_report['valid_price_rows'] == 0:
            quality_report['issues'].append('No rows with valid positive prices found')
            quality_report['recommendations'].append('Check price data source and trade execution logging')
    except COMMON_EXC as e:
        logger.error('Failed reading trade log: %s', e, exc_info=True)
        quality_report['issues'].append(f'Error accessing file: {e}')
    return quality_report

def normalize_score(score: float, cap: float=1.2) -> float:
    """Clip ``score`` to ``cap`` preserving sign."""
    try:
        score = float(score)
    except (ValueError, TypeError):
        return 0.0
    return max(-cap, min(cap, score))

def adjust_confidence(confidence: float, volatility: float, threshold: float=1.0) -> float:
    """Scale confidence by inverse volatility to reduce spam at high levels."""
    try:
        conf = float(confidence)
        vol = float(volatility)
    except (ValueError, TypeError):
        return 0.0
    factor = 1.0 if vol <= threshold else 1.0 / max(vol, 0.001)
    return max(0.0, min(1.0, conf * factor))

def volatility_regime_filter(atr: float, sma100: float) -> str:
    """Return volatility regime string based on ATR and SMA."""
    from ai_trading.telemetry import metrics_logger
    if sma100 == 0:
        return 'unknown'
    ratio = atr / sma100
    regime = 'high_vol' if ratio > 0.05 else 'low_vol'
    metrics_logger.log_volatility(ratio)
    metrics_logger.log_regime_toggle('generic', regime)
    return regime

def load_weights(path: str, default: 'np.ndarray | None'=None) -> 'np.ndarray':
    """Load signal weights array from ``path`` or return ``default``."""
    if np is None:
        logger.warning('numpy not available, using basic weight loading')
        if default is None:
            return []
        return default
    p = Path(path)
    if default is None:
        default = np.zeros(0)
    try:
        if p.exists():
            try:
                if path.endswith('.csv'):
                    try:
                        import pandas as pd
                        df = pd.read_csv(p, usecols=['signal_name', 'weight'])
                        if not df.empty:
                            weights = df['weight'].values
                            if isinstance(weights, np.ndarray):
                                logger.info('Successfully loaded weights from CSV: %s', path)
                                return weights
                    except (ImportError, ValueError) as e:
                        logger.debug('Pandas CSV read failed, trying numpy: %s', e)
                weights = np.loadtxt(p, delimiter=',')
                if isinstance(weights, np.ndarray):
                    logger.debug('Loaded weights using numpy from: %s', path)
                    return weights
            except (ValueError, OSError) as e:
                logger.debug('CSV/numpy loading failed, trying pickle: %s', e)
                try:
                    with open(p, 'rb') as f:
                        weights = pickle.load(f)
                        if isinstance(weights, np.ndarray):
                            logger.debug('Loaded weights from pickle: %s', path)
                            return weights
                        else:
                            logger.warning('Invalid weights format in %s, using default', path)
                except (pickle.PickleError, EOFError) as pickle_e:
                    logger.warning('Pickle loading also failed for %s: %s', path, pickle_e)
        else:
            logger.debug('Weights file %s not found, creating with default', path)
            if default.size > 0:
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(p, default, delimiter=',')
                    logger.info('Created default weights file: %s', path)
                except COMMON_EXC as e:
                    logger.error('Failed initializing weights file %s: %s', path, e)
    except COMMON_EXC as e:
        logger.warning('Failed to load weights from %s: %s', path, e)
    return default

def update_weights(weight_path: str, new_weights: 'np.ndarray', metrics: dict, history_file: str='metrics.json', n_history: int=5) -> bool:
    """Update signal weights and append metric history."""
    if np is None:
        logger.error('numpy not available for updating weights')
        return False
    if new_weights.size == 0:
        logger.error('update_weights called with empty weight array')
        return False
    p = Path(weight_path)
    prev = None
    try:
        if p.exists():
            prev = np.loadtxt(p, delimiter=',')
            if np.allclose(prev, new_weights):
                logger.info('META_WEIGHTS_UNCHANGED')
                return False
        np.savetxt(p, new_weights, delimiter=',')
        logger.info('META_WEIGHTS_UPDATED', extra={'previous': prev, 'current': new_weights.tolist()})
    except (OSError, ValueError) as exc:
        logger.exception('META_WEIGHT_UPDATE_FAILED: %s', exc)
        return False
    try:
        if Path(history_file).exists():
            with open(history_file, encoding='utf-8') as f:
                hist = json.load(f)
        else:
            hist = []
    except (OSError, json.JSONDecodeError) as e:
        logger.error('Failed to read metric history: %s', e)
        hist = []
    hist.append({'ts': datetime.now(UTC).isoformat(), **metrics})
    hist = hist[-n_history:]
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(hist, f)
    logger.info('META_METRICS', extra={'recent': hist})
    return True

def update_signal_weights(weights: dict[str, float], performance: dict[str, float]) -> dict[str, float] | None:
    if not weights or not performance:
        logger.error('Empty weights or performance dict passed to update_signal_weights')
        return None
    try:
        total_perf = sum(performance.values())
        if total_perf == 0:
            logger.warning('Total performance sum is zero, skipping weight update')
            return weights
        updated_weights = {}
        for key in weights:
            perf = performance.get(key, 0)
            updated_weights[key] = weights[key] * (perf / total_perf)
        norm_factor = sum(updated_weights.values())
        if norm_factor == 0:
            logger.warning('Normalization factor zero in weight update')
            return weights
        for key in updated_weights:
            updated_weights[key] /= norm_factor
        return updated_weights
    except (ZeroDivisionError, TypeError) as exc:
        logger.exception('Exception in update_signal_weights: %s', exc)
        return weights

def save_model_checkpoint(model: Any, filepath: str) -> None:
    """Serialize ``model`` to ``filepath`` using :mod:`pickle`."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info('MODEL_CHECKPOINT_SAVED', extra={'path': filepath})
    except (OSError, pickle.PickleError) as exc:
        logger.error('Failed to save model checkpoint: %s', exc, exc_info=True)

def load_model_checkpoint(filepath: str) -> Any | None:
    """Load a model from ``filepath`` previously saved with ``save_model_checkpoint``."""
    p = Path(filepath)
    if not p.exists():
        logger.warning('Checkpoint file missing: %s', filepath)
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info('MODEL_CHECKPOINT_LOADED', extra={'path': filepath})
        return model
    except (OSError, pickle.PickleError) as exc:
        logger.error('Failed to load model checkpoint: %s', exc, exc_info=True)
        return None

def retrain_meta_learner(trade_log_path: str=None, model_path: str='meta_model.pkl', history_path: str='meta_retrain_history.pkl', min_samples: int=10) -> bool:
    """Retrain the meta-learner model from trade logs.

    Parameters
    ----------
    trade_log_path : str
        CSV file containing historical trades.
    model_path : str
        Destination to write the trained model pickle.
    history_path : str
        Path to a pickle file storing retrain metrics history.
    min_samples : int
        Minimum number of samples required to train.

    Returns
    -------
    bool
        ``True`` if retraining succeeded and the checkpoint was written.
    """
    if trade_log_path is None:
        settings = get_settings()
        trade_log_path = settings.trade_log_file if config else 'trades.csv'
    logger.info('META_RETRAIN_START', extra={'trade_log': trade_log_path, 'model_path': model_path})
    quality_report = validate_trade_data_quality(trade_log_path)
    try:
        if pd is None:
            logger.error('pandas not available for meta learning')
            return False
        df = pd.read_csv(trade_log_path)
    except (OSError, AttributeError) as exc:
        logger.error('Failed reading trade log: %s', exc, exc_info=True)
        return False
    required_cols = {'entry_price', 'exit_price', 'side'}
    total_rows = len(df)
    valid = df[['entry_price', 'exit_price']].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
    try:
        valid_rows = int(valid.sum())
    except (ValueError, TypeError):
        valid_rows = total_rows
    cols_obj = getattr(df, 'columns', [])
    try:
        cols = set(cols_obj)
    except TypeError:
        cols = set()
    quality_report.update({'file_exists': bool(total_rows), 'has_valid_format': required_cols.issubset(cols), 'row_count': total_rows, 'valid_price_rows': valid_rows, 'data_quality_score': valid_rows / total_rows if total_rows else 0.0})
    if total_rows > 0 and 'Trade log file is empty' in quality_report['issues']:
        try:
            quality_report['issues'].remove('Trade log file is empty')
        except ValueError:
            pass
    logger.info('META_LEARNING_QUALITY_CHECK', extra={'file_exists': quality_report['file_exists'], 'valid_format': quality_report['has_valid_format'], 'total_rows': quality_report['row_count'], 'valid_price_rows': quality_report['valid_price_rows'], 'quality_score': quality_report['data_quality_score']})
    for issue in quality_report['issues']:
        logger.warning(f'META_LEARNING_DATA_ISSUE: {issue}')
    for rec in quality_report['recommendations']:
        logger.info(f'META_LEARNING_RECOMMENDATION: {rec}')
    if not required_cols.issubset(df.columns):
        logger.error('META_LEARNING_CRITICAL_ISSUES: Missing required columns')
        return False
    if valid.sum() < int(min_samples):
        logger.info('META_LEARNING_INSUFFICIENT_VALID_ROWS', extra={'valid_rows': int(valid.sum()), 'min_samples': int(min_samples)})
        return False
    original_rows = len(df)
    logger.debug(f'META_LEARNING_RAW_DATA: {original_rows} total rows loaded from {trade_log_path}')
    if len(df) > 0 and len(df.columns) >= 3:
        columns = df.columns.tolist()
        has_meta_headers = any((col in ['symbol', 'entry_price', 'exit_price', 'signal_tags'] for col in columns))
        if has_meta_headers:
            sample_size = min(5, len(df))
            audit_format_detected = False
            for i in range(sample_size):
                first_col = str(df.iloc[i, 0]).strip()
                if len(first_col) > 20 and '-' in first_col:
                    audit_format_detected = True
                    break
            if audit_format_detected:
                logger.info('META_LEARNING_MIXED_FORMAT: Detected meta-learning headers with audit data, attempting conversion')
                try:
                    df = _convert_mixed_format_to_meta(df)
                    logger.info(f'META_LEARNING_MIXED_FORMAT: Successfully converted {len(df)} rows')
                except COMMON_EXC as e:
                    logger.error(f'META_LEARNING_MIXED_FORMAT: Conversion failed: {e}')
                    _implement_fallback_data_recovery(trade_log_path, 10)
                    return False
            else:
                logger.debug('META_LEARNING_PURE_FORMAT: Detected pure meta-learning format')
        else:
            logger.info('META_LEARNING_AUDIT_CONVERSION: Attempting to convert pure audit format')
            try:
                df = _convert_audit_to_meta_format(df)
                if df.empty:
                    logger.error('METALEARN_EMPTY_TRADE_LOG - No valid trades found: Failed to convert audit rows')
                    _implement_fallback_data_recovery(trade_log_path, 10)
                    return False
            except COMMON_EXC as e:
                logger.error(f'META_LEARNING_AUDIT_CONVERSION: Conversion failed: {e}')
                return False
    try:
        df = df.dropna(subset=['entry_price', 'signal_tags', 'side'])
    except KeyError:
        logger.warning('META_LEARNING_MISSING_COLUMNS')
        return False
    try:
        df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
        df['exit_price'] = pd.to_numeric(df['exit_price'], errors='coerce')
        df['exit_price'] = df['exit_price'].fillna(df['entry_price'])
        initial_rows = len(df)
        logger.debug(f'META_LEARNING_INITIAL_DATA: {initial_rows} rows before validation')
        df = df.dropna(subset=['entry_price', 'exit_price'])
        after_numeric = len(df)
        if after_numeric < initial_rows:
            logger.info(f'META_LEARNING_PRICE_CONVERSION: Removed {initial_rows - after_numeric} rows with invalid price formats')
        if len(df) > 0:
            max_reasonable_price = 50000
            min_reasonable_price = 0.01
            price_issues = (df['entry_price'] > max_reasonable_price) | (df['entry_price'] < min_reasonable_price) | (df['exit_price'] > max_reasonable_price) | (df['exit_price'] < min_reasonable_price)
            if price_issues.any():
                problematic_rows = df[price_issues]
                logger.warning(f'META_LEARNING_PRICE_OUTLIERS: Found {len(problematic_rows)} rows with unrealistic prices')
                logger.debug(f"Price outliers range: entry({problematic_rows['entry_price'].min():.2f}-{problematic_rows['entry_price'].max():.2f}), exit({problematic_rows['exit_price'].min():.2f}-{problematic_rows['exit_price'].max():.2f})")
                df = df[~price_issues]
        if len(df) > 0:
            price_change_pct = abs((df['exit_price'] - df['entry_price']) / df['entry_price'])
            extreme_moves = price_change_pct > 10.0
            if extreme_moves.any():
                extreme_count = extreme_moves.sum()
                logger.warning(f'META_LEARNING_EXTREME_MOVES: Found {extreme_count} trades with >1000% price moves')
                df.loc[extreme_moves, 'extreme_move'] = True
        if len(df) > 0:
            positive_prices = (df['entry_price'] > 0) & (df['exit_price'] > 0)
            df = df[positive_prices]
            if not positive_prices.all():
                logger.info(f'META_LEARNING_NEGATIVE_PRICES: Filtered out {(~positive_prices).sum()} trades with non-positive prices')
        if len(df) == 0:
            logger.warning('METALEARN_INVALID_PRICES - No trades with valid prices after comprehensive validation. This may indicate data quality issues or insufficient trading history. Meta-learning will continue with default weights.', extra={'initial_rows': initial_rows, 'trade_log_path': trade_log_path, 'min_samples': min_samples, 'suggestion': 'Check trade logging and price data integrity'})
            _implement_fallback_data_recovery(trade_log_path, min_samples)
            return False
        final_rows = len(df)
        retention_rate = final_rows / initial_rows * 100 if initial_rows > 0 else 0
        logger.info(f'META_LEARNING_DATA_QUALITY: Retained {final_rows}/{initial_rows} trades ({retention_rate:.1f}%)')
        if final_rows > 0:
            entry_stats = {'min': float(df['entry_price'].min()), 'max': float(df['exit_price'].max()), 'mean': float(df['entry_price'].mean())}
            exit_stats = {'min': float(df['exit_price'].min()), 'max': float(df['exit_price'].max()), 'mean': float(df['exit_price'].mean())}
            logger.debug(f"META_LEARNING_PRICE_STATS: Entry prices ${entry_stats['min']:.2f}-${entry_stats['max']:.2f} (avg: ${entry_stats['mean']:.2f})")
            logger.debug(f"META_LEARNING_PRICE_STATS: Exit prices ${exit_stats['min']:.2f}-${exit_stats['max']:.2f} (avg: ${exit_stats['mean']:.2f})")
    except COMMON_EXC as e:
        logger.error('META_LEARNING_PRICE_VALIDATION_ERROR: %s', e, exc_info=True)
        return False
    if len(df) < min_samples:
        logger.warning('META_RETRAIN_INSUFFICIENT_DATA', extra={'rows': len(df)})
        return False
    direction = np.where(df['side'] == 'buy', 1, -1)
    df['pnl'] = np.where(pd.notna(df['exit_price']), (df['exit_price'] - df['entry_price']) * direction, 0.0)
    df['outcome'] = (df['pnl'] > 0).astype(int)
    tags = sorted({t for row in df['signal_tags'] for t in str(row).split('+')})
    X = np.array([[int(t in str(row).split('+')) for t in tags] for row in df['signal_tags']])
    y = df['outcome'].values
    sample_w = df['pnl'].abs() + 0.001
    if not SKLEARN_AVAILABLE:
        logger.info('sklearn disabled, meta-learning disabled')
        return False
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0, fit_intercept=True)
    import inspect
    try:
        sig = inspect.signature(model.fit)
        if 'sample_weight' in sig.parameters:
            model.fit(X, y, sample_weight=sample_w)
        else:
            model.fit(X, y)
        model.predict(X)
    except (ValueError, RuntimeError) as exc:
        logger.exception('Meta-learner training failed: %s', exc)
        return False
    save_model_checkpoint(model, model_path)
    metrics = {'timestamp': datetime.now(UTC).isoformat(), 'samples': len(y), 'model_path': model_path}
    hist: list[dict[str, Any]] = []
    if Path(history_path).exists():
        loaded = load_model_checkpoint(history_path)
        if isinstance(loaded, list):
            hist = loaded
    hist.append(metrics)
    hist = hist[-5:]
    try:
        with open(history_path, 'wb') as f:
            pickle.dump(hist, f)
    except (OSError, pickle.PickleError) as exc:
        logger.error('Failed to update retrain history: %s', exc, exc_info=True)
    logger.info('META_RETRAIN_SUCCESS', extra={'samples': len(y), 'model': model_path})
    return True

def _implement_fallback_data_recovery(trade_log_path: str, min_samples: int) -> None:
    """Implement fallback mechanisms when historical data is insufficient."""
    logger.info('META_LEARNING_FALLBACK: Implementing data recovery procedures')
    try:
        if not Path(trade_log_path).exists():
            logger.error(f'META_LEARNING_FALLBACK: Trade log file does not exist: {trade_log_path}')
            _create_emergency_trade_log(trade_log_path)
            return
        file_size = Path(trade_log_path).stat().st_size
        if file_size == 0:
            logger.error(f'META_LEARNING_FALLBACK: Trade log file is empty: {trade_log_path}')
            _create_emergency_trade_log(trade_log_path)
            return
        with open(trade_log_path, 'r') as f:
            header_line = f.readline().strip()
            if not header_line:
                logger.error(f'META_LEARNING_FALLBACK: Trade log file has no header: {trade_log_path}')
                _create_emergency_trade_log(trade_log_path)
                return
            required_cols = ['entry_price', 'exit_price', 'signal_tags', 'side']
            header_cols = [col.strip() for col in header_line.split(',')]
            missing_cols = [col for col in required_cols if col not in header_cols]
            if missing_cols:
                logger.error(f'META_LEARNING_FALLBACK: Missing required columns {missing_cols} in {trade_log_path}')
                _backup_and_fix_trade_log(trade_log_path, header_cols, required_cols)
                return
        logger.info(f'META_LEARNING_FALLBACK: Trade log format appears valid, insufficient data for min_samples={min_samples}')
        _attempt_synthetic_data_generation(trade_log_path, min_samples)
    except COMMON_EXC as e:
        logger.error(f'META_LEARNING_FALLBACK: Error during data recovery: {e}')
        _create_emergency_trade_log(trade_log_path)

def _create_emergency_trade_log(trade_log_path: str) -> None:
    """Create an emergency trade log with proper format."""
    try:
        Path(trade_log_path).parent.mkdir(parents=True, exist_ok=True)
        headers = ['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl', 'signal_tags']
        with open(trade_log_path, 'w') as f:
            f.write(','.join(headers) + '\n')
        logger.info(f'META_LEARNING_EMERGENCY: Created new trade log with proper format: {trade_log_path}')
    except COMMON_EXC as e:
        logger.error(f'META_LEARNING_EMERGENCY: Failed to create emergency trade log: {e}')

def _backup_and_fix_trade_log(trade_log_path: str, current_cols: list, required_cols: list) -> None:
    """Backup existing log and attempt to fix format issues."""
    try:
        backup_path = f"{trade_log_path}.backup.{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        import shutil
        shutil.copy2(trade_log_path, backup_path)
        logger.info(f'META_LEARNING_BACKUP: Backed up corrupted trade log to {backup_path}')
        missing_cols = [col for col in required_cols if col not in current_cols]
        current_cols + missing_cols
        if pd is not None:
            try:
                df = pd.read_csv(trade_log_path)
                for col in missing_cols:
                    if col in ['entry_price', 'exit_price']:
                        df[col] = 0.0
                    elif col == 'signal_tags':
                        df[col] = 'unknown'
                    elif col == 'side':
                        df[col] = 'buy'
                    else:
                        df[col] = ''
                df.to_csv(trade_log_path, index=False)
                logger.info(f'META_LEARNING_FIX: Added missing columns {missing_cols} to trade log')
            except COMMON_EXC as e:
                logger.error(f'META_LEARNING_FIX: Failed to fix trade log format: {e}')
                _create_emergency_trade_log(trade_log_path)
        else:
            _create_emergency_trade_log(trade_log_path)
    except COMMON_EXC as e:
        logger.error(f'META_LEARNING_BACKUP: Failed to backup/fix trade log: {e}')
        _create_emergency_trade_log(trade_log_path)

def _generate_bootstrap_training_data(trade_log_path: str, target_samples: int) -> None:
    """
    Generate bootstrap training data based on existing trade patterns.

    AI-AGENT-REF: Smart bootstrap data generation for faster meta-learning activation.
    Uses existing trade patterns to generate realistic synthetic data for training.
    """
    try:
        existing_data = []
        if pd is not None and os.path.exists(trade_log_path):
            try:
                df = pd.read_csv(trade_log_path)
                if not df.empty and 'entry_price' in df.columns:
                    valid_df = df.dropna(subset=['entry_price', 'side', 'signal_tags'])
                    existing_data = valid_df.to_dict('records')
                    logger.info(f'META_LEARNING_BOOTSTRAP: Found {len(existing_data)} existing trades for pattern analysis')
            except COMMON_EXC as e:
                logger.debug(f'Could not read existing data for bootstrap: {e}')
        if len(existing_data) < 2:
            logger.warning('META_LEARNING_BOOTSTRAP_INSUFFICIENT_PATTERNS: Need at least 2 trades for pattern analysis')
            return
        symbols_used = list({trade.get('symbol', 'SPY') for trade in existing_data})
        sides_used = list({trade.get('side', 'buy') for trade in existing_data})
        avg_entry_price = np.mean([float(trade.get('entry_price', 100)) for trade in existing_data if trade.get('entry_price')])
        patterns_to_generate = max(target_samples - len(existing_data), 5)
        bootstrap_trades = []
        for i in range(patterns_to_generate):
            random.choice(existing_data)
            bootstrap_trade = {'timestamp': datetime.now(UTC).isoformat(), 'symbol': random.choice(symbols_used) if symbols_used else 'SPY', 'side': random.choice(sides_used) if sides_used else 'buy', 'entry_price': _generate_realistic_price(avg_entry_price), 'exit_price': '', 'quantity': random.choice([10, 25, 50, 100]), 'signal_tags': f'bootstrap_pattern_{i}+realistic_variation', 'confidence': round(random.uniform(0.3, 0.9), 2), 'strategy': 'bootstrap_generated'}
            entry_price = bootstrap_trade['entry_price']
            is_winner = i % 3 < 2
            if is_winner:
                exit_multiplier = random.uniform(1.005, 1.04)
                exit_price = entry_price * exit_multiplier
            else:
                exit_multiplier = random.uniform(0.97, 0.995)
                exit_price = entry_price * exit_multiplier
            bootstrap_trade['exit_price'] = round(exit_price, 2)
            qty = bootstrap_trade['quantity']
            if bootstrap_trade['side'] == 'buy':
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty
            bootstrap_trade['pnl'] = round(pnl, 2)
            bootstrap_trades.append(bootstrap_trade)
        if bootstrap_trades:
            _append_bootstrap_trades_to_log(trade_log_path, bootstrap_trades)
            logger.info(f'META_LEARNING_BOOTSTRAP_GENERATED: Added {len(bootstrap_trades)} bootstrap trades')
    except COMMON_EXC as e:
        logger.error(f'META_LEARNING_BOOTSTRAP_GENERATION_FAILED: {e}')
        raise

def _generate_realistic_price(base_price: float) -> float:
    """Generate realistic stock price with variation."""
    if base_price <= 0:
        base_price = 100.0
    variation = random.uniform(0.8, 1.2)
    price = base_price * variation
    price = max(1.0, min(price, 10000.0))
    return round(price, 2)

def _append_bootstrap_trades_to_log(trade_log_path: str, bootstrap_trades: list) -> None:
    """Append bootstrap trades to the trade log."""
    try:
        if not os.path.exists(trade_log_path):
            _create_emergency_trade_log(trade_log_path)
        with open(trade_log_path, 'r') as f:
            first_line = f.readline().strip()
            headers = [h.strip() for h in first_line.split(',')]
        with open(trade_log_path, 'a', newline='') as f:
            if bootstrap_trades:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                for trade in bootstrap_trades:
                    row = {header: trade.get(header, '') for header in headers}
                    writer.writerow(row)
        logger.info(f'META_LEARNING_BOOTSTRAP_APPENDED: Added {len(bootstrap_trades)} trades to {trade_log_path}')
    except COMMON_EXC as e:
        logger.error(f'Failed to append bootstrap trades: {e}')
        raise
    '\n    Generate synthetic training data for meta learning when insufficient real data exists.\n\n    AI-AGENT-REF: Enhanced meta learning fallback mechanism to handle insufficient data gracefully.\n    Creates realistic synthetic trades based on market patterns to bootstrap the meta learning system.\n    '
    try:
        logger.info(f'META_LEARNING_SYNTHETIC: Attempting to generate synthetic data for bootstrapping (need {min_samples} samples)')
        existing_data = []
        if os.path.exists(trade_log_path):
            try:
                if pd is not None:
                    existing_df = pd.read_csv(trade_log_path)
                    if not existing_df.empty:
                        existing_data = existing_df.to_dict('records')
                        logger.info(f'META_LEARNING_SYNTHETIC: Found {len(existing_data)} existing trades to use as patterns')
            except COMMON_EXC as e:
                logger.debug(f'Could not read existing data for patterns: {e}')
        if len(existing_data) < min_samples // 2:
            synthetic_trades = _generate_synthetic_trades(min_samples // 4, existing_data)
            if synthetic_trades:
                _append_synthetic_trades_to_log(trade_log_path, synthetic_trades)
                logger.warning(f'META_LEARNING_SYNTHETIC: Generated {len(synthetic_trades)} synthetic trades for meta learning bootstrap')
            else:
                logger.info('META_LEARNING_SYNTHETIC: No synthetic data generated - insufficient patterns')
        else:
            logger.info('META_LEARNING_SYNTHETIC: Sufficient existing data patterns - skipping synthetic generation')
    except COMMON_EXC as e:
        logger.error(f'META_LEARNING_SYNTHETIC: Failed to generate synthetic data: {e}')

def _generate_synthetic_trades(num_trades: int, pattern_data: list) -> list:
    """Generate realistic synthetic trades based on existing patterns."""
    try:
        if not pattern_data and num_trades > 0:
            pattern_data = [{'symbol': 'SPY', 'entry_price': 400.0, 'exit_price': 404.0, 'side': 'buy', 'pnl': 4.0}, {'symbol': 'QQQ', 'entry_price': 300.0, 'exit_price': 297.0, 'side': 'buy', 'pnl': -3.0}]
        synthetic_trades = []
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']
        for i in range(min(num_trades, 10)):
            base_price = 100.0 + i * 50
            is_winner = i % 5 < 3
            entry_price = base_price
            if is_winner:
                exit_price = entry_price * (1.0 + random.uniform(0.005, 0.03))
            else:
                exit_price = entry_price * (1.0 - random.uniform(0.01, 0.025))
            trade = {'timestamp': datetime.now(UTC).isoformat(), 'symbol': random.choice(symbols), 'side': 'buy', 'entry_price': round(entry_price, 2), 'exit_price': round(exit_price, 2), 'quantity': random.choice([10, 25, 50, 100]), 'pnl': round((exit_price - entry_price) * 50, 2), 'signal_tags': 'synthetic_bootstrap_data'}
            synthetic_trades.append(trade)
        return synthetic_trades
    except COMMON_EXC as e:
        logger.error(f'Failed to generate synthetic trade patterns: {e}')
        return []

def _append_synthetic_trades_to_log(trade_log_path: str, synthetic_trades: list) -> None:
    """Append synthetic trades to the trade log for meta learning."""
    try:
        if not os.path.exists(trade_log_path):
            _create_emergency_trade_log(trade_log_path)
        with open(trade_log_path, 'a', newline='') as f:
            if synthetic_trades:
                writer = csv.DictWriter(f, fieldnames=synthetic_trades[0].keys())
                writer.writerows(synthetic_trades)
        logger.info(f'META_LEARNING_SYNTHETIC: Appended {len(synthetic_trades)} synthetic trades to {trade_log_path}')
    except COMMON_EXC as e:
        logger.error(f'Failed to append synthetic trades: {e}')

def _convert_mixed_format_to_meta(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Convert mixed format trade logs (meta headers with audit data) to pure meta-learning format.

    Input: Meta headers but audit data rows like UUID,timestamp,symbol,side,qty,price,mode,status
    Output: symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward
    """
    try:
        if df.empty:
            return df
        meta_rows = []
        position_tracker = {}
        logger.debug(f'META_LEARNING_MIXED_CONVERSION: Processing {len(df)} rows with mixed format')
        for idx, row in df.iterrows():
            try:
                if len(row) >= 6:
                    order_id = str(row.iloc[0]).strip()
                    timestamp = str(row.iloc[1]).strip()
                    symbol = str(row.iloc[2]).strip()
                    side = str(row.iloc[3]).strip().lower()
                    qty = str(row.iloc[4]).strip()
                    price = str(row.iloc[5]).strip()
                    str(row.iloc[6]).strip() if len(row) >= 7 else 'live'
                    status = str(row.iloc[7]).strip().lower() if len(row) >= 8 else 'filled'
                    if status not in ['filled', 'partially_filled'] or symbol in ['', 'symbol']:
                        continue
                    try:
                        qty_val = abs(float(qty))
                        price_val = float(price)
                        if qty_val <= 0 or price_val <= 0:
                            continue
                    except (ValueError, TypeError):
                        continue
                    if side == 'buy':
                        meta_row = {'symbol': symbol, 'entry_time': timestamp, 'entry_price': price_val, 'exit_time': '', 'exit_price': price_val, 'qty': qty_val, 'side': side, 'strategy': 'live_trading', 'classification': 'mixed_format_conversion', 'signal_tags': f'order_id:{order_id[:8]}', 'confidence': 0.5, 'reward': 0.0}
                        if symbol not in position_tracker:
                            position_tracker[symbol] = []
                        position_tracker[symbol].append({'entry_price': price_val, 'entry_time': timestamp, 'qty': qty_val, 'meta_row': meta_row})
                        meta_rows.append(meta_row)
                    elif side == 'sell' and symbol in position_tracker and position_tracker[symbol]:
                        position = position_tracker[symbol].pop(0)
                        position['meta_row']['exit_time'] = timestamp
                        position['meta_row']['exit_price'] = price_val
                        entry_price = position['entry_price']
                        if entry_price > 0:
                            return_pct = (price_val - entry_price) / entry_price
                            position['meta_row']['reward'] = return_pct
            except COMMON_EXC as e:
                logger.debug(f'META_LEARNING_MIXED_CONVERSION: Error processing row {idx}: {e}')
                continue
        if meta_rows:
            result_df = pd.DataFrame(meta_rows)
            logger.info(f'META_LEARNING_MIXED_CONVERSION: Successfully converted {len(result_df)} trades from mixed format')
            return result_df
        else:
            logger.warning('META_LEARNING_MIXED_CONVERSION: No valid trades found after conversion')
            return pd.DataFrame()
    except COMMON_EXC as e:
        logger.error(f'META_LEARNING_MIXED_CONVERSION: Conversion failed: {e}')
        return pd.DataFrame()

def _convert_audit_to_meta_format(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Convert audit format trade logs to meta-learning format.

    Audit format: order_id,timestamp,symbol,side,qty,price,mode,status
    Meta format: symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward
    """
    try:
        meta_rows = []
        position_tracker = {}
        audit_rows = []
        for _idx, row in df.iterrows():
            first_col = str(row.iloc[0]).strip()
            if len(first_col) > 20 and '-' in first_col:
                audit_rows.append(row)
        logger.debug(f'META_LEARNING_AUDIT_CONVERSION: Found {len(audit_rows)} audit format rows to convert')
        for row in audit_rows:
            try:
                if len(row) >= 6:
                    order_id = str(row.iloc[0]).strip()
                    timestamp = str(row.iloc[1]).strip()
                    symbol = str(row.iloc[2]).strip()
                    side = str(row.iloc[3]).strip().lower()
                    qty = str(row.iloc[4]).strip()
                    price = str(row.iloc[5]).strip()
                    status = str(row.iloc[7]).strip().lower() if len(row) >= 8 else 'unknown'
                    invalid_statuses = ['pending', 'cancelled', 'canceled', 'rejected', 'failed', 'error', 'unknown']
                    if status in invalid_statuses:
                        logger.debug(f'Skipping audit row with invalid status: {status}')
                        continue
                    try:
                        qty_val = float(qty)
                        price_val = float(price)
                    except (ValueError, TypeError):
                        logger.debug(f'Invalid numeric values in audit row: qty={qty}, price={price}')
                        continue
                    if symbol not in position_tracker:
                        position_tracker[symbol] = []
                    meta_row = {'symbol': symbol, 'entry_time': timestamp, 'entry_price': price_val, 'exit_time': '', 'exit_price': '', 'qty': qty_val, 'side': side, 'strategy': 'audit_converted', 'classification': 'converted', 'signal_tags': f'audit_order_{order_id[:8]}', 'confidence': '0.5', 'reward': ''}
                    opposite_side = 'sell' if side == 'buy' else 'buy'
                    matching_positions = [p for p in position_tracker[symbol] if p['side'] == opposite_side and (not p.get('matched', False))]
                    if matching_positions:
                        match = matching_positions[0]
                        match['matched'] = True
                        meta_row['exit_time'] = timestamp
                        meta_row['exit_price'] = price_val
                        if side == 'buy':
                            pnl = (price_val - match['entry_price']) * qty_val
                        else:
                            pnl = (match['entry_price'] - price_val) * qty_val
                        meta_row['reward'] = pnl
                        for existing_row in meta_rows:
                            if existing_row['symbol'] == symbol and existing_row['entry_price'] == match['entry_price'] and (existing_row['side'] == opposite_side) and (not existing_row['exit_price']):
                                existing_row['exit_time'] = timestamp
                                existing_row['exit_price'] = price_val
                                if side == 'sell':
                                    existing_row['reward'] = (price_val - existing_row['entry_price']) * existing_row['qty']
                                else:
                                    existing_row['reward'] = (existing_row['entry_price'] - price_val) * existing_row['qty']
                                break
                    position_tracker[symbol].append(meta_row.copy())
                    meta_rows.append(meta_row)
            except COMMON_EXC as e:
                logger.debug(f"Failed to convert audit row {(row.iloc[0] if len(row) > 0 else 'unknown')}: {e}")
                continue
        if meta_rows:
            converted_df = pd.DataFrame(meta_rows)
            logger.info(f'META_LEARNING_AUDIT_CONVERSION: Successfully converted {len(meta_rows)} audit rows to meta format')
            return converted_df
        else:
            logger.warning('META_LEARNING_AUDIT_CONVERSION: No valid audit rows found for conversion')
            return pd.DataFrame()
    except COMMON_EXC as e:
        logger.error(f'META_LEARNING_AUDIT_CONVERSION: Conversion failed: {e}')
        return pd.DataFrame()

def optimize_signals(signal_data: Any, cfg: Any, model: Any | None=None, *, volatility: float=1.0) -> Any:
    """Optimize trading signals using ``model`` if provided."""
    if signal_data is None:
        logger.warning('optimize_signals received None signal_data, returning empty list')
        return []
    if hasattr(signal_data, '__len__') and len(signal_data) == 0:
        logger.warning('optimize_signals received empty signal_data, returning empty list')
        return []
    if model is not None:
        try:
            predictions = model.predict(signal_data)
            if predictions is None:
                logger.warning('Model prediction returned None, falling back to original signal_data')
                return signal_data if signal_data is not None else []
            return list(predictions)
        except (ValueError, RuntimeError, AttributeError) as exc:
            logger.exception('optimize_signals model prediction failed: %s', exc)
            return signal_data if signal_data is not None else []
    if model is None:
        try:
            model = load_model_checkpoint(cfg.MODEL_PATH)
        except AttributeError:
            logger.warning('cfg object missing MODEL_PATH attribute')
            return signal_data if signal_data is not None else []
    if model is None:
        logger.debug('No model available for signal optimization, returning original data')
        return signal_data if signal_data is not None else []
    try:
        preds = model.predict(signal_data)
        if preds is None:
            logger.warning('Model predict returned None')
            return signal_data if signal_data is not None else []
        if np is not None:
            preds = np.clip(preds, -1.2, 1.2)
            factor = 1.0 if volatility <= 1.0 else 1.0 / max(volatility, 0.001)
            preds = preds * factor
            return list(preds)
        else:
            preds = [max(-1.2, min(1.2, p)) for p in preds]
            factor = 1.0 if volatility <= 1.0 else 1.0 / max(volatility, 0.001)
            preds = [p * factor for p in preds]
            return preds
    except (ValueError, RuntimeError, TypeError) as exc:
        logger.exception('optimize_signals prediction processing failed: %s', exc)
        return signal_data if signal_data is not None else []

def trigger_rebalance_on_regime(df: 'pd.DataFrame') -> None:
    """Invoke the RL rebalancer when the market regime changes."""
    settings = get_settings()
    if not settings.enable_reinforcement_learning:
        return
    if find_spec('portfolio_rl') is None:
        raise RuntimeError('Reinforcement learning enabled but portfolio_rl module unavailable. Set ENABLE_REINFORCEMENT_LEARNING=False to disable')
    from portfolio_rl import PortfolioReinforcementLearner
    rl = PortfolioReinforcementLearner()
    if 'Regime' in df.columns and len(df) > 2:
        if df['Regime'].iloc[-1] != df['Regime'].iloc[-2]:
            state_data = df.tail(10).dropna().values.flatten()
            rl.rebalance_portfolio(state_data)

def trigger_meta_learning_conversion(trade_data: dict) -> bool:
    """Automatically convert audit logs to meta-learning format after trade execution."""
    try:
        symbol = trade_data.get('symbol', 'UNKNOWN')
        logger.info('META_LEARNING_TRIGGER | symbol=%s', symbol)
        current_config = None
        if config is not None:
            current_config = config
        if current_config is None:
            import sys
            if 'config' in sys.modules:
                current_config = sys.modules['config']
        import sys
        if 'config' in sys.modules and hasattr(sys.modules['config'], 'TRADE_LOG_FILE'):
            current_config = sys.modules['config']
        if current_config is None:
            logger.warning('Config not available for meta-learning conversion')
            return False
        trade_log_path = getattr(current_config, 'TRADE_LOG_FILE', 'logs/trades.csv')
        try:
            trade_log_path_obj = Path(trade_log_path)
        except (TypeError, ValueError) as e:
            logger.warning('METALEARN_INVALID_PATH | invalid path format: %s, error: %s', trade_log_path, e)
            return False
        if not trade_log_path_obj.exists():
            logger.warning('METALEARN_NO_TRADE_LOG | %s does not exist', trade_log_path)
            return False
        if not trade_log_path_obj.is_file():
            logger.warning('METALEARN_NOT_A_FILE | %s is not a regular file', trade_log_path)
            return False
        try:
            file_stat = trade_log_path_obj.stat()
            if file_stat.st_size == 0:
                logger.warning('METALEARN_EMPTY_FILE | %s is empty', trade_log_path)
                return False
        except (OSError, FileNotFoundError, PermissionError) as e:
            logger.warning('METALEARN_FILE_ACCESS_ERROR | cannot access %s: %s', trade_log_path, e)
            return False
        except COMMON_EXC as e:
            logger.warning('METALEARN_UNEXPECTED_FILE_ERROR | unexpected error accessing %s: %s', trade_log_path, e)
            return False
        quality_report = validate_trade_data_quality(trade_log_path)
        if not quality_report.get('file_exists', False):
            logger.warning('METALEARN_FILE_NOT_EXIST | trade log not accessible')
            return False
        if quality_report.get('row_count', 0) == 0:
            logger.warning('METALEARN_EMPTY_TRADE_LOG | no rows found in trade log')
            return False
        mixed_format = quality_report.get('mixed_format_detected', False)
        audit_rows = quality_report.get('audit_format_rows', 0)
        meta_rows = quality_report.get('meta_format_rows', 0)
        conversion_needed = mixed_format or (audit_rows > 0 and meta_rows == 0)
        if conversion_needed:
            if mixed_format:
                logger.info('METALEARN_MIXED_FORMAT_DETECTED | triggering conversion')
            else:
                logger.info('METALEARN_AUDIT_FORMAT_DETECTED | triggering conversion')
            if pd is not None:
                try:
                    df = pd.read_csv(trade_log_path, header=None)
                    converted_df = _convert_audit_to_meta_format(df)
                    if not converted_df.empty:
                        converted_df.to_csv(trade_log_path, index=False)
                        logger.info('METALEARN_CONVERSION_SUCCESS | symbol=%s converted_rows=%d', symbol, len(converted_df))
                        return True
                    else:
                        logger.warning('METALEARN_CONVERSION_EMPTY | no valid conversion result')
                        return False
                except COMMON_EXC as e:
                    logger.error('METALEARN_CONVERSION_ERROR | symbol=%s error=%s', symbol, e)
                    return False
            else:
                logger.warning('METALEARN_NO_PANDAS | pandas not available for conversion')
                return False
        else:
            trade_log_path_obj = Path(trade_log_path)
            if not trade_log_path_obj.exists() or not trade_log_path_obj.is_file():
                logger.warning('METALEARN_FILE_DISAPPEARED | trade log file no longer exists')
                return False
            logger.debug('METALEARN_NO_CONVERSION_NEEDED | trade log format is already correct')
            return True
    except COMMON_EXC as exc:
        logger.error('METALEARN_TRIGGER_ERROR | symbol=%s error=%s', trade_data.get('symbol', 'UNKNOWN'), exc)
        return False

def convert_audit_to_meta(trade_data: dict) -> dict | None:
    """Convert audit trade data to meta-learning format."""
    try:
        converted_data = {'symbol': trade_data.get('symbol', ''), 'entry_time': trade_data.get('timestamp', ''), 'entry_price': trade_data.get('price', 0.0), 'exit_time': '', 'exit_price': '', 'qty': trade_data.get('qty', 0), 'side': trade_data.get('side', ''), 'strategy': trade_data.get('strategy', 'auto_converted'), 'classification': 'converted', 'signal_tags': f"order_{trade_data.get('order_id', '')[:8]}", 'confidence': trade_data.get('confidence', 0.5), 'reward': ''}
        logger.info('METALEARN_SINGLE_CONVERSION | symbol=%s', converted_data['symbol'])
        return converted_data
    except COMMON_EXC as exc:
        logger.error('METALEARN_SINGLE_CONVERSION_ERROR | error=%s', exc)
        return None

def store_meta_learning_data(converted_data: dict) -> bool:
    """Store converted meta-learning data."""
    try:
        if config is None:
            logger.warning('Config not available for storing meta-learning data')
            return False
        meta_log_path = getattr(config, 'META_LEARNING_LOG_FILE', 'logs/meta_trades.csv')
        meta_log_dir = Path(meta_log_path).parent
        meta_log_dir.mkdir(parents=True, exist_ok=True)
        if pd is not None:
            df = pd.DataFrame([converted_data])
            if Path(meta_log_path).exists():
                df.to_csv(meta_log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(meta_log_path, index=False)
            logger.info('METALEARN_DATA_STORED | symbol=%s path=%s', converted_data.get('symbol', 'UNKNOWN'), meta_log_path)
            return True
        else:
            logger.warning('METALEARN_STORE_NO_PANDAS | pandas not available')
            return False
    except COMMON_EXC as exc:
        logger.error('METALEARN_STORE_ERROR | error=%s', exc)
        return False

def load_global_signal_performance(min_trades: int=3, threshold: float=0.4) -> dict[str, float] | None:
    """Load global signal performance with enhanced error handling.

    This function is available in both bot_engine.py and meta_learning.py for
    improved module integration and testing compatibility.
    """
    try:
        import sys
        if 'bot_engine' not in sys.modules:
            logger.info('META_LEARNING_FALLBACK: bot_engine not available, using fallback implementation')
            trade_log_file = getattr(config, 'TRADE_LOG_FILE', 'trades.csv') if config else 'trades.csv'
            if not os.path.exists(trade_log_file):
                logger.info('METALEARN_NO_HISTORY: Trade log file not found')
                return None
            if pd is None:
                logger.warning('METALEARN_NO_PANDAS: pandas not available for signal performance loading')
                return None
            try:
                df = pd.read_csv(trade_log_file, on_bad_lines='skip', engine='python', usecols=['exit_price', 'entry_price', 'signal_tags', 'side']).dropna(subset=['exit_price', 'entry_price', 'signal_tags'])
                if df.empty:
                    logger.warning('METALEARN_EMPTY_TRADE_LOG - No valid trades found')
                    return {}
                df['exit_price'] = pd.to_numeric(df['exit_price'], errors='coerce')
                df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
                df['signal_tags'] = df['signal_tags'].astype(str)
                df = df.dropna(subset=['exit_price', 'entry_price'])
                if df.empty:
                    logger.warning('METALEARN_NO_VALID_PRICES - No trades with valid prices')
                    return {}
                df['pnl'] = (df['exit_price'] - df['entry_price']) * df['side'].map({'buy': 1, 'sell': -1})
                signal_performance = {}
                for _, row in df.iterrows():
                    tags = str(row['signal_tags']).split('+')
                    for tag in tags:
                        tag = tag.strip()
                        if tag:
                            if tag not in signal_performance:
                                signal_performance[tag] = {'total_pnl': 0.0, 'trades': 0}
                            signal_performance[tag]['total_pnl'] += row['pnl']
                            signal_performance[tag]['trades'] += 1
                result = {}
                for tag, data in signal_performance.items():
                    if data['trades'] >= min_trades:
                        avg_pnl = data['total_pnl'] / data['trades']
                        if abs(avg_pnl) >= threshold:
                            result[tag] = avg_pnl
                logger.info(f'META_LEARNING_SIGNAL_PERFORMANCE: Loaded {len(result)} signals with {min_trades}+ trades')
                return result if result else {}
            except COMMON_EXC as e:
                logger.error(f'META_LEARNING_SIGNAL_PERFORMANCE_ERROR: {e}')
                return None
        else:
            bot_engine = sys.modules['bot_engine']
            if hasattr(bot_engine, 'load_global_signal_performance'):
                return bot_engine.load_global_signal_performance(min_trades, threshold)
            else:
                logger.warning('META_LEARNING_FUNCTION_MISSING: load_global_signal_performance not found in bot_engine')
                return None
    except COMMON_EXC as exc:
        logger.error('META_LEARNING_LOAD_PERFORMANCE_ERROR: %s', exc)
        return None