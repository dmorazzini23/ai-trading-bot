"""
Time series cross-validation splits with purging and embargo.

Provides leak-proof data splitting for financial time series,
including purged group time series splits and walk-forward analysis.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
import numpy as np
from typing import TYPE_CHECKING, cast
from ai_trading.logging import logger
from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:
    import pandas as pd

class PurgedGroupTimeSeriesSplit:
    """
    Time series cross-validation with purging and embargo.
    
    Ensures no data leakage by purging observations that overlap
    with the test set timeline and applying embargo periods.
    """

    def __init__(self, n_splits: int=5, test_size: int | float | None=None, embargo_pct: float=0.01, purge_pct: float=0.02):
        """
        Initialize purged group time series split.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (int for observations, float for fraction)
            embargo_pct: Embargo period as fraction of total observations
            purge_pct: Purge period as fraction of total observations
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: "pd.DataFrame" | np.ndarray,
        y: "pd.Series" | np.ndarray | None = None,
        groups: "pd.Series" | np.ndarray | None = None,
        t1: "pd.Series" | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for train/test splits.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            groups: Group labels (optional, uses index if not provided)
            t1: End times for each observation (for proper purging)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        pd = load_pandas()
        try:
            if hasattr(X, 'index'):
                indices = X.index
                n_samples = len(X)
            else:
                indices = np.arange(len(X))
                n_samples = len(X)
            if hasattr(X, 'index') and (not isinstance(indices, pd.DatetimeIndex)):
                try:
                    indices = pd.to_datetime(indices)
                except (ValueError, TypeError) as e:
                    logger.debug('Datetime index cast failed; using positional indices: %s', e)
            if self.test_size is None:
                test_size = n_samples // (self.n_splits + 1)
            elif isinstance(self.test_size, float):
                test_size = int(n_samples * self.test_size)
            else:
                test_size = self.test_size
            embargo_size = int(n_samples * self.embargo_pct)
            purge_size = int(n_samples * self.purge_pct)
            logger.debug(f'Split parameters: n_samples={n_samples}, test_size={test_size}, embargo_size={embargo_size}, purge_size={purge_size}')
            for i in range(self.n_splits):
                test_start = int(n_samples * (i + 1) / (self.n_splits + 1))
                test_end = min(test_start + test_size, n_samples)
                test_indices = np.arange(test_start, test_end)
                train_end = test_start - purge_size
                train_indices = np.arange(0, max(0, train_end))
                if embargo_size > 0:
                    embargo_cutoff = test_start - embargo_size
                    train_indices = train_indices[train_indices < embargo_cutoff]
                if t1 is not None:
                    train_indices = self._purge_overlapping(train_indices, test_indices, t1, indices)
                train_indices = train_indices[(train_indices >= 0) & (train_indices < n_samples)]
                test_indices = test_indices[(test_indices >= 0) & (test_indices < n_samples)]
                if len(train_indices) > 0 and len(test_indices) > 0:
                    logger.debug(f'Split {i}: train_size={len(train_indices)}, test_size={len(test_indices)}')
                    yield (train_indices, test_indices)
                else:
                    logger.warning(f'Split {i}: insufficient data after purging')
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f'Error in split generation: {e}')
            return

    def _purge_overlapping(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        t1: "pd.Series",
        full_index: pd.Index | np.ndarray,
    ) -> np.ndarray:
        """
        Purge training observations that overlap with test period.
        
        Args:
            train_indices: Training indices
            test_indices: Test indices  
            t1: End times for each observation
            full_index: Full index of the dataset
            
        Returns:
            Purged training indices
        """
        pd = load_pandas()
        try:
            if len(test_indices) == 0:
                return train_indices
            test_start_idx = test_indices[0]
            test_end_idx = test_indices[-1]
            if hasattr(full_index, 'to_series'):
                test_start_time = full_index[test_start_idx]
                full_index[test_end_idx]
            else:
                test_start_time = test_start_idx
            purged_train = []
            for idx in train_indices:
                try:
                    if idx < len(t1.index):
                        obs_end_time = t1.iloc[idx]
                        if not pd.isna(obs_end_time) and obs_end_time < test_start_time:
                            purged_train.append(idx)
                    elif idx < test_start_idx - len(test_indices):
                        purged_train.append(idx)
                except (IndexError, KeyError, ValueError, TypeError):
                    continue
            return cast(np.ndarray, np.array(purged_train))
        except (KeyError, ValueError, IndexError, TypeError) as e:
            logger.error(f'Error in purging overlapping observations: {e}')
            return train_indices

    def get_n_splits(
        self,
        X: "pd.DataFrame" | np.ndarray | None = None,
        y: "pd.Series" | np.ndarray | None = None,
        groups: "pd.Series" | np.ndarray | None = None,
    ) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


def validate_no_leakage(train_indices: np.ndarray, test_indices: np.ndarray, timeline: pd.DatetimeIndex | np.ndarray, t1: pd.Series | None=None) -> bool:
    """
    Validate that there's no data leakage between train and test sets.
    
    Args:
        train_indices: Training set indices
        test_indices: Test set indices
        timeline: Timeline of observations
        t1: End times for each observation
        
    Returns:
        True if no leakage detected, False otherwise
    """
    try:
        pd = load_pandas()
        overlap = np.intersect1d(train_indices, test_indices)
        if len(overlap) > 0:
            logger.error(f'Direct index overlap detected: {len(overlap)} indices')
            return False
        if hasattr(timeline, '__getitem__'):
            if len(train_indices) > 0 and len(test_indices) > 0:
                max_train_time = timeline[train_indices].max()
                min_test_time = timeline[test_indices].min()
                if t1 is not None:
                    for train_idx in train_indices:
                        if train_idx < len(t1):
                            obs_end = t1.iloc[train_idx]
                            if pd.notna(obs_end) and obs_end >= min_test_time:
                                logger.error(f'Training observation {train_idx} ends in test period')
                                return False
                if max_train_time >= min_test_time:
                    logger.warning('Potential temporal leakage: training data overlaps test period')
                    return False
        logger.debug('No data leakage detected')
        return True
    except (ValueError, TypeError) as e:
        logger.error(f'Error validating leakage: {e}')
        return False
