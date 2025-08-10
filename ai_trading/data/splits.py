"""
Time series cross-validation splits with purging and embargo.

Provides leak-proof data splitting for financial time series,
including purged group time series splits and walk-forward analysis.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, Union, List, Dict
from datetime import datetime, timedelta
import logging

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger

# sklearn is a hard dependency
from sklearn.model_selection import BaseCrossValidator


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Time series cross-validation with purging and embargo.
    
    Ensures no data leakage by purging observations that overlap
    with the test set timeline and applying embargo periods.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[Union[int, float]] = None,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.02
    ):
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
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        t1: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
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
        try:
            # Get indices
            if hasattr(X, 'index'):
                indices = X.index
                n_samples = len(X)
            else:
                indices = np.arange(len(X))
                n_samples = len(X)
            
            # Convert to datetime index if possible
            if hasattr(X, 'index') and not isinstance(indices, pd.DatetimeIndex):
                try:
                    indices = pd.to_datetime(indices)
                except Exception as e:
                    from ai_trading.logging import logger
                    logger.debug("Datetime index cast failed; using positional indices: %s", e)
            
            # Calculate test size
            if self.test_size is None:
                test_size = n_samples // (self.n_splits + 1)
            elif isinstance(self.test_size, float):
                test_size = int(n_samples * self.test_size)
            else:
                test_size = self.test_size
            
            # Calculate embargo and purge sizes
            embargo_size = int(n_samples * self.embargo_pct)
            purge_size = int(n_samples * self.purge_pct)
            
            logger.debug(f"Split parameters: n_samples={n_samples}, test_size={test_size}, "
                        f"embargo_size={embargo_size}, purge_size={purge_size}")
            
            # Generate splits
            for i in range(self.n_splits):
                # Calculate test set boundaries
                test_start = int(n_samples * (i + 1) / (self.n_splits + 1))
                test_end = min(test_start + test_size, n_samples)
                
                # Test indices
                test_indices = np.arange(test_start, test_end)
                
                # Training indices (before test set, with purging and embargo)
                train_end = test_start - purge_size
                train_indices = np.arange(0, max(0, train_end))
                
                # Apply embargo (remove observations too close to test start)
                if embargo_size > 0:
                    embargo_cutoff = test_start - embargo_size
                    train_indices = train_indices[train_indices < embargo_cutoff]
                
                # Purge overlapping observations if t1 is provided
                if t1 is not None:
                    train_indices = self._purge_overlapping(
                        train_indices, test_indices, t1, indices
                    )
                
                # Ensure indices are valid
                train_indices = train_indices[
                    (train_indices >= 0) & (train_indices < n_samples)
                ]
                test_indices = test_indices[
                    (test_indices >= 0) & (test_indices < n_samples)
                ]
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    logger.debug(f"Split {i}: train_size={len(train_indices)}, "
                                f"test_size={len(test_indices)}")
                    yield train_indices, test_indices
                else:
                    logger.warning(f"Split {i}: insufficient data after purging")
                    
        except Exception as e:
            logger.error(f"Error in split generation: {e}")
            return
    
    def _purge_overlapping(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        t1: pd.Series,
        full_index: Union[pd.Index, np.ndarray]
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
        try:
            if len(test_indices) == 0:
                return train_indices
            
            # Get test period start and end
            test_start_idx = test_indices[0]
            test_end_idx = test_indices[-1]
            
            if hasattr(full_index, 'to_series'):
                test_start_time = full_index[test_start_idx]
                test_end_time = full_index[test_end_idx]
            else:
                # For non-datetime indices, use index positions
                test_start_time = test_start_idx
                test_end_time = test_end_idx
            
            # Find training observations that don't overlap
            purged_train = []
            for idx in train_indices:
                try:
                    if idx < len(t1.index):
                        obs_end_time = t1.iloc[idx]
                        
                        # Check if observation ends before test period starts
                        if pd.isna(obs_end_time) or obs_end_time < test_start_time:
                            purged_train.append(idx)
                    else:
                        # If t1 index doesn't cover this observation, include it
                        # only if it's well before the test period
                        if idx < test_start_idx - len(test_indices):
                            purged_train.append(idx)
                except Exception:
                    # If there's any issue with the time comparison,
                    # err on the side of caution and exclude
                    continue
            
            return np.array(purged_train)
            
        except Exception as e:
            logger.error(f"Error in purging overlapping observations: {e}")
            return train_indices
    
    def get_n_splits(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


def walkforward_splits(
    dates: Union[pd.DatetimeIndex, List[datetime]],
    mode: str = "rolling",
    train_span: Union[int, timedelta] = 252,  # 1 year of trading days
    test_span: Union[int, timedelta] = 21,    # 1 month of trading days
    embargo_pct: float = 0.01
) -> List[Dict[str, Union[datetime, List[datetime]]]]:
    """
    Generate walk-forward analysis splits.
    
    Args:
        dates: Timeline of dates for analysis
        mode: 'rolling' or 'anchored' walk-forward
        train_span: Training period length (days or timedelta)
        test_span: Test period length (days or timedelta)  
        embargo_pct: Embargo period as fraction of training period
        
    Returns:
        List of split dictionaries with train/test periods
    """
    try:
        # Convert to DatetimeIndex if needed
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)
        
        # Convert spans to timedelta if needed
        if isinstance(train_span, int):
            train_span = timedelta(days=train_span)
        if isinstance(test_span, int):
            test_span = timedelta(days=test_span)
        
        # Calculate embargo period
        embargo_period = timedelta(days=int(train_span.days * embargo_pct))
        
        splits = []
        start_date = dates.min()
        end_date = dates.max()
        
        # Initial training end
        current_train_end = start_date + train_span
        
        while current_train_end + test_span <= end_date:
            # Training period
            if mode == "rolling":
                train_start = current_train_end - train_span
            else:  # anchored
                train_start = start_date
            
            # Apply embargo
            test_start = current_train_end + embargo_period
            test_end = test_start + test_span
            
            # Ensure test end doesn't exceed available data
            if test_end > end_date:
                test_end = end_date
            
            # Get actual dates in the periods
            train_dates = dates[(dates >= train_start) & (dates < current_train_end)]
            test_dates = dates[(dates >= test_start) & (dates < test_end)]
            
            if len(train_dates) > 0 and len(test_dates) > 0:
                split_info = {
                    'train_start': train_start,
                    'train_end': current_train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_dates': train_dates.tolist(),
                    'test_dates': test_dates.tolist(),
                    'mode': mode,
                    'embargo_days': embargo_period.days
                }
                splits.append(split_info)
                
                logger.debug(f"Walk-forward split: train {train_start.date()} to "
                           f"{current_train_end.date()}, test {test_start.date()} to "
                           f"{test_end.date()}")
            
            # Move to next period
            current_train_end += test_span
        
        logger.info(f"Generated {len(splits)} walk-forward splits using {mode} mode")
        return splits
        
    except Exception as e:
        logger.error(f"Error generating walk-forward splits: {e}")
        return []


def validate_no_leakage(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    timeline: Union[pd.DatetimeIndex, np.ndarray],
    t1: Optional[pd.Series] = None
) -> bool:
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
        # Check for direct overlap
        overlap = np.intersect1d(train_indices, test_indices)
        if len(overlap) > 0:
            logger.error(f"Direct index overlap detected: {len(overlap)} indices")
            return False
        
        # Check temporal ordering
        if hasattr(timeline, '__getitem__'):
            if len(train_indices) > 0 and len(test_indices) > 0:
                max_train_time = timeline[train_indices].max()
                min_test_time = timeline[test_indices].min()
                
                if max_train_time >= min_test_time:
                    logger.warning("Potential temporal leakage: training data overlaps test period")
                    
                    # If t1 is provided, check for observation overlap
                    if t1 is not None:
                        for train_idx in train_indices:
                            if train_idx < len(t1):
                                obs_end = t1.iloc[train_idx]
                                if pd.notna(obs_end) and obs_end >= min_test_time:
                                    logger.error(f"Training observation {train_idx} ends in test period")
                                    return False
        
        logger.debug("No data leakage detected")
        return True
        
    except Exception as e:
        logger.error(f"Error validating leakage: {e}")
        return False