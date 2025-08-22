"""
Pre-signal data sanitization for outlier detection and data quality.

Provides data cleaning pipeline with winsorization, volume filtering,
and stale data detection to guard against poor quality market data.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SanitizationConfig:
    """Configuration for data sanitization."""
    # Outlier detection
    mad_threshold: float = 3.0  # MAD multiplier for outlier detection
    zscore_threshold: float = 4.0  # Z-score threshold for outlier detection
    winsorize_limits: Tuple[float, float] = (0.01, 0.01)  # (lower, upper) percentiles

    # Volume filtering
    min_volume_percentile: float = 5.0  # Minimum volume percentile
    min_absolute_volume: int = 1000  # Minimum absolute volume

    # Stale data detection
    max_gap_hours: float = 24.0  # Maximum gap in hours before considering stale
    max_price_staleness: int = 5  # Maximum consecutive bars with same price

    # Price validation
    min_price: float = 0.01  # Minimum valid price
    max_price_change: float = 0.5  # Maximum single-bar price change (50%)

    # Enable/disable features
    enable_outlier_detection: bool = True
    enable_volume_filtering: bool = True
    enable_stale_detection: bool = True
    enable_price_validation: bool = True

    # Logging
    log_rejections: bool = True


class DataSanitizer:
    """
    Data sanitization pipeline for market data quality control.
    
    Provides comprehensive data cleaning including outlier detection,
    volume filtering, stale data detection, and price validation.
    """

    def __init__(self, config: Optional[SanitizationConfig] = None):
        """
        Initialize data sanitizer.
        
        Args:
            config: Sanitization configuration
        """
        self.config = config or SanitizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Rejection statistics
        self._rejection_stats = {
            'outliers': 0,
            'low_volume': 0,
            'stale_gaps': 0,
            'stale_prices': 0,
            'invalid_prices': 0,
            'excessive_moves': 0,
            'total_processed': 0
        }

    def sanitize_bars(
        self,
        bars: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Sanitize OHLCV bar data.
        
        Args:
            bars: DataFrame with OHLCV data
            symbol: Symbol for logging
            
        Returns:
            Tuple of (cleaned_bars, sanitization_report)
        """
        if bars.empty:
            return bars, {'status': 'empty', 'rejections': {}}

        original_count = len(bars)
        self._rejection_stats['total_processed'] += original_count

        # Create copy to avoid modifying original
        clean_bars = bars.copy()

        # Track rejections
        rejection_mask = pd.Series(False, index=clean_bars.index)
        rejection_reasons = pd.Series('', index=clean_bars.index)

        # Apply sanitization steps
        if self.config.enable_price_validation:
            price_mask, price_reasons = self._validate_prices(clean_bars)
            rejection_mask |= price_mask
            rejection_reasons = self._update_reasons(rejection_reasons, price_reasons)

        if self.config.enable_outlier_detection:
            outlier_mask, outlier_reasons = self._detect_outliers(clean_bars)
            rejection_mask |= outlier_mask
            rejection_reasons = self._update_reasons(rejection_reasons, outlier_reasons)

        if self.config.enable_volume_filtering:
            volume_mask, volume_reasons = self._filter_low_volume(clean_bars)
            rejection_mask |= volume_mask
            rejection_reasons = self._update_reasons(rejection_reasons, volume_reasons)

        if self.config.enable_stale_detection:
            stale_mask, stale_reasons = self._detect_stale_data(clean_bars)
            rejection_mask |= stale_mask
            rejection_reasons = self._update_reasons(rejection_reasons, stale_reasons)

        # Remove rejected bars
        rejected_bars = clean_bars[rejection_mask]
        clean_bars = clean_bars[~rejection_mask]

        # Log rejections if enabled
        if self.config.log_rejections and len(rejected_bars) > 0:
            self._log_rejections(symbol, rejected_bars, rejection_reasons[rejection_mask])

        # Create sanitization report
        report = {
            'symbol': symbol,
            'original_count': original_count,
            'cleaned_count': len(clean_bars),
            'rejected_count': len(rejected_bars),
            'rejection_rate': len(rejected_bars) / original_count if original_count > 0 else 0,
            'rejections': self._count_rejection_reasons(rejection_reasons[rejection_mask]),
            'time_range': self._get_time_range(bars) if not bars.empty else None
        }

        return clean_bars, report

    def _validate_prices(self, bars: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Validate price data for basic sanity checks."""
        rejection_mask = pd.Series(False, index=bars.index)
        rejection_reasons = pd.Series('', index=bars.index)

        # Get price columns
        price_cols = self._get_price_columns(bars)

        for col in price_cols:
            if col not in bars.columns:
                continue

            prices = bars[col]

            # Check minimum price
            min_price_mask = prices < self.config.min_price
            rejection_mask |= min_price_mask
            rejection_reasons.loc[min_price_mask] = f'price_too_low_{col}'
            self._rejection_stats['invalid_prices'] += min_price_mask.sum()

            # Check for NaN/infinite prices
            invalid_mask = ~np.isfinite(prices)
            rejection_mask |= invalid_mask
            rejection_reasons.loc[invalid_mask] = f'invalid_price_{col}'
            self._rejection_stats['invalid_prices'] += invalid_mask.sum()

        # Check for excessive price moves
        if 'close' in bars.columns:
            close_prices = bars['close']
            pct_change = close_prices.pct_change().abs()
            excessive_move_mask = pct_change > self.config.max_price_change
            rejection_mask |= excessive_move_mask
            rejection_reasons.loc[excessive_move_mask] = 'excessive_price_move'
            self._rejection_stats['excessive_moves'] += excessive_move_mask.sum()

        return rejection_mask, rejection_reasons

    def _detect_outliers(self, bars: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect outliers using MAD and Z-score methods."""
        rejection_mask = pd.Series(False, index=bars.index)
        rejection_reasons = pd.Series('', index=bars.index)

        price_cols = self._get_price_columns(bars)

        for col in price_cols:
            if col not in bars.columns:
                continue

            prices = bars[col].dropna()
            if len(prices) < 10:  # Need minimum data for outlier detection
                continue

            # MAD-based outlier detection
            median_price = prices.median()
            mad = np.median(np.abs(prices - median_price))

            if mad > 0:  # Avoid division by zero
                mad_scores = np.abs(prices - median_price) / mad
                mad_outliers = mad_scores > self.config.mad_threshold

                outlier_mask = mad_outliers.reindex(bars.index, fill_value=False)
                rejection_mask |= outlier_mask
                rejection_reasons.loc[outlier_mask] = f'mad_outlier_{col}'
                self._rejection_stats['outliers'] += outlier_mask.sum()

            # Z-score based outlier detection
            if len(prices) > 2:
                z_scores = np.abs((prices - prices.mean()) / prices.std())
                z_outliers = z_scores > self.config.zscore_threshold

                outlier_mask = z_outliers.reindex(bars.index, fill_value=False)
                rejection_mask |= outlier_mask
                rejection_reasons.loc[outlier_mask] = f'zscore_outlier_{col}'
                self._rejection_stats['outliers'] += outlier_mask.sum()

        return rejection_mask, rejection_reasons

    def _filter_low_volume(self, bars: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Filter bars with low volume."""
        rejection_mask = pd.Series(False, index=bars.index)
        rejection_reasons = pd.Series('', index=bars.index)

        volume_cols = self._get_volume_columns(bars)

        for col in volume_cols:
            if col not in bars.columns:
                continue

            volumes = bars[col].dropna()
            if len(volumes) == 0:
                continue

            # Absolute minimum volume
            low_absolute_mask = volumes < self.config.min_absolute_volume
            abs_mask = low_absolute_mask.reindex(bars.index, fill_value=False)
            rejection_mask |= abs_mask
            rejection_reasons.loc[abs_mask] = f'low_absolute_volume_{col}'
            self._rejection_stats['low_volume'] += abs_mask.sum()

            # Percentile-based volume filter
            if len(volumes) >= 20:  # Need sufficient data for percentile
                volume_threshold = np.percentile(volumes, self.config.min_volume_percentile)
                low_percentile_mask = volumes < volume_threshold

                perc_mask = low_percentile_mask.reindex(bars.index, fill_value=False)
                rejection_mask |= perc_mask
                rejection_reasons.loc[perc_mask] = f'low_percentile_volume_{col}'
                self._rejection_stats['low_volume'] += perc_mask.sum()

        return rejection_mask, rejection_reasons

    def _detect_stale_data(self, bars: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect stale or suspicious data patterns."""
        rejection_mask = pd.Series(False, index=bars.index)
        rejection_reasons = pd.Series('', index=bars.index)

        # Time gap detection
        if isinstance(bars.index, pd.DatetimeIndex):
            time_diffs = bars.index.to_series().diff()
            max_gap = pd.Timedelta(hours=self.config.max_gap_hours)

            gap_mask = time_diffs > max_gap
            rejection_mask |= gap_mask
            rejection_reasons.loc[gap_mask] = 'time_gap'
            self._rejection_stats['stale_gaps'] += gap_mask.sum()

        # Price staleness detection (consecutive bars with same price)
        if 'close' in bars.columns:
            close_prices = bars['close']

            # Find runs of identical prices
            price_changes = close_prices.diff() == 0
            consecutive_counts = price_changes.groupby((~price_changes).cumsum()).cumsum()

            stale_price_mask = consecutive_counts >= self.config.max_price_staleness
            rejection_mask |= stale_price_mask
            rejection_reasons.loc[stale_price_mask] = 'stale_prices'
            self._rejection_stats['stale_prices'] += stale_price_mask.sum()

        return rejection_mask, rejection_reasons

    def _get_price_columns(self, bars: pd.DataFrame) -> List[str]:
        """Get price columns from DataFrame."""
        price_patterns = ['open', 'high', 'low', 'close', 'price', 'adj_close', 'vwap']
        price_cols = []

        for col in bars.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in price_patterns):
                price_cols.append(col)

        return price_cols

    def _get_volume_columns(self, bars: pd.DataFrame) -> List[str]:
        """Get volume columns from DataFrame."""
        volume_patterns = ['volume', 'vol', 'shares']
        volume_cols = []

        for col in bars.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in volume_patterns):
                volume_cols.append(col)

        return volume_cols

    def _update_reasons(self, existing: pd.Series, new: pd.Series) -> pd.Series:
        """Update rejection reasons, combining multiple reasons."""
        combined = existing.copy()

        for idx in new.index:
            if new.loc[idx]:  # If there's a new reason
                if existing.loc[idx]:  # If there's already a reason
                    combined.loc[idx] = f"{existing.loc[idx]},{new.loc[idx]}"
                else:
                    combined.loc[idx] = new.loc[idx]

        return combined

    def _log_rejections(
        self,
        symbol: str,
        rejected_bars: pd.DataFrame,
        reasons: pd.Series
    ) -> None:
        """Log rejected bars for debugging."""
        if len(rejected_bars) == 0:
            return

        self.logger.warning(
            f"Rejected {len(rejected_bars)} bars for {symbol}. "
            f"Reasons: {reasons.value_counts().to_dict()}"
        )

        # Log first few rejected bars for debugging
        if self.logger.isEnabledFor(logging.DEBUG):
            for idx, reason in reasons.head(3).items():
                bar_data = rejected_bars.loc[idx].to_dict()
                self.logger.debug(f"Rejected bar {idx} for {symbol}: {reason} - {bar_data}")

    def _count_rejection_reasons(self, reasons: pd.Series) -> Dict[str, int]:
        """Count rejection reasons for reporting."""
        reason_counts = {}

        for reason_str in reasons:
            if not reason_str:
                continue

            # Handle multiple reasons separated by commas
            individual_reasons = reason_str.split(',')
            for reason in individual_reasons:
                reason = reason.strip()
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return reason_counts

    def _get_time_range(self, bars: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Get time range of bars for reporting."""
        if bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
            return None

        return {
            'start': bars.index.min().isoformat(),
            'end': bars.index.max().isoformat()
        }

    def winsorize_series(
        self,
        series: pd.Series,
        limits: Optional[Tuple[float, float]] = None
    ) -> pd.Series:
        """
        Winsorize a series by capping extreme values.
        
        Args:
            series: Series to winsorize
            limits: (lower_percentile, upper_percentile) as decimals
            
        Returns:
            Winsorized series
        """
        if limits is None:
            limits = self.config.winsorize_limits

        if len(series.dropna()) < 10:  # Need minimum data
            return series

        lower_percentile, upper_percentile = limits
        lower_bound = series.quantile(lower_percentile)
        upper_bound = series.quantile(1 - upper_percentile)

        return series.clip(lower=lower_bound, upper=upper_bound)

    def get_rejection_stats(self) -> Dict[str, Union[int, float]]:
        """Get sanitization statistics."""
        stats = self._rejection_stats.copy()

        if stats['total_processed'] > 0:
            stats['rejection_rate'] = (
                sum(v for k, v in stats.items() if k != 'total_processed') /
                stats['total_processed']
            )
        else:
            stats['rejection_rate'] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset rejection statistics."""
        for key in self._rejection_stats:
            self._rejection_stats[key] = 0


# Global sanitizer instance
_global_sanitizer: Optional[DataSanitizer] = None


def get_data_sanitizer(config: Optional[SanitizationConfig] = None) -> DataSanitizer:
    """Get or create global data sanitizer instance."""
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = DataSanitizer(config)
    return _global_sanitizer


def sanitize_bars(
    bars: pd.DataFrame,
    symbol: str = "UNKNOWN",
    config: Optional[SanitizationConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to sanitize bars.
    
    Args:
        bars: DataFrame with OHLCV data
        symbol: Symbol for logging
        config: Optional sanitization configuration
        
    Returns:
        Tuple of (cleaned_bars, sanitization_report)
    """
    sanitizer = get_data_sanitizer(config)
    return sanitizer.sanitize_bars(bars, symbol)


def winsorize_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    limits: Tuple[float, float] = (0.01, 0.01)
) -> pd.DataFrame:
    """
    Winsorize specified columns of a DataFrame.
    
    Args:
        df: DataFrame to winsorize
        columns: Columns to winsorize (None for all numeric columns)
        limits: (lower_percentile, upper_percentile) as decimals
        
    Returns:
        DataFrame with winsorized columns
    """
    if df.empty:
        return df

    result = df.copy()
    sanitizer = get_data_sanitizer()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            result[col] = sanitizer.winsorize_series(df[col], limits)

    return result
