"""Market data validation for production trading safety.

Provides comprehensive OHLC validation, price spike detection,
stale data detection, and gap detection to prevent cascading failures
from bad market data.

AI-AGENT-REF: Critical market data validation for production readiness
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Data validation severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Data validation result."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    data_quality_score: float  # 0.0 to 1.0
    details: Dict[str, Any]


class MarketDataValidator:
    """Comprehensive market data validation for production trading."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.max_price_spike_pct = 10.0  # 10% max price spike
        self.max_data_age_minutes = 15   # 15 minutes max data age
        self.min_volume_threshold = 1000  # Minimum volume
        self.max_gap_pct = 5.0           # 5% max price gap
        
        # Quality score weights
        self.quality_weights = {
            'ohlc_consistency': 0.25,
            'price_stability': 0.20,
            'volume_quality': 0.15,
            'data_freshness': 0.15,
            'gap_analysis': 0.15,
            'completeness': 0.10
        }
    
    def validate_ohlc_data(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        """Comprehensive OHLC data validation."""
        try:
            if df.empty:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Empty dataset for {symbol}",
                    data_quality_score=0.0,
                    details={"issue": "empty_data"}
                )
            
            # Required columns check
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing columns for {symbol}: {missing_cols}",
                    data_quality_score=0.0,
                    details={"missing_columns": missing_cols}
                )
            
            validation_scores = {}
            issues = []
            
            # 1. OHLC relationship validation
            ohlc_score, ohlc_issues = self._validate_ohlc_relationships(df)
            validation_scores['ohlc_consistency'] = ohlc_score
            issues.extend(ohlc_issues)
            
            # 2. Price spike detection
            spike_score, spike_issues = self._detect_price_spikes(df, symbol)
            validation_scores['price_stability'] = spike_score
            issues.extend(spike_issues)
            
            # 3. Volume validation
            volume_score, volume_issues = self._validate_volume_patterns(df)
            validation_scores['volume_quality'] = volume_score
            issues.extend(volume_issues)
            
            # 4. Data freshness check
            freshness_score, freshness_issues = self._check_data_freshness(df)
            validation_scores['data_freshness'] = freshness_score
            issues.extend(freshness_issues)
            
            # 5. Gap detection
            gap_score, gap_issues = self._detect_price_gaps(df)
            validation_scores['gap_analysis'] = gap_score
            issues.extend(gap_issues)
            
            # 6. Data completeness
            completeness_score, completeness_issues = self._check_completeness(df)
            validation_scores['completeness'] = completeness_score
            issues.extend(completeness_issues)
            
            # Calculate overall quality score
            quality_score = sum(
                self.quality_weights[category] * score
                for category, score in validation_scores.items()
            )
            
            # Determine overall validation result
            is_valid = quality_score >= 0.7 and not any(
                issue['severity'] in ['error', 'critical'] for issue in issues
            )
            
            # Determine worst severity
            severities = [issue['severity'] for issue in issues]
            if 'critical' in severities:
                severity = ValidationSeverity.CRITICAL
            elif 'error' in severities:
                severity = ValidationSeverity.ERROR
            elif 'warning' in severities:
                severity = ValidationSeverity.WARNING
            else:
                severity = ValidationSeverity.INFO
            
            message = f"Data validation for {symbol}: quality={quality_score:.2f}"
            if issues:
                message += f", {len(issues)} issue(s) found"
            
            return ValidationResult(
                is_valid=is_valid,
                severity=severity,
                message=message,
                data_quality_score=quality_score,
                details={
                    'scores': validation_scores,
                    'issues': issues,
                    'row_count': len(df)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error validating data for {symbol}: {e}")
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation error for {symbol}: {str(e)}",
                data_quality_score=0.0,
                details={"error": str(e)}
            )
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> Tuple[float, List[Dict]]:
        """Validate OHLC price relationships."""
        issues = []
        epsilon = 1e-8
        
        # Check that high >= max(open, close) and low <= min(open, close)
        invalid_high = (df['high'] < df[['open', 'close']].max(axis=1) - epsilon)
        invalid_low = (df['low'] > df[['open', 'close']].min(axis=1) + epsilon)
        
        high_violations = invalid_high.sum()
        low_violations = invalid_low.sum()
        total_violations = high_violations + low_violations
        
        if total_violations > 0:
            issues.append({
                'type': 'ohlc_relationship',
                'severity': 'error' if total_violations > len(df) * 0.05 else 'warning',
                'message': f"OHLC relationship violations: {total_violations} rows",
                'details': {'high_violations': high_violations, 'low_violations': low_violations}
            })
        
        # Check for negative prices
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if negative_prices > 0:
            issues.append({
                'type': 'negative_prices',
                'severity': 'critical',
                'message': f"Negative or zero prices found: {negative_prices} rows",
                'details': {'negative_count': negative_prices}
            })
        
        # Score based on violation percentage
        violation_pct = total_violations / len(df) if len(df) > 0 else 1.0
        score = max(0.0, 1.0 - violation_pct * 10)  # Heavy penalty for violations
        
        return score, issues
    
    def _detect_price_spikes(self, df: pd.DataFrame, symbol: str) -> Tuple[float, List[Dict]]:
        """Detect abnormal price spikes."""
        issues = []
        
        if len(df) < 2:
            return 1.0, issues
        
        # Calculate percentage changes
        close_pct_change = df['close'].pct_change().abs()
        high_pct_change = df['high'].pct_change().abs()
        low_pct_change = df['low'].pct_change().abs()
        
        # Detect spikes above threshold
        spike_threshold = self.max_price_spike_pct / 100.0
        
        close_spikes = (close_pct_change > spike_threshold).sum()
        high_spikes = (high_pct_change > spike_threshold).sum()
        low_spikes = (low_pct_change > spike_threshold).sum()
        
        total_spikes = close_spikes + high_spikes + low_spikes
        
        if total_spikes > 0:
            max_spike = max(
                close_pct_change.max(),
                high_pct_change.max(),
                low_pct_change.max()
            ) * 100
            
            severity = 'critical' if max_spike > 20 else 'warning'
            issues.append({
                'type': 'price_spikes',
                'severity': severity,
                'message': f"Price spikes detected for {symbol}: {total_spikes} occurrences, max: {max_spike:.1f}%",
                'details': {
                    'total_spikes': total_spikes,
                    'max_spike_pct': max_spike,
                    'close_spikes': close_spikes,
                    'high_spikes': high_spikes,
                    'low_spikes': low_spikes
                }
            })
        
        # Score based on spike frequency
        spike_frequency = total_spikes / len(df) if len(df) > 0 else 0
        score = max(0.0, 1.0 - spike_frequency * 20)  # Heavy penalty for frequent spikes
        
        return score, issues
    
    def _validate_volume_patterns(self, df: pd.DataFrame) -> Tuple[float, List[Dict]]:
        """Validate volume patterns and detect anomalies."""
        issues = []
        
        if 'volume' not in df.columns:
            return 0.0, [{'type': 'missing_volume', 'severity': 'error', 
                         'message': 'Volume data missing', 'details': {}}]
        
        volume = df['volume'].fillna(0)
        
        # Check for negative volume
        negative_volume = (volume < 0).sum()
        if negative_volume > 0:
            issues.append({
                'type': 'negative_volume',
                'severity': 'error',
                'message': f"Negative volume found: {negative_volume} rows",
                'details': {'negative_count': negative_volume}
            })
        
        # Check for zero volume
        zero_volume = (volume == 0).sum()
        zero_volume_pct = zero_volume / len(df) if len(df) > 0 else 0
        
        if zero_volume_pct > 0.1:  # More than 10% zero volume
            issues.append({
                'type': 'low_volume',
                'severity': 'warning',
                'message': f"High percentage of zero volume: {zero_volume_pct:.1%}",
                'details': {'zero_volume_pct': zero_volume_pct}
            })
        
        # Check for volume spikes
        if len(volume) > 1:
            volume_changes = volume.pct_change().abs()
            volume_spikes = (volume_changes > 10.0).sum()  # 1000% volume spikes
            
            if volume_spikes > 0:
                issues.append({
                    'type': 'volume_spikes',
                    'severity': 'warning',
                    'message': f"Volume spikes detected: {volume_spikes} occurrences",
                    'details': {'volume_spikes': volume_spikes}
                })
        
        # Score based on volume quality
        volume_quality = 1.0 - zero_volume_pct - (negative_volume / len(df) if len(df) > 0 else 0)
        score = max(0.0, volume_quality)
        
        return score, issues
    
    def _check_data_freshness(self, df: pd.DataFrame) -> Tuple[float, List[Dict]]:
        """Check if data is fresh enough for trading."""
        issues = []
        
        if df.empty:
            return 0.0, [{'type': 'empty_data', 'severity': 'critical',
                         'message': 'No data available', 'details': {}}]
        
        # Get the latest timestamp
        if hasattr(df.index, 'max'):
            latest_time = df.index.max()
        elif 'timestamp' in df.columns:
            latest_time = pd.to_datetime(df['timestamp']).max()
        else:
            # Assume data is current if no timestamp info
            return 1.0, issues
        
        current_time = pd.Timestamp.now(tz='UTC')
        if latest_time.tz is None:
            latest_time = latest_time.tz_localize('UTC')
        
        age_minutes = (current_time - latest_time).total_seconds() / 60
        
        if age_minutes > self.max_data_age_minutes:
            severity = 'critical' if age_minutes > 60 else 'warning'
            issues.append({
                'type': 'stale_data',
                'severity': severity,
                'message': f"Data is stale: {age_minutes:.1f} minutes old",
                'details': {'age_minutes': age_minutes, 'latest_time': latest_time.isoformat()}
            })
        
        # Score based on data age (exponential decay)
        score = np.exp(-age_minutes / self.max_data_age_minutes)
        
        return score, issues
    
    def _detect_price_gaps(self, df: pd.DataFrame) -> Tuple[float, List[Dict]]:
        """Detect significant price gaps between periods."""
        issues = []
        
        if len(df) < 2:
            return 1.0, issues
        
        # Calculate gaps (difference between current open and previous close)
        prev_close = df['close'].shift(1)
        current_open = df['open']
        
        gaps = ((current_open - prev_close) / prev_close).abs()
        gap_threshold = self.max_gap_pct / 100.0
        
        significant_gaps = (gaps > gap_threshold).sum()
        
        if significant_gaps > 0:
            max_gap = gaps.max() * 100
            issues.append({
                'type': 'price_gaps',
                'severity': 'warning' if max_gap < 15 else 'error',
                'message': f"Significant price gaps detected: {significant_gaps} occurrences, max: {max_gap:.1f}%",
                'details': {
                    'gap_count': significant_gaps,
                    'max_gap_pct': max_gap,
                    'avg_gap_pct': gaps.mean() * 100
                }
            })
        
        # Score based on gap frequency
        gap_frequency = significant_gaps / len(df) if len(df) > 0 else 0
        score = max(0.0, 1.0 - gap_frequency * 10)
        
        return score, issues
    
    def _check_completeness(self, df: pd.DataFrame) -> Tuple[float, List[Dict]]:
        """Check data completeness (missing values)."""
        issues = []
        
        if df.empty:
            return 0.0, [{'type': 'empty_data', 'severity': 'critical',
                         'message': 'No data available', 'details': {}}]
        
        # Check for missing values in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        missing_data = {}
        
        for col in critical_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = missing_count / len(df)
                
                if missing_pct > 0:
                    missing_data[col] = {
                        'count': missing_count,
                        'percentage': missing_pct * 100
                    }
        
        if missing_data:
            total_missing_pct = sum(data['percentage'] for data in missing_data.values()) / len(missing_data)
            severity = 'critical' if total_missing_pct > 10 else 'warning'
            
            issues.append({
                'type': 'missing_data',
                'severity': severity,
                'message': f"Missing data detected: {total_missing_pct:.1f}% average across critical columns",
                'details': missing_data
            })
        
        # Score based on completeness
        total_missing_rate = sum(
            df[col].isna().sum() / len(df) for col in critical_cols if col in df.columns
        ) / len(critical_cols)
        
        score = max(0.0, 1.0 - total_missing_rate)
        
        return score, issues
    
    def fill_price_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill price gaps using forward fill and interpolation."""
        if df.empty:
            return df
        
        df_filled = df.copy()
        
        # Forward fill first
        df_filled = df_filled.fillna(method='ffill')
        
        # Then interpolate remaining gaps
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].interpolate(method='linear')
        
        # Fill any remaining NaN with the last valid value
        df_filled = df_filled.fillna(method='bfill')
        
        return df_filled
    
    def clean_anomalous_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean anomalous data points using statistical methods."""
        if df.empty:
            return df
        
        df_cleaned = df.copy()
        
        # Remove extreme outliers using IQR method
        for col in ['open', 'high', 'low', 'close']:
            if col in df_cleaned.columns:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds (3 * IQR)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers at bounds
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Ensure OHLC relationships are maintained after cleaning
        df_cleaned['high'] = df_cleaned[['open', 'high', 'low', 'close']].max(axis=1)
        df_cleaned['low'] = df_cleaned[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Log cleaning actions
        if not df.equals(df_cleaned):
            self.logger.info(f"Cleaned anomalous data for {symbol}")
        
        return df_cleaned


# Global validator instance
_market_data_validator: Optional[MarketDataValidator] = None


def get_market_data_validator() -> MarketDataValidator:
    """Get or create global market data validator instance."""
    global _market_data_validator
    if _market_data_validator is None:
        _market_data_validator = MarketDataValidator()
    return _market_data_validator


def validate_market_data(df: pd.DataFrame, symbol: str) -> ValidationResult:
    """Convenience function for validating market data."""
    validator = get_market_data_validator()
    return validator.validate_ohlc_data(df, symbol)


def clean_market_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convenience function for cleaning market data."""
    validator = get_market_data_validator()
    
    # First fill gaps
    df_filled = validator.fill_price_gaps(df)
    
    # Then clean anomalies
    df_cleaned = validator.clean_anomalous_data(df_filled, symbol)
    
    return df_cleaned