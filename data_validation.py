"""
Data validation utilities for trading bot.

Provides functions to validate data freshness, quality, and trading readiness.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# Use centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    logger = logging.getLogger(__name__)

# AI-AGENT-REF: Data staleness validation to prevent trading on stale data


def is_market_hours(current_time: datetime = None) -> bool:
    """
    Check if current time is during market trading hours.
    
    Parameters
    ----------
    current_time : datetime, optional
        Time to check, defaults to current UTC time
        
    Returns
    -------
    bool
        True if during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    # Convert to ET (Eastern Time)
    try:
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        et_time = current_time.astimezone(et_tz)
    except ImportError:
        # Fallback: approximate ET as UTC-5 (ignoring DST for simplicity)
        et_time = current_time.replace(tzinfo=timezone.utc) - timedelta(hours=5)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if et_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's during trading hours (9:30 AM - 4:00 PM ET)
    market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= et_time <= market_close


def get_staleness_threshold(symbol: str = None, current_time: datetime = None) -> int:
    """
    Get appropriate staleness threshold based on market conditions.
    
    Parameters
    ----------
    symbol : str, optional
        Symbol being checked (for future symbol-specific logic)
    current_time : datetime, optional
        Current time, defaults to now
        
    Returns
    -------
    int
        Staleness threshold in minutes
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    # During market hours: stricter threshold
    if is_market_hours(current_time):
        return 15  # 15 minutes during active trading
    
    # After hours/weekends: more lenient threshold
    # Data can be older since markets are closed
    try:
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        et_time = current_time.astimezone(et_tz)
    except ImportError:
        et_time = current_time.replace(tzinfo=timezone.utc) - timedelta(hours=5)
    
    # Weekend: very lenient (data from Friday close is acceptable)
    if et_time.weekday() >= 5:
        return 4320  # 72 hours (3 days) for weekend
    
    # Weekday after hours: moderately lenient
    return 60  # 1 hour after market close


def check_data_freshness(
    df: pd.DataFrame, 
    symbol: str, 
    max_staleness_minutes: int = None
) -> Dict[str, Union[bool, str, datetime]]:
    """
    Check if market data is fresh enough for trading.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with timestamp index
    symbol : str
        Symbol being checked
    max_staleness_minutes : int, optional
        Maximum allowed staleness in minutes. If None, uses intelligent 
        defaults based on market hours and time of day.
        
    Returns
    -------
    Dict
        Report with freshness status and details
    """
    now = datetime.now(timezone.utc)
    
    # Use intelligent default if not specified
    if max_staleness_minutes is None:
        max_staleness_minutes = get_staleness_threshold(symbol, now)
    
    report = {
        'symbol': symbol,
        'is_fresh': False,
        'last_data_time': None,
        'minutes_stale': None,
        'staleness_threshold': max_staleness_minutes,
        'market_hours': is_market_hours(now),
        'reason': None,
        'current_time': now
    }
    
    try:
        if df.empty:
            report['reason'] = 'No data available'
            return report
            
        # Get the most recent timestamp
        if hasattr(df.index, 'tz_localize') or hasattr(df.index, 'tz_convert'):
            # Ensure timezone-aware comparison
            if df.index.tz is None:
                last_time = df.index[-1].tz_localize('UTC')
            else:
                last_time = df.index[-1].tz_convert(timezone.utc)
        else:
            # Fallback for simple datetime index
            last_time = df.index[-1]
            if not hasattr(last_time, 'tzinfo') or last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
        
        report['last_data_time'] = last_time
        time_diff = now - last_time
        minutes_stale = time_diff.total_seconds() / 60
        report['minutes_stale'] = minutes_stale
        
        if minutes_stale <= max_staleness_minutes:
            report['is_fresh'] = True
            market_status = "during market hours" if report['market_hours'] else "outside market hours"
            report['reason'] = f'Data is fresh ({minutes_stale:.1f} minutes old, {market_status})'
        else:
            market_status = "during market hours" if report['market_hours'] else "outside market hours"
            report['reason'] = f'Data is stale ({minutes_stale:.1f} minutes old, max {max_staleness_minutes} {market_status})'
            
    except Exception as e:
        report['reason'] = f'Error checking data freshness: {e}'
        logger.error(f"Data freshness check failed for {symbol}: {e}")
        
    return report


def validate_trading_data(
    data: Dict[str, pd.DataFrame], 
    max_staleness_minutes: int = None,
    min_data_points: int = 20
) -> Dict[str, Dict]:
    """
    Validate trading data for multiple symbols.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dictionary mapping symbols to their market data
    max_staleness_minutes : int, optional
        Maximum allowed data staleness in minutes. If None, uses intelligent
        defaults based on market hours and time of day.
    min_data_points : int
        Minimum required data points per symbol
        
    Returns
    -------
    Dict[str, Dict]
        Validation report for each symbol
    """
    validation_results = {}
    
    for symbol, df in data.items():
        result = check_data_freshness(df, symbol, max_staleness_minutes)
        
        # Additional validations
        if df.empty:
            result['has_sufficient_data'] = False
            result['data_points'] = 0
        else:
            result['data_points'] = len(df)
            result['has_sufficient_data'] = len(df) >= min_data_points
            
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        result['has_required_columns'] = len(missing_columns) == 0
        result['missing_columns'] = missing_columns
        
        # Overall trading readiness
        result['trading_ready'] = (
            result['is_fresh'] and 
            result['has_sufficient_data'] and 
            result['has_required_columns']
        )
        
        validation_results[symbol] = result
        
    return validation_results


def get_stale_symbols(validation_results: Dict[str, Dict]) -> List[str]:
    """
    Get list of symbols with stale data.
    
    Parameters
    ----------
    validation_results : Dict[str, Dict]
        Results from validate_trading_data
        
    Returns
    -------
    List[str]
        List of symbols with stale or invalid data
    """
    stale_symbols = []
    
    for symbol, result in validation_results.items():
        if not result.get('trading_ready', False):
            stale_symbols.append(symbol)
            
    return stale_symbols


def log_data_validation_summary(validation_results: Dict[str, Dict]) -> None:
    """
    Log a summary of data validation results.
    
    Parameters
    ----------
    validation_results : Dict[str, Dict]
        Results from validate_trading_data
    """
    total_symbols = len(validation_results)
    fresh_symbols = sum(1 for r in validation_results.values() if r.get('is_fresh', False))
    ready_symbols = sum(1 for r in validation_results.values() if r.get('trading_ready', False))
    stale_symbols = get_stale_symbols(validation_results)
    
    logger.info(f"Data validation summary: {ready_symbols}/{total_symbols} symbols ready for trading")
    logger.info(f"Fresh data: {fresh_symbols}/{total_symbols} symbols")
    
    if stale_symbols:
        logger.warning(f"Stale data detected for {len(stale_symbols)} symbols: {stale_symbols}")
        
        # Log detailed reasons for stale data
        for symbol in stale_symbols[:5]:  # Limit to first 5 for log brevity
            result = validation_results[symbol]
            logger.warning(f"{symbol}: {result.get('reason', 'Unknown issue')}")
            
        if len(stale_symbols) > 5:
            logger.warning(f"... and {len(stale_symbols) - 5} more symbols with stale data")


def should_halt_trading(validation_results: Dict[str, Dict], max_stale_ratio: float = 0.5) -> bool:
    """
    Determine if trading should be halted due to too many stale symbols.
    
    Parameters
    ----------
    validation_results : Dict[str, Dict]
        Results from validate_trading_data
    max_stale_ratio : float
        Maximum ratio of stale symbols before halting trading
        
    Returns
    -------
    bool
        True if trading should be halted due to data quality issues
    """
    if not validation_results:
        logger.error("No validation results available - halting trading")
        return True
        
    total_symbols = len(validation_results)
    ready_symbols = sum(1 for r in validation_results.values() if r.get('trading_ready', False))
    stale_ratio = 1.0 - (ready_symbols / total_symbols)
    
    if stale_ratio > max_stale_ratio:
        logger.error(
            f"Too many symbols have stale data ({stale_ratio:.1%} > {max_stale_ratio:.1%}) - "
            f"halting trading to prevent losses"
        )
        return True
        
    return False


# AI-AGENT-REF: Emergency data validation for critical trading decisions
def emergency_data_check(df: pd.DataFrame, symbol: str) -> bool:
    """
    Perform emergency data validation before critical trades.
    
    This is a fast, lightweight check for the most critical data issues
    that could cause immediate trading losses.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data to validate
    symbol : str
        Symbol being validated
        
    Returns
    -------
    bool
        True if data passes emergency validation, False if trading should be blocked
    """
    try:
        # Basic data existence check
        if df.empty:
            logger.error(f"EMERGENCY: No data for {symbol} - blocking trade")
            return False
            
        # Check for recent data with emergency thresholds (more conservative than normal)
        # Use 2x the normal threshold for emergency checks, but cap at 60 minutes
        emergency_threshold = min(get_staleness_threshold(symbol) * 2, 60)
        freshness = check_data_freshness(df, symbol, max_staleness_minutes=emergency_threshold)
        if not freshness['is_fresh']:
            logger.error(f"EMERGENCY: Stale data for {symbol} - {freshness['reason']} - blocking trade")
            return False
            
        # Check for valid price data
        if 'Close' in df.columns:
            last_price = df['Close'].iloc[-1]
            if pd.isna(last_price) or last_price <= 0:
                logger.error(f"EMERGENCY: Invalid price data for {symbol} (price: {last_price}) - blocking trade")
                return False
                
        logger.debug(f"Emergency data check passed for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"EMERGENCY: Data validation error for {symbol}: {e} - blocking trade")
        return False


# AI-AGENT-REF: Enhanced data integrity monitoring system
def validate_trade_log_integrity(trade_log_path: str) -> Dict[str, Union[bool, List, str]]:
    """
    Comprehensive trade log integrity validation.
    
    Validates trade log file format, data consistency, and detects corruption.
    
    Parameters
    ----------
    trade_log_path : str
        Path to the trade log CSV file
        
    Returns
    -------
    Dict
        Comprehensive integrity report with validation results
    """
    integrity_report = {
        'file_exists': False,
        'file_readable': False,
        'valid_format': False,
        'data_consistent': False,
        'total_trades': 0,
        'corrupted_rows': [],
        'missing_columns': [],
        'data_quality_issues': [],
        'recommendations': [],
        'integrity_score': 0.0,
        'validation_timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    try:
        from pathlib import Path
        
        # Check file existence
        if not Path(trade_log_path).exists():
            integrity_report['recommendations'].append(f"Create trade log file: {trade_log_path}")
            return integrity_report
        
        integrity_report['file_exists'] = True
        
        # Check file readability
        try:
            df = pd.read_csv(trade_log_path)
            integrity_report['file_readable'] = True
            integrity_report['total_trades'] = len(df)
        except Exception as e:
            integrity_report['data_quality_issues'].append(f"File read error: {e}")
            integrity_report['recommendations'].append("Check file format and encoding")
            return integrity_report
        
        # Validate CSV structure and required columns
        required_columns = ['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl']
        missing_cols = [col for col in required_columns if col not in df.columns]
        integrity_report['missing_columns'] = missing_cols
        
        if not missing_cols:
            integrity_report['valid_format'] = True
        else:
            integrity_report['data_quality_issues'].append(f"Missing columns: {missing_cols}")
            integrity_report['recommendations'].append("Update trade logging to include all required columns")
        
        # Data consistency validation
        if integrity_report['valid_format'] and len(df) > 0:
            corrupted_rows = []
            data_issues = []
            
            for idx, row in df.iterrows():
                issues = []
                
                # Validate prices
                try:
                    entry_price = float(row['entry_price'])
                    exit_price = float(row['exit_price'])
                    
                    if entry_price <= 0:
                        issues.append("invalid_entry_price")
                    if exit_price <= 0:
                        issues.append("invalid_exit_price")
                    if entry_price > 50000 or exit_price > 50000:
                        issues.append("extreme_high_price")
                        
                except (ValueError, TypeError):
                    issues.append("non_numeric_prices")
                
                # Validate quantities
                try:
                    qty = float(row['quantity'])
                    if qty <= 0:
                        issues.append("invalid_quantity")
                except (ValueError, TypeError):
                    issues.append("non_numeric_quantity")
                
                # Validate side
                if row['side'] not in ['buy', 'sell']:
                    issues.append("invalid_side")
                
                # Validate timestamp
                try:
                    pd.to_datetime(row['timestamp'])
                except:
                    issues.append("invalid_timestamp")
                
                if issues:
                    corrupted_rows.append({'row': idx, 'issues': issues})
            
            integrity_report['corrupted_rows'] = corrupted_rows
            
            if len(corrupted_rows) == 0:
                integrity_report['data_consistent'] = True
            else:
                corruption_rate = len(corrupted_rows) / len(df)
                data_issues.append(f"Data corruption in {len(corrupted_rows)} rows ({corruption_rate:.1%})")
                
                if corruption_rate > 0.1:  # More than 10% corrupted
                    integrity_report['recommendations'].append("Investigate data logging system for corruption source")
                
            integrity_report['data_quality_issues'].extend(data_issues)
        
        # Calculate integrity score
        score = 0.0
        if integrity_report['file_exists']:
            score += 0.2
        if integrity_report['file_readable']:
            score += 0.2
        if integrity_report['valid_format']:
            score += 0.3
        if integrity_report['data_consistent']:
            score += 0.3
        
        integrity_report['integrity_score'] = score
        
        # Add recommendations based on score
        if score < 0.7:
            integrity_report['recommendations'].append("Trade log integrity is compromised - manual review required")
        elif score < 0.9:
            integrity_report['recommendations'].append("Trade log has minor issues - monitor closely")
        
    except Exception as e:
        integrity_report['data_quality_issues'].append(f"Validation error: {e}")
        integrity_report['recommendations'].append("Check trade log file system and permissions")
    
    return integrity_report


def monitor_real_time_data_quality(price_data: Dict[str, float], volume_data: Dict[str, float] = None) -> Dict[str, Union[bool, List]]:
    """
    Real-time data quality monitoring for live trading.
    
    Performs rapid validation of incoming price and volume data to detect
    anomalies that could indicate data feed issues or corruption.
    
    Parameters
    ----------
    price_data : Dict[str, float]
        Current price data by symbol
    volume_data : Dict[str, float], optional
        Current volume data by symbol
        
    Returns
    -------
    Dict
        Real-time quality assessment
    """
    quality_report = {
        'data_quality_ok': True,
        'anomalies_detected': [],
        'warning_symbols': [],
        'critical_symbols': [],
        'recommendations': []
    }
    
    try:
        for symbol, price in price_data.items():
            symbol_issues = []
            
            # Price validation
            if pd.isna(price) or price <= 0:
                symbol_issues.append("invalid_price")
                quality_report['critical_symbols'].append(symbol)
                
            elif price > 50000:  # Extremely high price
                symbol_issues.append("extreme_high_price")
                quality_report['warning_symbols'].append(symbol)
                
            elif price < 0.01:  # Extremely low price
                symbol_issues.append("extreme_low_price")
                quality_report['warning_symbols'].append(symbol)
            
            # Volume validation (if provided)
            if volume_data and symbol in volume_data:
                volume = volume_data[symbol]
                if pd.isna(volume) or volume < 0:
                    symbol_issues.append("invalid_volume")
                    quality_report['warning_symbols'].append(symbol)
            
            if symbol_issues:
                quality_report['anomalies_detected'].append({
                    'symbol': symbol,
                    'issues': symbol_issues,
                    'price': price,
                    'volume': volume_data.get(symbol) if volume_data else None
                })
        
        # Set overall quality status
        if quality_report['critical_symbols']:
            quality_report['data_quality_ok'] = False
            quality_report['recommendations'].append("Critical data issues detected - halt trading for affected symbols")
        
        elif len(quality_report['warning_symbols']) > len(price_data) * 0.2:  # More than 20% have warnings
            quality_report['data_quality_ok'] = False
            quality_report['recommendations'].append("Widespread data quality issues - review data feed")
        
        elif quality_report['warning_symbols']:
            quality_report['recommendations'].append("Monitor data quality closely for warning symbols")
        
    except Exception as e:
        quality_report['data_quality_ok'] = False
        quality_report['anomalies_detected'].append({
            'symbol': 'SYSTEM',
            'issues': ['validation_error'],
            'error': str(e)
        })
        quality_report['recommendations'].append("Data quality monitoring system error - manual review required")
    
    return quality_report


def create_data_recovery_procedures(issues: List[str]) -> List[str]:
    """
    Generate specific data recovery procedures based on detected issues.
    
    Parameters
    ----------
    issues : List[str]
        List of data quality issues detected
        
    Returns
    -------
    List[str]
        Ordered list of recovery procedures to execute
    """
    procedures = []
    
    if "missing_file" in issues or "file_not_readable" in issues:
        procedures.extend([
            "1. Check file system permissions and disk space",
            "2. Verify trade logging service is running",
            "3. Restore from backup if available",
            "4. Initialize new trade log with proper format"
        ])
    
    if "missing_columns" in issues or "invalid_format" in issues:
        procedures.extend([
            "1. Backup current trade log file",
            "2. Update trade logging code to include required columns",
            "3. Migrate existing data to new format",
            "4. Validate new format before resuming trading"
        ])
    
    if "data_corruption" in issues:
        procedures.extend([
            "1. Identify corruption source (data feed, storage, logging code)",
            "2. Clean corrupted records or restore from clean backup",
            "3. Implement additional data validation in logging pipeline",
            "4. Monitor for recurring corruption patterns"
        ])
    
    if "stale_data" in issues:
        procedures.extend([
            "1. Check data feed connection and latency",
            "2. Verify system clock synchronization",
            "3. Restart data collection services if needed",
            "4. Implement data freshness alerts"
        ])
    
    if not procedures:
        procedures.append("No specific issues detected - continue monitoring")
    
    return procedures