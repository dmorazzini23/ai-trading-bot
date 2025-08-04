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


def check_data_freshness(
    df: pd.DataFrame, 
    symbol: str, 
    max_staleness_minutes: int = 15
) -> Dict[str, Union[bool, str, datetime]]:
    """
    Check if market data is fresh enough for trading.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with timestamp index
    symbol : str
        Symbol being checked
    max_staleness_minutes : int
        Maximum allowed staleness in minutes
        
    Returns
    -------
    Dict
        Report with freshness status and details
    """
    now = datetime.now(timezone.utc)
    
    report = {
        'symbol': symbol,
        'is_fresh': False,
        'last_data_time': None,
        'minutes_stale': None,
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
            report['reason'] = f'Data is fresh ({minutes_stale:.1f} minutes old)'
        else:
            report['reason'] = f'Data is stale ({minutes_stale:.1f} minutes old, max {max_staleness_minutes})'
            
    except Exception as e:
        report['reason'] = f'Error checking data freshness: {e}'
        logger.error(f"Data freshness check failed for {symbol}: {e}")
        
    return report


def validate_trading_data(
    data: Dict[str, pd.DataFrame], 
    max_staleness_minutes: int = 15,
    min_data_points: int = 20
) -> Dict[str, Dict]:
    """
    Validate trading data for multiple symbols.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dictionary mapping symbols to their market data
    max_staleness_minutes : int
        Maximum allowed data staleness in minutes
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
            
        # Check for recent data (last 30 minutes)
        freshness = check_data_freshness(df, symbol, max_staleness_minutes=30)
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