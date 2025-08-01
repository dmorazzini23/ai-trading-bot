#!/usr/bin/env python3
"""
AI-AGENT-REF: Comprehensive health check script for AI Trading Bot
Monitors system health, dependencies, data connectivity, and trading status.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging for health checks."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/health_check.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def check_system_dependencies() -> Tuple[bool, str]:
    """Check if system dependencies are properly installed."""
    checks = []
    
    # Check Python version
    if sys.version_info[:3] == (3, 12, 3):
        checks.append("✅ Python 3.12.3")
    else:
        checks.append(f"❌ Python version mismatch: {sys.version_info}")
        return False, "\n".join(checks)
    
    # Check TA-Lib
    try:
        import talib
        checks.append("✅ TA-Lib C library and Python package")
    except ImportError:
        checks.append("⚠️  TA-Lib not available - using fallback implementation")
    
    # Check critical Python packages
    critical_packages = [
        'pandas', 'numpy', 'requests', 'alpaca', 'scikit-learn'
    ]
    
    for package in critical_packages:
        try:
            __import__(package)
            checks.append(f"✅ {package}")
        except ImportError:
            checks.append(f"❌ {package} not available")
            return False, "\n".join(checks)
    
    return True, "\n".join(checks)

def check_configuration() -> Tuple[bool, str]:
    """Check if configuration is properly set up."""
    checks = []
    
    try:
        import config
        
        # Check critical environment variables
        required_vars = [
            'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(config, var, None):
                missing_vars.append(var)
        
        if missing_vars:
            checks.append(f"❌ Missing required config: {', '.join(missing_vars)}")
            return False, "\n".join(checks)
        else:
            checks.append("✅ Required configuration variables set")
        
        # Check file paths
        log_file = getattr(config, 'TRADE_LOG_FILE', None)
        if log_file:
            log_dir = os.path.dirname(log_file)
            if os.path.exists(log_dir) and os.access(log_dir, os.W_OK):
                checks.append(f"✅ Trade log directory writable: {log_dir}")
            else:
                checks.append(f"❌ Trade log directory not writable: {log_dir}")
                return False, "\n".join(checks)
        
        return True, "\n".join(checks)
        
    except Exception as e:
        checks.append(f"❌ Configuration error: {e}")
        return False, "\n".join(checks)

def check_data_connectivity() -> Tuple[bool, str]:
    """Check connectivity to data sources."""
    checks = []
    
    try:
        # Test basic internet connectivity
        import requests
        
        # Test Alpaca connectivity
        response = requests.get("https://api.alpaca.markets", timeout=10)
        if response.status_code in [200, 401, 403]:  # 401/403 are fine, means server is up
            checks.append("✅ Alpaca API connectivity")
        else:
            checks.append(f"❌ Alpaca API unreachable: {response.status_code}")
            return False, "\n".join(checks)
        
        # Test data fetcher
        try:
            from data_fetcher import fetch_market_hours
            market_info = fetch_market_hours()
            if market_info:
                checks.append("✅ Market hours data fetcher")
            else:
                checks.append("⚠️  Market hours data unavailable")
        except Exception as e:
            checks.append(f"⚠️  Data fetcher issue: {str(e)[:100]}")
        
        return True, "\n".join(checks)
        
    except Exception as e:
        checks.append(f"❌ Data connectivity error: {e}")
        return False, "\n".join(checks)

def check_file_permissions() -> Tuple[bool, str]:
    """Check file and directory permissions."""
    checks = []
    
    # Check critical directories
    dirs_to_check = [
        'data', 'logs', 'models'
    ]
    
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            if os.access(dir_name, os.R_OK | os.W_OK):
                checks.append(f"✅ {dir_name}/ directory permissions")
            else:
                checks.append(f"❌ {dir_name}/ directory not writable")
                return False, "\n".join(checks)
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                checks.append(f"✅ Created {dir_name}/ directory")
            except OSError as e:
                checks.append(f"❌ Cannot create {dir_name}/ directory: {e}")
                return False, "\n".join(checks)
    
    # Check specific files
    try:
        import config
        trade_log_file = getattr(config, 'TRADE_LOG_FILE', 'data/trades.csv')
        
        # Try to create/write to trade log file
        log_dir = os.path.dirname(trade_log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        with open(trade_log_file, 'a') as f:
            f.write(f"# Health check at {datetime.now(timezone.utc).isoformat()}\n")
        
        checks.append(f"✅ Trade log file writable: {trade_log_file}")
        
    except Exception as e:
        checks.append(f"❌ Trade log file permission error: {e}")
        return False, "\n".join(checks)
    
    return True, "\n".join(checks)

def check_trading_system() -> Tuple[bool, str]:
    """Check trading system components."""
    checks = []
    
    try:
        # Check bot engine import
        from bot_engine import BotContext
        checks.append("✅ Bot engine import")
        
        # Check execution engine
        from trade_execution import ExecutionEngine
        checks.append("✅ Trade execution engine")
        
        # Check signal generation
        from signals import calculate_technical_indicators
        checks.append("✅ Signal generation")
        
        # Check risk management
        from risk_engine import RiskEngine
        checks.append("✅ Risk management engine")
        
        return True, "\n".join(checks)
        
    except Exception as e:
        checks.append(f"❌ Trading system component error: {e}")
        return False, "\n".join(checks)

def run_comprehensive_health_check() -> Dict[str, Any]:
    """Run all health checks and return results."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("AI Trading Bot - Comprehensive Health Check")
    logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)
    
    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'overall_status': 'HEALTHY',
        'checks': {}
    }
    
    # Define all health checks
    health_checks = [
        ('System Dependencies', check_system_dependencies),
        ('Configuration', check_configuration),
        ('Data Connectivity', check_data_connectivity),
        ('File Permissions', check_file_permissions),
        ('Trading System', check_trading_system),
    ]
    
    failed_checks = []
    
    for check_name, check_func in health_checks:
        logger.info(f"\n🔍 {check_name}:")
        try:
            status, details = check_func()
            results['checks'][check_name] = {
                'status': 'PASS' if status else 'FAIL',
                'details': details
            }
            
            logger.info(details)
            
            if not status:
                failed_checks.append(check_name)
                
        except Exception as e:
            error_msg = f"Check failed with exception: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            results['checks'][check_name] = {
                'status': 'ERROR',
                'details': error_msg
            }
            failed_checks.append(check_name)
    
    # Overall status
    if failed_checks:
        results['overall_status'] = 'UNHEALTHY'
        logger.error(f"\n❌ HEALTH CHECK FAILED - Issues found in: {', '.join(failed_checks)}")
    else:
        logger.info(f"\n✅ ALL HEALTH CHECKS PASSED - System is operational")
    
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    import json
    
    # Run health check
    results = run_comprehensive_health_check()
    
    # Save results to file
    os.makedirs('logs', exist_ok=True)
    with open('logs/health_check_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Exit with appropriate code
    if results['overall_status'] == 'HEALTHY':
        sys.exit(0)
    else:
        sys.exit(1)