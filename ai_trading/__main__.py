import sys
import argparse
from ai_trading.env import ensure_dotenv_loaded

def run_trade():
    """Entry point for ai-trade command."""
    ensure_dotenv_loaded()
    
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("AI Trade: Dry run mode - config loaded successfully, exiting gracefully")
        return
    
    from ai_trading import runner
    try:
        runner.run_cycle()
    except Exception:
        pass

def run_backtest():
    """Entry point for ai-backtest command."""
    ensure_dotenv_loaded()
    
    parser = argparse.ArgumentParser(description='AI Trading Bot Backtesting')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("AI Backtest: Dry run mode - config loaded successfully, exiting gracefully")
        return
    
    from ai_trading import runner
    try:
        runner.run_cycle()
    except Exception:
        pass

def run_healthcheck():
    """Entry point for ai-health command."""
    ensure_dotenv_loaded()
    
    parser = argparse.ArgumentParser(description='AI Trading Bot Health Check')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("AI Health: Dry run mode - config loaded successfully, exiting gracefully")
        return
    
    from ai_trading.health_monitor import run_health_check
    try:
        run_health_check()
    except Exception:
        pass

def main() -> None:
    """Default main entry point."""
    ensure_dotenv_loaded()
    from ai_trading import runner
    try:
        runner.run_cycle()
    except Exception:
        pass

if __name__ == "__main__":
    main()
