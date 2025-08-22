from __future__ import annotations

import argparse
import sys

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def run_trade() -> None:
    """Entrypoint for live trading loop."""  # AI-AGENT-REF
    p = argparse.ArgumentParser(description="AI Trading Bot")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--symbols", type=str)
    args = p.parse_args()
    if args.dry_run:
        logger.info("AI Trade: Dry run - exiting")
        return
    from ai_trading.settings import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    from ai_trading import runner

    try:
        runner.run_cycle()
    except KeyboardInterrupt:
        logger.info("Trade interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error("Trade failed: %s", e, exc_info=True)
        sys.exit(1)


def run_backtest() -> None:
    p = argparse.ArgumentParser(description="AI Trading Bot Backtesting")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--symbols", type=str)
    args = p.parse_args()
    if args.dry_run:
        logger.info("AI Backtest: Dry run - exiting")
        return
    from ai_trading.settings import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    from ai_trading import runner

    try:
        runner.run_cycle()
    except KeyboardInterrupt:
        logger.info("Backtest interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error("Backtest failed: %s", e, exc_info=True)
        sys.exit(1)


def run_healthcheck() -> None:
    p = argparse.ArgumentParser(description="AI Trading Bot Health Check")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        logger.info("AI Health: Dry run - exiting")
        return
    from ai_trading.settings import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    from ai_trading.health_monitor import run_health_check

    try:
        run_health_check()
    except KeyboardInterrupt:
        logger.info("Health check interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="AI Trading Bot")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--symbols", type=str)
    args = p.parse_args()
    if args.dry_run:
        logger.info("AI Main: Dry run - exiting")
        return
    from ai_trading.settings import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    from ai_trading import runner

    try:
        runner.run_cycle()
    except KeyboardInterrupt:
        logger.info("Main interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error("Main failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        if "--dry-run" in sys.argv:
            logger.warning("dry-run: ignoring startup exception: %s", e)
            sys.exit(0)
        raise
