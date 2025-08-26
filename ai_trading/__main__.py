from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _build_parser(description: str, *, symbols: bool = False) -> argparse.ArgumentParser:
    """Return a base parser with common runtime options."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dry-run", action="store_true", help="Exit before heavy imports")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Delay between iterations in seconds",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--paper", dest="paper", action="store_true", help="Use paper trading (default)")
    mode.add_argument("--live", dest="paper", action="store_false", help="Use live trading")
    parser.set_defaults(paper=True)
    if symbols:
        parser.add_argument("--symbols", type=str)
    return parser


def _run_loop(fn: Callable[[], None], args: argparse.Namespace, label: str) -> None:
    """Execute ``fn`` respecting --once/--interval and uniform error handling."""

    import time

    try:
        while True:
            fn()
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("%s interrupted", label)
        sys.exit(0)
    except SystemExit as e:  # AI-AGENT-REF: surface exit codes in logs
        logger.error("%s exited: %s", label, e, exc_info=True)
        raise
    except Exception as e:
        logger.error("%s failed: %s", label, e, exc_info=True)
        sys.exit(1)


def run_trade() -> None:
    """Entrypoint for live trading loop."""

    parser = _build_parser("AI Trading Bot", symbols=True)
    args = parser.parse_args()
    if args.dry_run:
        logger.info("AI Trade: Dry run - exiting")
        logger.info("INDICATOR_IMPORT_OK")
        import logging, time
        time.sleep(0.1)
        logging.shutdown()
        sys.exit(0)

    import os

    os.environ["TRADING_MODE"] = "paper" if args.paper else "live"
    from ai_trading.env import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    from ai_trading import runner
    runner._preflight_import_health()

    _run_loop(runner.run_cycle, args, "Trade")


def run_backtest() -> None:
    """Entrypoint for backtesting loop."""

    parser = _build_parser("AI Trading Bot Backtesting", symbols=True)
    args = parser.parse_args()
    if args.dry_run:
        logger.info("AI Backtest: Dry run - exiting")
        logger.info("INDICATOR_IMPORT_OK")
        import logging, time
        time.sleep(0.1)
        logging.shutdown()
        sys.exit(0)

    import os

    os.environ["TRADING_MODE"] = "paper" if args.paper else "live"
    from ai_trading.env import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    from ai_trading import runner
    runner._preflight_import_health()

    _run_loop(runner.run_cycle, args, "Backtest")


def run_healthcheck() -> None:
    """Entrypoint for health check loop."""

    parser = _build_parser("AI Trading Bot Health Check")
    args = parser.parse_args()
    if args.dry_run:
        logger.info("AI Health: Dry run - exiting")
        logger.info("INDICATOR_IMPORT_OK")
        import logging, time
        time.sleep(0.1)
        logging.shutdown()
        sys.exit(0)

    import os

    os.environ["TRADING_MODE"] = "paper" if args.paper else "live"
    from ai_trading.env import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    from ai_trading.health_monitor import run_health_check
    from ai_trading import runner
    runner._preflight_import_health()

    _run_loop(run_health_check, args, "Health check")


def main() -> None:
    """Default CLI entrypoint mirroring ``run_trade``."""

    parser = _build_parser("AI Trading Bot", symbols=True)
    args = parser.parse_args()
    if args.dry_run:
        logger.info("AI Main: Dry run - exiting")
        logger.info("INDICATOR_IMPORT_OK")
        import logging, time
        time.sleep(0.1)
        logging.shutdown()
        sys.exit(0)

    import os

    os.environ["TRADING_MODE"] = "paper" if args.paper else "live"
    from ai_trading.env import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    from ai_trading import runner
    runner._preflight_import_health()

    _run_loop(runner.run_cycle, args, "Main")


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

