import argparse
import logging
import sys

from ai_trading.env import ensure_dotenv_loaded

logger = logging.getLogger(__name__)


def run_trade():
    """Entry point for ai-trade command."""
    ensure_dotenv_loaded()

    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")

    args = parser.parse_args()

    if args.dry_run:
        logger.info(
            "AI Trade: Dry run mode - config loaded successfully, exiting gracefully"
        )
        return

    from ai_trading import runner

    try:
        runner.run_cycle()
    except KeyboardInterrupt:
        logger.info("Trade execution interrupted by user")
        sys.exit(0)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(
            "Trade execution failed - module import error: %s",
            e,
            extra={"component": "trade_entry", "error_type": "import"},
        )
        sys.exit(1)
    except OSError as e:
        logger.error(
            "Trade execution failed - I/O error: %s",
            e,
            extra={"component": "trade_entry", "error_type": "io"},
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Trade execution failed - unexpected error: %s",
            e,
            exc_info=True,
            extra={"component": "trade_entry", "error_type": "unexpected"},
        )
        sys.exit(1)


def run_backtest():
    """Entry point for ai-backtest command."""
    ensure_dotenv_loaded()

    parser = argparse.ArgumentParser(description="AI Trading Bot Backtesting")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")

    args = parser.parse_args()

    if args.dry_run:
        logger.info(
            "AI Backtest: Dry run mode - config loaded successfully, exiting gracefully"
        )
        return

    from ai_trading import runner

    try:
        runner.run_cycle()
    except KeyboardInterrupt:
        logger.info("Backtest execution interrupted by user")
        sys.exit(0)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(
            "Backtest execution failed - module import error: %s",
            e,
            extra={
                "component": "backtest_entry",
                "error_type": "import",
            },
        )
        sys.exit(1)
    except OSError as e:
        logger.error(
            "Backtest execution failed - I/O error: %s",
            e,
            extra={
                "component": "backtest_entry",
                "error_type": "io",
            },
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Backtest execution failed - unexpected error: %s",
            e,
            exc_info=True,
            extra={
                "component": "backtest_entry",
                "error_type": "unexpected",
            },
        )
        sys.exit(1)


def run_healthcheck():
    """Entry point for ai-health command."""
    ensure_dotenv_loaded()

    parser = argparse.ArgumentParser(description="AI Trading Bot Health Check")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")

    args = parser.parse_args()

    if args.dry_run:
        logger.info(
            "AI Health: Dry run mode - config loaded successfully, exiting gracefully"
        )
        return

    from ai_trading.health_monitor import run_health_check

    try:
        run_health_check()
    except KeyboardInterrupt:
        logger.info("Health check interrupted by user")
        sys.exit(0)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(
            "Health check failed - module import error: %s",
            e,
            extra={"component": "health_entry", "error_type": "import"},
        )
        sys.exit(1)
    except OSError as e:
        logger.error(
            "Health check failed - I/O error: %s",
            e,
            extra={"component": "health_entry", "error_type": "io"},
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Health check failed - unexpected error: %s",
            e,
            exc_info=True,
            extra={"component": "health_entry", "error_type": "unexpected"},
        )
        sys.exit(1)


def main() -> None:
    """Default main entry point."""
    ensure_dotenv_loaded()
    if "--dry-run" in sys.argv:
        logger.info(
            "AI Main: Dry run mode - config loaded successfully, exiting gracefully"
        )
        return
    from ai_trading import runner

    try:
        runner.run_cycle()
    except KeyboardInterrupt:
        logger.info("Main execution interrupted by user")
        sys.exit(0)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(
            "Main execution failed - module import error: %s",
            e,
            extra={"component": "main_entry", "error_type": "import"},
        )
        sys.exit(1)
    except OSError as e:
        logger.error(
            "Main execution failed - I/O error: %s",
            e,
            extra={"component": "main_entry", "error_type": "io"},
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Main execution failed - unexpected error: %s",
            e,
            exc_info=True,
            extra={"component": "main_entry", "error_type": "unexpected"},
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # AI-AGENT-REF: ignore errors for dry-run
        if "--dry-run" in sys.argv:
            logger.warning("dry-run: ignoring startup exception: %s", e)
            sys.exit(0)
        raise
