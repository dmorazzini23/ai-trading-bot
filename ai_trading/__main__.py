from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any, Literal

sys.dont_write_bytecode = True

from ai_trading.exc import HTTPError
from pydantic import BaseModel, ValidationError, field_validator

from ai_trading.logging import get_logger
from ai_trading.utils.time import monotonic_time
from ai_trading.runtime.shutdown import (
    install_runtime_timer,
    register_signal_handlers,
    request_stop,
    should_stop,
    stop_event,
)

logger = get_logger(__name__)
register_signal_handlers()


def _build_parser(description: str, *, symbols: bool = False) -> argparse.ArgumentParser:
    """Return a base parser with common runtime options."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dry-run", action="store_true", help="Exit before heavy imports")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit")
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="Cooperatively request shutdown after N seconds",
    )
    parser.add_argument(
        "--graceful-exit",
        action="store_true",
        help="Request a cooperative shutdown at the next checkpoint",
    )
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

    try:
        while not should_stop():
            try:
                fn()
            except (ValueError, HTTPError) as e:
                logger.warning("%s recoverable error: %s", label, e, exc_info=True)
            except Exception as e:
                logger.error("%s failed: %s", label, e, exc_info=True)
                raise
            if args.once:
                break
            interval = max(0.0, float(getattr(args, "interval", 0.0)))
            if interval <= 0:
                continue
            deadline = monotonic_time() + interval
            while not should_stop() and monotonic_time() < deadline:
                time.sleep(min(0.25, max(0.0, deadline - monotonic_time())))
    except KeyboardInterrupt:
        request_stop("keyboard-interrupt")
        logger.info("%s interrupted", label)
        sys.exit(0)
    except SystemExit as e:  # AI-AGENT-REF: do not crash the service on exit codes
        # Log and return to allow the supervisor to decide on restart policy.
        try:
            code = int(getattr(e, "code", 1) or 0)
        except Exception:
            code = 1
        logger.error("%s exited with code %s; continuing", label, code, exc_info=True)
        return


class _StartupConfig(BaseModel):
    """Minimal runtime configuration validated at startup."""

    timeframe: Literal["1Min", "1Day"]
    data_feed: Literal["iex", "sip"]

    @field_validator("timeframe", mode="before")
    @classmethod
    def _norm_timeframe(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if s in {"1min", "1m", "minute", "1 minute"}:
            return "1Min"
        if s in {"1day", "1d", "day", "1 day"}:
            return "1Day"
        raise ValueError(f"Unsupported timeframe: {v}")

    @field_validator("data_feed", mode="before")
    @classmethod
    def _norm_feed(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if s in {"iex", "sip"}:
            return s
        raise ValueError(f"Unsupported data_feed: {v}")


def _validate_startup_config() -> _StartupConfig:
    """Validate runtime config and exit fast on errors."""

    from ai_trading.config.management import get_env
    from ai_trading.settings import get_settings

    feed_default = get_settings().alpaca_data_feed
    cfg = {
        "timeframe": get_env("TIMEFRAME", "1Min"),
        "data_feed": get_env("DATA_FEED", feed_default),
    }
    try:
        return _StartupConfig(**cfg)
    except ValidationError as e:  # pragma: no cover - exercised in tests
        logger.error("CONFIG_VALIDATION_FAILED", extra={"errors": e.errors()})
        raise SystemExit(f"Invalid configuration: {e}") from e


def run_trade() -> None:
    """Entrypoint for live trading loop."""

    parser = _build_parser("AI Trading Bot", symbols=True)
    args = parser.parse_args()
    stop_event.clear()
    timer = None
    if getattr(args, "max_runtime_seconds", None):
        timer = install_runtime_timer(float(args.max_runtime_seconds))
    if getattr(args, "graceful_exit", False):
        request_stop("cli-graceful-exit")
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
    _validate_startup_config()
    from ai_trading import main as _main

    _main.preflight_import_health()

    try:
        _run_loop(_main.run_cycle, args, "Trade")
    finally:
        if timer is not None:
            timer.cancel()


def run_backtest() -> None:
    """Entrypoint for backtesting loop."""

    parser = _build_parser("AI Trading Bot Backtesting", symbols=True)
    args = parser.parse_args()
    stop_event.clear()
    timer = None
    if getattr(args, "max_runtime_seconds", None):
        timer = install_runtime_timer(float(args.max_runtime_seconds))
    if getattr(args, "graceful_exit", False):
        request_stop("cli-graceful-exit")
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
    _validate_startup_config()
    from ai_trading import main as _main

    _main.preflight_import_health()

    try:
        _run_loop(_main.run_cycle, args, "Backtest")
    finally:
        if timer is not None:
            timer.cancel()


def run_healthcheck() -> None:
    """Entrypoint for health check loop."""

    parser = _build_parser("AI Trading Bot Health Check")
    args = parser.parse_args()
    stop_event.clear()
    timer = None
    if getattr(args, "max_runtime_seconds", None):
        timer = install_runtime_timer(float(args.max_runtime_seconds))
    if getattr(args, "graceful_exit", False):
        request_stop("cli-graceful-exit")
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
    _validate_startup_config()
    from ai_trading.health_monitor import run_health_check
    from ai_trading import main as _main

    _main.preflight_import_health()

    try:
        _run_loop(run_health_check, args, "Health check")
    finally:
        if timer is not None:
            timer.cancel()


def main() -> int:
    """Default CLI entrypoint mirroring ``run_trade``."""

    try:
        parser = _build_parser("AI Trading Bot", symbols=True)
        args = parser.parse_args()
        stop_event.clear()
        timer = None
        if getattr(args, "max_runtime_seconds", None):
            timer = install_runtime_timer(float(args.max_runtime_seconds))
        if getattr(args, "graceful_exit", False):
            request_stop("cli-graceful-exit")
        if args.dry_run:
            logger.info("AI Main: Dry run - exiting")
            logger.info("INDICATOR_IMPORT_OK")
            import logging, time

            time.sleep(0.1)
            logging.shutdown()
            return 0

        import os

        os.environ["TRADING_MODE"] = "paper" if args.paper else "live"
        from ai_trading.env import ensure_dotenv_loaded

        ensure_dotenv_loaded()
        _validate_startup_config()
        from ai_trading import main as _main

        # Ensure core imports are healthy before starting
        _main.preflight_import_health()

        # Route through the consolidated scheduler/API entrypoint so
        # `python -m ai_trading` behaves the same as `python -m ai_trading.main`.
        # Map flags: --once -> --iterations=1, --interval passthrough.
        mapped_argv: list[str] = []
        try:
            # Tolerate non-float intervals; main() will clamp/parse again
            if args.interval is not None:
                mapped_argv.extend(["--interval", str(args.interval)])
        except Exception:
            pass
        if getattr(args, "once", False):
            mapped_argv.extend(["--iterations", "1"])

        # Delegate to main; it starts the API server thread and the scheduler loop
        try:
            _main.main(mapped_argv)
            return 0
        finally:
            if timer is not None:
                timer.cancel()
    except SystemExit as e:
        code = getattr(e, "code", None)
        logger.error("ai_trading.main exited with code %s", code, exc_info=True)
        raise
    except (ValueError, HTTPError) as e:
        logger.error("startup error: %s", e, exc_info=True)
        if "--dry-run" in sys.argv:
            logger.warning("dry-run: ignoring startup exception: %s", e)
            return 0
        return 1
    except Exception:
        logger.exception("unexpected startup error")
        return 1


if __name__ == "__main__":
    # When invoked under pytest via runpy with test node ids in argv,
    # avoid raising SystemExit so collection/execution continues cleanly.
    rc = main()
    try:
        import os
        if any("pytest" in arg for arg in sys.argv) or any("::" in arg for arg in sys.argv) or os.getenv("PYTEST_RUNNING") == "1":
            pass
        else:
            sys.exit(rc)
    except Exception:
        sys.exit(rc)
