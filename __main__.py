"""Module entrypoint for ``python -m ai-trading-bot``."""

from __future__ import annotations

import logging
import warnings
import os
import time

# AI-AGENT-REF: suppress noisy external library warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

try:
    from alpaca_trade_api.rest import REST
except Exception:  # pragma: no cover - optional dependency
    REST = object  # type: ignore

from run import main as entrypoint
import config
from logger import get_logger

# AI-AGENT-REF: throttle HEALTH_ROWS and SKIP_COOLDOWN logs
_LAST_HEALTH_LOG_TIME = 0.0
_skip_cooldown_symbols: set[str] = set()


def check_health_rows(rows: int) -> None:
    """Log HEALTH_ROWS at most every 2 seconds."""
    global _LAST_HEALTH_LOG_TIME
    now = time.time()
    if now - _LAST_HEALTH_LOG_TIME > 2:
        get_logger(__name__).info("HEALTH_ROWS", extra={"rows": rows})
        _LAST_HEALTH_LOG_TIME = now


def skip_cooldown(symbols: list[str]) -> None:
    """Batch SKIP_COOLDOWN symbols until flushed."""
    global _skip_cooldown_symbols
    _skip_cooldown_symbols.update(symbols)


def flush_skip_cooldown_log() -> None:
    global _skip_cooldown_symbols
    if _skip_cooldown_symbols:
        get_logger(__name__).info(
            "SKIP_COOLDOWN_BATCHED | %s",
            ", ".join(sorted(_skip_cooldown_symbols)),
        )
        _skip_cooldown_symbols.clear()


def main() -> None:  # pragma: no cover - thin wrapper for entrypoint
    """Invoke :func:`run.main` when module is executed as a package."""
    os.environ.setdefault("BACKTEST_SERIAL", "1")  # AI-AGENT-REF: ensure serial backtests
    if config.FORCE_TRADES:
        get_logger(__name__).warning(
            "\ud83d\ude80 FORCE_TRADES is ENABLED. This run will ignore normal health halts!"
        )
    entrypoint()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

