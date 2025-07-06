"""Module entrypoint for ``python -m ai-trading-bot``."""

from __future__ import annotations

import logging
import warnings
import os

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

