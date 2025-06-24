"""Module entrypoint for ``python -m ai-trading-bot``."""

from __future__ import annotations

import logging

try:
    from alpaca_trade_api.rest import REST
except Exception:  # pragma: no cover - optional dependency
    REST = object  # type: ignore

from run import main as entrypoint


def main() -> None:  # pragma: no cover - thin wrapper for entrypoint
    """Invoke :func:`run.main` when module is executed as a package."""
    entrypoint()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

