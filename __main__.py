"""Minimal trading entrypoint for ``python -m ai-trading-bot``."""

from __future__ import annotations

import logging

try:
    from alpaca_trade_api.rest import REST
except Exception:  # pragma: no cover - optional dependency
    REST = object  # type: ignore

from utils import pre_trade_health_check


def run_trades(symbols: list[str], api: REST) -> None:
    """Placeholder trade execution logic."""
    for sym in symbols:
        logging.info("Executing trade for %s", sym)


def main() -> None:
    """Fetch symbols, run health check, and execute trades."""
    symbols = ["AAPL", "MSFT"]
    api = REST()  # type: ignore[arg-type]
    health = pre_trade_health_check(symbols, api)
    symbols_to_trade = [s for s, ok in health.items() if ok]
    if not symbols_to_trade:
        logging.warning("No symbols passed health check; skipping this cycle")
    else:
        run_trades(symbols_to_trade, api)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    logging.basicConfig(level=logging.INFO)
    main()

