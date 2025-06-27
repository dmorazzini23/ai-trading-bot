"""Alias to run module for backward compatibility."""
import asyncio
import importlib
import os

from alpaca.trading.client import TradingClient

import alpaca_api
import run as _run

_run = importlib.reload(_run)

create_flask_app = _run.create_flask_app
run_flask_app = _run.run_flask_app
run_bot = _run.run_bot
validate_environment = _run.validate_environment
main = _run.main


def start_trade_updates_loop() -> None:
    """Convenience wrapper to run trade update streaming."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    paper = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").startswith("https://paper")
    client = TradingClient(api_key, secret, paper=paper)
    asyncio.run(
        alpaca_api.start_trade_updates_stream(api_key, secret, client, paper=paper)
    )

if __name__ == "__main__":
    _run.main()
