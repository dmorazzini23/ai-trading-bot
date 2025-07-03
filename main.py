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

import pandas as pd
from logger import get_logger, log_performance_metrics
from bot_engine import compute_current_positions, ctx as bot_ctx


def summarize_trades(path: str = os.getenv("TRADE_LOG_FILE", "trades.csv")) -> None:
    """Log summary of attempted vs skipped trades from ``path``."""
    log = get_logger(__name__)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - I/O errors
        log.warning("SUMMARY_READ_FAIL %s", exc)
        return
    attempted = len(df)
    skipped = (
        df[df.get("status") == "skipped"].groupby("reason").size().to_dict()
        if "status" in df.columns
        else {}
    )
    log.info("TRADE_RUN_SUMMARY", extra={"attempted": attempted, "skipped": skipped})

    exposure = abs(df.get("qty", 0) * df.get("price", 0)).sum() if not df.empty else 0
    equity_curve = df.get("equity", []).tolist() if "equity" in df.columns else df.get("price", []).cumsum().tolist()
    regime = df.get("regime").iloc[-1] if "regime" in df.columns and not df.empty else "unknown"
    log_performance_metrics(exposure_pct=exposure, equity_curve=equity_curve, regime=regime)


def screen_candidates_with_logging(candidates: list[str], tickers: list[str]) -> list[str]:
    """Return final candidate list with fallback and position filtering."""  # AI-AGENT-REF: candidate logging
    log = get_logger(__name__)
    log.info("Number of screened candidates: %s", len(candidates))
    if not candidates:
        log.warning(
            "No candidates found after filtering, using top 5 tickers fallback."
        )
        candidates = tickers[:5]

    positions = compute_current_positions(bot_ctx)
    filtered = [c for c in candidates if c not in positions]
    if not filtered:
        log.info("All candidates already held, skipping new buys.")
        return []

    return filtered


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
    summarize_trades()
