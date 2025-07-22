"""Entry point for running the trading bot with graceful shutdown.

Simple runner that restarts the trading bot on failures.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import NoReturn

import requests
from trade_execution import recent_buys
import numpy as np
from indicators import (
    vwap,
    donchian_channel,
    obv,
    stochastic_rsi,
    hurst_exponent,
)
try:
    from ai_trading.capital_scaling import (
        fractional_kelly,
        volatility_parity,
        cvar_scaling,
        kelly_fraction,
    )
except ImportError:  # AI-AGENT-REF: gracefully handle optional scaling helpers
    fractional_kelly = lambda *a, **k: None  # type: ignore
    volatility_parity = lambda *a, **k: None  # type: ignore
    cvar_scaling = lambda *a, **k: None  # type: ignore
    kelly_fraction = lambda *a, **k: None  # type: ignore
from utils import get_phase_logger, log_cpu_usage

# AI-AGENT-REF: allow bot override with fallback to bot_engine
try:
    import bot
except ImportError:
    import bot_engine as bot
else:
    if bot is None or not hasattr(bot, "main"):
        import bot_engine as bot

main = bot.main

logger = get_phase_logger(__name__, "RUNNER")

_shutdown = False


def _handle_signal(signum: int, _unused_frame) -> None:
    """Handle termination signals by setting the shutdown flag."""
    global _shutdown
    logger.info("Received signal %s, shutting down", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _run_forever() -> NoReturn:
    """Run ``bot.main`` in a loop until a shutdown signal is received."""

    global _shutdown

    while not _shutdown:
        try:
            main()
        except SystemExit as exc:  # graceful exit
            if exc.code == 0:
                _shutdown = True
                break
            logger.error("Bot exited with code %s", exc.code)
        except requests.exceptions.RequestException as exc:
            logger.exception("API request failed", exc_info=exc)
            raise
        except (RuntimeError, ValueError) as exc:
            logger.exception("Unexpected error", exc_info=exc)
            raise

        if not _shutdown:
            log_cpu_usage(logger)
            if any(time.time() - ts < 2 for ts in recent_buys.values()):
                logger.info("Post-buy sync wait")
                time.sleep(2)
            # AI-AGENT-REF: slow down runner loop to once per minute
            time.sleep(60)




def start_runner(*, once: bool = False) -> None:
    logger.info("Runner starting")
    if once:
        main()
    else:
        _run_forever()


# AI-AGENT-REF: experimental trading loop utilities
def vwap_signal(prices: np.ndarray, volumes: np.ndarray, last_close: float) -> str:
    return "buy" if last_close < vwap(prices, volumes) else "sell"


def donchian_breakout_signal(highs: np.ndarray, lows: np.ndarray, last_close: float) -> str:
    channel = donchian_channel(highs, lows)
    if last_close > channel["upper"]:
        return "buy"
    if last_close < channel["lower"]:
        return "sell"
    return "hold"


def obv_divergence_signal(closes: np.ndarray, volumes: np.ndarray) -> str:
    obv_vals = obv(closes, volumes)
    return "buy" if obv_vals[-1] > np.mean(obv_vals) else "sell"


def stochastic_rsi_signal(prices: np.ndarray) -> str:
    rsi_vals = stochastic_rsi(prices)
    if rsi_vals[-1] < 20:
        return "buy"
    if rsi_vals[-1] > 80:
        return "sell"
    return "hold"


def run_trading_loop(historical_data: list[dict], portfolio, regime: str = "neutral"):
    prices = np.array([c["close"] for c in historical_data])
    highs = np.array([c["high"] for c in historical_data])
    lows = np.array([c["low"] for c in historical_data])
    closes = np.array([c["close"] for c in historical_data])
    volumes = np.array([c["volume"] for c in historical_data])

    signals = [
        vwap_signal(prices, volumes, closes[-1]),
        donchian_breakout_signal(highs, lows, closes[-1]),
        obv_divergence_signal(closes, volumes),
        stochastic_rsi_signal(closes),
    ]

    majority_signal = max(set(signals), key=signals.count)

    win_rate, win_loss_ratio = 0.55, 1.8
    base_kelly = kelly_fraction(win_rate, win_loss_ratio)
    adjusted_kelly = fractional_kelly(base_kelly, regime)
    returns = np.diff(closes) / closes[:-1]
    vol_scaled = volatility_parity(np.ones_like(returns), np.std(returns))
    cvar_factor = cvar_scaling(returns)

    position_size = adjusted_kelly * np.mean(vol_scaled) * cvar_factor * portfolio.cash

    if majority_signal == "buy":
        portfolio.buy(position_size)
    elif majority_signal == "sell":
        portfolio.sell(position_size)
    else:
        portfolio.hold()

    return portfolio


if __name__ == "__main__":
    import sys
    import os
    once = "--once" in sys.argv or os.getenv("RUN_ONCE", "1") == "1"
    start_runner(once=once)
