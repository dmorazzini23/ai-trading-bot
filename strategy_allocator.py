import sys, importlib
import logging
import math
import os
import time  # AI-AGENT-REF: needed for monotonic timestamps

from typing import Dict, List

# AI-AGENT-REF: reload real module if tests insert stub
if (
    "strategy_allocator" in sys.modules
    and not hasattr(sys.modules["strategy_allocator"], "__file__")
):
    importlib.reload(importlib.import_module(__name__))

from trade_execution import recent_buys

from strategies import TradeSignal
from utils import get_phase_logger

logger = get_phase_logger(__name__, "REBALANCE")
logger.debug("=== STRATEGY_ALLOCATOR LOADED === %s", __file__)


class StrategyAllocator:
    """Dynamic allocation of strategy weights."""

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {}
        self.last_direction: Dict[str, str] = {}
        self.last_confidence: Dict[str, float] = {}
        self.hold_protect: Dict[str, int] = {}
        self.hold_cycles = int(os.getenv("HOLD_PROTECT_CYCLES", 5))
        self.delta_threshold = float(os.getenv("DELTA_THRESHOLD", 0.3))
        # AI-AGENT-REF: track cooldown log state for throttling
        self._last_cooldown_log: Dict[str, float] = {}
        self._last_signal_hash: int | None = None

    def update_reward(self, strategy: str, reward: float) -> None:
        try:
            if not strategy:
                logger.warning("update_reward called with empty strategy name")
                return
            if not isinstance(reward, (int, float)) or math.isnan(reward):
                logger.warning(
                    "update_reward received invalid reward %r for %s",
                    reward,
                    strategy,
                )
                return

            w = self.weights.get(strategy, 1.0)
            self.weights[strategy] = max(0.1, min(2.0, w + reward))
            logger.debug(
                "Updated weight for %s from %.3f to %.3f",
                strategy,
                w,
                self.weights[strategy],
            )
        except Exception as exc:  # pragma: no cover - unexpected error
            logger.exception("update_reward failed: %s", exc)

    def allocate(self, signals: Dict[str, List[TradeSignal]], *, volatility: float | None = None) -> List[TradeSignal]:
        try:
            if not isinstance(signals, dict) or not signals:
                logger.warning("allocate called with empty or invalid signals input")
                return []

            vol = volatility if volatility is not None else float(os.getenv("RECENT_VOLATILITY", 1))

            current_hash = hash(
                tuple(
                    sorted(
                        (sig.symbol, round(getattr(sig, "confidence", 0), 4))
                        for sig_list in signals.values()
                        for sig in sig_list
                    )
                )
            )

            for sym in list(self.hold_protect):
                # AI-AGENT-REF: ensure cooldown decrements only once per cycle
                if vol < float(os.getenv("VOL_THRESHOLD", 1.2)):
                    self.hold_protect[sym] = max(0, self.hold_protect[sym] - 2)
                else:
                    self.hold_protect[sym] = max(0, self.hold_protect[sym] - 1)
                if self.hold_protect[sym] <= 0:
                    last_ts = self._last_cooldown_log.get(sym, 0.0)
                    if (
                        current_hash != self._last_signal_hash
                        or time.monotonic() - last_ts >= 15
                    ):
                        logger.info("COOLDOWN_EXPIRED", extra={"symbol": sym})
                        self._last_cooldown_log[sym] = time.monotonic()
                    self.hold_protect.pop(sym, None)

            # AI-AGENT-REF: map of symbols with buy signals to avoid premature sells
            positive_symbols = {
                s.symbol
                for lst in signals.values()
                for s in lst
                if getattr(s, "side", "") == "buy"
            }
            results: List[TradeSignal] = []
            for strat, sigs in signals.items():
                candidates_dict = {s.symbol: getattr(s, "confidence", 0) for s in sigs}
                logger.info(
                    "Strategy %s evaluating candidates: %s",
                    strat,
                    candidates_dict,
                )
                if not isinstance(sigs, list) or not sigs:
                    logger.warning(
                        "No signals for strategy %s. Candidates: %s",
                        strat,
                        candidates_dict,
                    )
                    continue

                weight = self.weights.get(strat, 1.0)
                if (
                    not isinstance(weight, (int, float))
                    or math.isnan(weight)
                    or weight <= 0
                ):
                    logger.warning(
                        "Invalid weight %r for strategy %s; defaulting to 1.0",
                        weight,
                        strat,
                    )
                    weight = 1.0

                before_count = len(results)
                for s in sigs:
                    if not isinstance(s, TradeSignal):
                        logger.warning("Invalid TradeSignal %r in %s", s, strat)
                        continue
                    if not isinstance(s.weight, (int, float)) or math.isnan(s.weight):
                        logger.warning(
                            "Signal weight invalid for %s; using default 1.0", s.symbol
                        )
                        s.weight = 1.0
                    if (
                        s.side == "sell"
                        and s.symbol in recent_buys
                        and time.time() - recent_buys[s.symbol] < 120
                    ):
                        # AI-AGENT-REF: allow sells despite recent buy; log only
                        logger.info(
                            "RECENT_BUY_SELL", extra={"symbol": s.symbol}
                        )
                    if s.side == "sell" and s.symbol in positive_symbols:
                        logger.info(
                            "HOLDING %s: indicator signals still positive", s.symbol
                        )
                        continue
                    last_dir = self.last_direction.get(s.symbol)
                    last_conf = self.last_confidence.get(s.symbol, 0.0)
                    if last_dir and last_dir != s.side:
                        if s.side == "sell" and self.hold_protect.get(s.symbol, 0) > 0:
                            # AI-AGENT-REF: remove hold protection block; log only
                            logger.info("HOLD_PROTECT_ACTIVE", extra={"symbol": s.symbol})
                        if s.confidence < last_conf + self.delta_threshold:
                            logger.info("SKIP_DELTA_THRESHOLD", extra={"symbol": s.symbol})
                            continue
                    s.weight *= weight
                    results.append(s)
                    if s.side == "sell":
                        logger.info(
                            "EXITING %s: indicator signals turned negative",
                            s.symbol,
                        )
                    self.last_direction[s.symbol] = s.side
                    self.last_confidence[s.symbol] = s.confidence
                    if s.side == "buy":
                        self.hold_protect[s.symbol] = self.hold_cycles

                logger.debug(
                    "Allocated %d signals for %s with weight %.3f",
                    len(results) - before_count,
                    strat,
                    weight,
                )
                final_cands = {s.symbol: s.weight for s in results if s.strategy == strat}
                logger.info(
                    "Final candidate signals for %s: %s",
                    strat,
                    final_cands,
                )
                if not final_cands:
                    logger.warning("No final candidates for %s", strat)

            self._last_signal_hash = current_hash
            return results
        except Exception as exc:  # pragma: no cover - unexpected error
            logger.exception("allocate failed: %s", exc)
            return []


__all__ = ["StrategyAllocator"]
