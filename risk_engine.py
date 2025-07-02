import logging
import os
import random
import warnings
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=".*invalid escape sequence.*",
    category=SyntaxWarning,
    module="pandas_ta.*",
)

from strategies import TradeSignal
from utils import get_phase_logger

logger = get_phase_logger(__name__, "RISK_CHECK")

random.seed(42)
np.random.seed(42)


MAX_DRAWDOWN = 0.05

# Simple global risk state used by utility helpers
HARD_STOP = False
MAX_TRADES = 10
CURRENT_TRADES = 0


class RiskEngine:
    """Cross-strategy risk manager."""

    def __init__(self) -> None:
        self.global_limit = 1.0
        self.asset_limits: Dict[str, float] = {}
        self.strategy_limits: Dict[str, float] = {}
        self.exposure: Dict[str, float] = {}
        self.hard_stop = False

    def _dynamic_cap(self, asset_class: str, volatility: float | None = None, cash_ratio: float | None = None) -> float:
        """Return exposure cap for ``asset_class`` adjusted by volatility and cash."""
        cap = self.asset_limits.get(asset_class, self.global_limit)
        if os.getenv("DYNAMIC_EXPOSURE_CAP", "0") != "1":
            return cap
        base = float(os.getenv("BASE_CAP", 1.0))
        adj = float(os.getenv("VOLATILITY_ADJUST", 0.2))
        vol = volatility if volatility is not None else 1.0
        dyn = base * (1 + vol * adj)
        if cash_ratio is not None:
            dyn *= max(0.2, min(1.0, cash_ratio))
        return min(cap, dyn)

    def can_trade(self, signal: TradeSignal, *, pending: float = 0.0, volatility: float | None = None, cash_ratio: float | None = None) -> bool:
        if self.hard_stop:
            logger.error("TRADING_HALTED_RISK_LIMIT")
            return False
        if not isinstance(signal, TradeSignal):
            logger.error("can_trade called with invalid signal type")
            return False

        asset_exp = self.exposure.get(signal.asset_class, 0.0) + max(pending, 0.0)
        asset_cap = self._dynamic_cap(signal.asset_class, volatility, cash_ratio)
        if asset_exp + signal.weight > asset_cap:
            logger.warning(
                "Exposure cap exceeded for %s: %.2f vs %.2f",
                signal.asset_class,
                asset_exp + signal.weight,
                asset_cap,
            )
            if os.getenv("FORCE_CONTINUE_ON_EXPOSURE", "false").lower() != "true":
                return False
            logger.warning("FORCE_CONTINUE_ON_EXPOSURE enabled; overriding cap")

        strat_cap = self.strategy_limits.get(signal.strategy, self.global_limit)
        if signal.weight > strat_cap:
            logger.warning(
                "Strategy %s weight %.2f exceeds cap %.2f",
                signal.strategy,
                signal.weight,
                strat_cap,
            )
            if os.getenv("FORCE_CONTINUE_ON_EXPOSURE", "false").lower() != "true":
                return False
            logger.warning("FORCE_CONTINUE_ON_EXPOSURE enabled; overriding cap")
        return True

    def register_fill(self, signal: TradeSignal) -> None:
        if not isinstance(signal, TradeSignal):
            logger.error("register_fill called with invalid signal type")
            return

        prev = self.exposure.get(signal.asset_class, 0.0)
        self.exposure[signal.asset_class] = prev + signal.weight
        logger.info(
            "EXPOSURE_UPDATED",
            extra={
                "asset": signal.asset_class,
                "prev": prev,
                "new": self.exposure[signal.asset_class],
            },
        )

    def check_max_drawdown(self, api) -> bool:
        try:
            account = api.get_account()
            pnl = float(account.equity) - float(account.last_equity)
            # Assume account returns equity strings convertible to float
            if pnl < -MAX_DRAWDOWN * float(account.last_equity):
                logger.error("HARD_STOP_MAX_DRAWDOWN", extra={"pnl": pnl})
                self.hard_stop = True
                return False
            return True
        except Exception as exc:  # TODO: narrow exception type
            logger.error("check_max_drawdown failed: %s", exc)
            return False

    def position_size(
        self, signal: TradeSignal, cash: float, price: float, api=None
    ) -> int:
        logger.debug(
            "Pricing inputs for %s | cash: %.2f, price: %.2f, signal: %s",
            getattr(signal, "symbol", "N/A"),
            cash,
            price,
            signal,
        )
        if self.hard_stop:
            logger.error("HARD_STOP_ACTIVE")
            return 0
        if not isinstance(signal, TradeSignal):
            logger.error("position_size called with invalid signal type")
            return 0

        if api is not None and not self.check_max_drawdown(api):
            return 0

        if cash <= 0 or price <= 0:
            logger.warning(
                "Invalid cash %.2f or price %.2f for position sizing", cash, price
            )
            return 0

        if not self.can_trade(signal):
            return 0

        weight = self._apply_weight_limits(signal)

        dollars = cash * min(weight, 1.0)
        if np.isnan(dollars) or np.isnan(price):
            logger.error("position_size received NaN inputs")
            return 0
        try:
            qty = int(round(dollars / price))
        except (ZeroDivisionError, OverflowError, TypeError, ValueError) as exc:
            logger.error("position_size division error: %s", exc)
            return 0
        return max(qty, 0)

    def _apply_weight_limits(self, signal: TradeSignal) -> float:
        """Return signal weight after applying asset and strategy caps."""
        asset_cap = self._dynamic_cap(signal.asset_class)
        asset_rem = max(asset_cap - self.exposure.get(signal.asset_class, 0.0), 0.0)
        strat_cap = self.strategy_limits.get(signal.strategy, self.global_limit)
        weight = signal.weight
        if weight > asset_rem:
            logger.info("ADJUST_WEIGHT_ASSET", extra={"orig": weight, "new": asset_rem})
            weight = asset_rem
        if weight > strat_cap:
            logger.info(
                "ADJUST_WEIGHT_STRATEGY", extra={"orig": weight, "new": strat_cap}
            )
            weight = strat_cap
        return weight

    def compute_volatility(self, returns: np.ndarray) -> dict:
        if not isinstance(returns, np.ndarray) or returns.size == 0:
            logger.warning("Empty or invalid returns series—skipping risk computation")
            return {"volatility": 0.0}

        try:
            vol = float(np.std(returns))
        except (ValueError, TypeError) as exc:
            logger.error("Failed computing volatility: %s", exc)
            vol = 0.0
        return {"volatility": vol}


def calculate_position_size(*args, **kwargs) -> int:
    """Convenience wrapper supporting simple and advanced usage."""
    engine = RiskEngine()
    if len(args) == 2 and not kwargs:
        cash, price = args
        dummy = TradeSignal(
            symbol="DUMMY", side="buy", confidence=1.0, strategy="default"
        )
        return engine.position_size(dummy, cash, price)
    if len(args) >= 3:
        signal, cash, price = args[:3]
        api = args[3] if len(args) > 3 else kwargs.get("api")
        return engine.position_size(signal, cash, price, api)
    raise TypeError("Invalid arguments for calculate_position_size")


def check_max_drawdown(state: Dict[str, float]) -> bool:
    """Return True if current drawdown exceeds the maximum allowed."""
    return state.get("current_drawdown", 0) > state.get("max_drawdown", 0)


def can_trade() -> bool:
    """Return False when trading should be halted."""
    return not HARD_STOP and CURRENT_TRADES < MAX_TRADES


def register_trade(size: int) -> dict | None:
    """Register a trade and increment the count if allowed."""
    global CURRENT_TRADES
    if not can_trade() or size <= 0:
        return None
    CURRENT_TRADES += 1
    return {"size": size}


import pandas_ta as ta


def apply_trailing_atr_stop(df: pd.DataFrame, entry_price: float) -> None:
    """Exit position if price falls below an ATR-based trailing stop."""
    try:
        if entry_price <= 0:
            logger.warning("apply_trailing_atr_stop invalid entry price: %.2f", entry_price)
            return
        atr = df.ta.atr()
        trailing_stop = entry_price - 2 * atr
        last_valid_close = df["Close"].dropna()
        if not last_valid_close.empty:
            price = last_valid_close.iloc[-1]
        else:
            logger.critical("All NaNs in close column for ATR stop")
            price = 0.0
        logger.debug("Latest 5 rows for ATR stop:\n%s", df.tail(5))
        logger.debug("Computed price for ATR stop: %s", price)
        if price <= 0 or pd.isna(price):
            logger.critical("Invalid price computed for ATR stop: %s", price)
            return
        if price < trailing_stop.iloc[-1]:
            print(f"Hit stop: price {price} vs {trailing_stop.iloc[-1]}")
            sell()  # noqa: F821 - example placeholder
            schedule_reentry_check("SYMBOL", lookahead_days=2)
    except Exception as e:  # pragma: no cover - defensive
        print(f"ATR stop error: {e}")


def schedule_reentry_check(symbol: str, lookahead_days: int) -> None:
    """Log a re-entry check after a stop out."""
    print(f"Scheduling reentry for {symbol} in {lookahead_days} days.")
