import logging
import random
from typing import Dict

import numpy as np

from strategies import TradeSignal

random.seed(42)
np.random.seed(42)

logger = logging.getLogger(__name__)

MAX_DRAWDOWN = 0.05


class RiskEngine:
    """Cross-strategy risk manager."""

    def __init__(self) -> None:
        self.global_limit = 1.0
        self.asset_limits: Dict[str, float] = {}
        self.strategy_limits: Dict[str, float] = {}
        self.exposure: Dict[str, float] = {}
        self.hard_stop = False

    def can_trade(self, signal: TradeSignal) -> bool:
        if self.hard_stop:
            logger.error("TRADING_HALTED_RISK_LIMIT")
            return False
        if not isinstance(signal, TradeSignal):
            logger.error("can_trade called with invalid signal type")
            return False

        asset_exp = self.exposure.get(signal.asset_class, 0.0)
        asset_cap = self.asset_limits.get(signal.asset_class, self.global_limit)
        if asset_exp + signal.weight > asset_cap:
            logger.info(
                "Trade blocked: %s exposure %.2f exceeds cap %.2f",
                signal.asset_class,
                asset_exp + signal.weight,
                asset_cap,
            )
            return False

        strat_cap = self.strategy_limits.get(signal.strategy, self.global_limit)
        allowed = signal.weight <= strat_cap
        if not allowed:
            logger.info(
                "Trade blocked: strategy %s weight %.2f exceeds cap %.2f",
                signal.strategy,
                signal.weight,
                strat_cap,
            )
        return allowed

    def register_fill(self, signal: TradeSignal) -> None:
        if not isinstance(signal, TradeSignal):
            logger.error("register_fill called with invalid signal type")
            return

        prev = self.exposure.get(signal.asset_class, 0.0)
        self.exposure[signal.asset_class] = prev + signal.weight
        logger.debug(
            "Registered fill for %s: exposure %.2f -> %.2f",
            signal.asset_class,
            prev,
            self.exposure[signal.asset_class],
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
        except Exception as exc:
            logger.error("check_max_drawdown failed: %s", exc)
            return False

    def position_size(
        self, signal: TradeSignal, cash: float, price: float, api=None
    ) -> int:
        if self.hard_stop:
            logger.error("HARD_STOP_ACTIVE")
            return 0
        if not isinstance(signal, TradeSignal):
            logger.error("position_size called with invalid signal type")
            return 0

        if api is not None and not self.check_max_drawdown(api):
            return 0

        if cash <= 0 or price <= 0:
            logger.error(
                "Invalid cash %.2f or price %.2f for position sizing", cash, price
            )
            return 0

        if not self.can_trade(signal):
            return 0

        dollars = cash * min(signal.weight, 1.0)
        try:
            qty = int(dollars / price)
        except Exception as exc:
            logger.error("position_size division error: %s", exc)
            return 0
        return max(qty, 0)

    def compute_volatility(self, returns: np.ndarray) -> dict:
        if not isinstance(returns, np.ndarray) or returns.size == 0:
            logger.warning("Empty or invalid returns series—skipping risk computation")
            return {"volatility": 0.0}

        try:
            vol = float(np.std(returns))
        except Exception as exc:
            logger.error("Failed computing volatility: %s", exc)
            vol = 0.0
        return {"volatility": vol}
