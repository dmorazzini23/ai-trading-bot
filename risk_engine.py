from typing import Dict
import random
import numpy as np
from strategies import TradeSignal

random.seed(42)
np.random.seed(42)

MAX_DRAWDOWN = 0.05

class RiskEngine:
    """Cross-strategy risk manager."""

    def __init__(self) -> None:
        self.global_limit = 1.0
        self.asset_limits: Dict[str, float] = {}
        self.strategy_limits: Dict[str, float] = {}
        self.exposure: Dict[str, float] = {}

    def can_trade(self, signal: TradeSignal) -> bool:
        asset_exp = self.exposure.get(signal.asset_class, 0.0)
        asset_cap = self.asset_limits.get(signal.asset_class, self.global_limit)
        if asset_exp + signal.weight > asset_cap:
            return False
        strat_cap = self.strategy_limits.get(signal.strategy, self.global_limit)
        return signal.weight <= strat_cap

    def register_fill(self, signal: TradeSignal) -> None:
        self.exposure[signal.asset_class] = self.exposure.get(signal.asset_class, 0.0) + signal.weight

    def check_max_drawdown(self, api) -> bool:
        account = api.get_account()
        pnl = float(account.equity) - float(account.last_equity)
        if pnl < -MAX_DRAWDOWN * float(account.last_equity):
            return False
        return True

    def position_size(self, signal: TradeSignal, cash: float, price: float, api=None) -> int:
        if api is not None and not self.check_max_drawdown(api):
            return 0
        if not self.can_trade(signal) or price <= 0:
            return 0
        dollars = cash * min(signal.weight, 1.0)
        qty = int(dollars / price)
        return max(qty, 0)
