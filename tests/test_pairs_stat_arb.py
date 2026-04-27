from __future__ import annotations

import numpy as np

from ai_trading.strategies.pairs_stat_arb import PairsStatArbStrategy


def test_pairs_stat_arb_uses_sell_short_for_short_leg() -> None:
    base = np.linspace(-0.05, 0.05, 60)
    left = base.copy()
    right = base.copy()
    left[-1] += 0.03
    other = np.sin(np.linspace(0, 12, 60)) * 0.01
    fourth = np.cos(np.linspace(0, 12, 60)) * 0.01
    strategy = PairsStatArbStrategy(min_universe=4)

    signals = strategy.generate_signals(
        {"returns": {"AAA": left, "BBB": right, "CCC": other, "DDD": fourth}}
    )

    assert [(signal.symbol, signal.side) for signal in signals] == [
        ("AAA", "sell_short"),
        ("BBB", "buy"),
    ]
