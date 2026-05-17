import math

from ai_trading.config.management import TradingConfig
from ai_trading.risk.kelly import KellyParams, institutional_kelly


def test_institutional_kelly_respects_global_cap():
    """Ensure institutional_kelly output never exceeds config cap."""
    cfg = TradingConfig.from_env()
    params = KellyParams(win_prob=0.6, win_loss_ratio=2.0, cap=1.0)
    assert institutional_kelly(params) == cfg.kelly_fraction_max


def test_institutional_kelly_invalid_probability_fails_closed():
    """Invalid probabilities must not produce a positive Kelly allocation."""

    for win_prob in (-0.01, 1.01, math.nan, math.inf):
        params = KellyParams(win_prob=win_prob, win_loss_ratio=2.0, cap=1.0)

        assert institutional_kelly(params) == 0.0


def test_institutional_kelly_invalid_ratio_or_cap_fails_closed():
    """Invalid ratio/cap values must not produce a positive allocation."""

    assert institutional_kelly(KellyParams(win_prob=0.6, win_loss_ratio=0.0, cap=1.0)) == 0.0
    assert institutional_kelly(KellyParams(win_prob=0.6, win_loss_ratio=2.0, cap=0.0)) == 0.0
