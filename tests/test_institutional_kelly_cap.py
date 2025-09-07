from ai_trading.config.management import TradingConfig
from ai_trading.risk.kelly import KellyParams, institutional_kelly


def test_institutional_kelly_respects_global_cap():
    """Ensure institutional_kelly output never exceeds config cap."""
    cfg = TradingConfig.from_env()
    params = KellyParams(win_prob=0.6, win_loss_ratio=2.0, cap=1.0)
    assert institutional_kelly(params) == cfg.kelly_fraction_max
