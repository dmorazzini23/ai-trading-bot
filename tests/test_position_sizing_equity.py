import logging
import ai_trading.position_sizing as ps
import ai_trading.core.runtime as rt


def test_get_max_position_size_uses_cached_equity(monkeypatch, caplog):
    # Ensure cache is clear
    ps._CACHE.value, ps._CACHE.ts = (None, None)

    # Stub equity fetcher to return a positive value
    monkeypatch.setattr(ps, "_get_equity_from_alpaca", lambda cfg: 1000.0)
    monkeypatch.setattr(rt, "_get_equity_from_alpaca", lambda cfg: 1000.0)

    class Cfg:
        capital_cap = 0.04
        dollar_risk_limit = 0.05
        max_position_size = None
        max_position_mode = "STATIC"

    cfg = Cfg()

    # First call populates equity on the config
    rt.build_runtime(cfg)
    assert getattr(cfg, "equity", None) == 1000.0

    # Force recompute and ensure no EQUITY_MISSING is logged
    cfg.max_position_size = 0.0
    caplog.set_level(logging.WARNING)
    caplog.clear()
    ps.get_max_position_size(cfg, cfg, force_refresh=True)
    assert not any(
        r.msg == "EQUITY_MISSING" for r in caplog.records if r.name == "ai_trading.position_sizing"
    )
