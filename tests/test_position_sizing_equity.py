import logging
from types import SimpleNamespace

import ai_trading.position_sizing as ps
import ai_trading.core.runtime as rt
from ai_trading.logging import logger_once


def test_get_max_position_size_uses_cached_equity(monkeypatch, caplog):
    # Ensure cache is clear
    ps._CACHE.value, ps._CACHE.ts, ps._CACHE.equity = (None, None, None)
    logger_once._emitted_keys.clear()

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
    cfg.max_position_size = None
    caplog.set_level(logging.WARNING)
    caplog.clear()
    ps.get_max_position_size(cfg, force_refresh=True)
    assert not any(
        r.msg == "EQUITY_MISSING" for r in caplog.records if r.name == "ai_trading.position_sizing"
    )


def test_get_max_position_size_auto_multiplies_equity(monkeypatch):
    ps._CACHE.value, ps._CACHE.ts, ps._CACHE.equity = (None, None, None)
    monkeypatch.setattr(ps, "_get_equity_from_alpaca", lambda cfg, force_refresh=False: 1000.0)

    cfg = SimpleNamespace(capital_cap=0.05)
    val = ps.get_max_position_size(cfg, auto=True, force_refresh=True)
    assert val == 50.0


def test_resolve_max_position_size_uses_real_equity_and_caches(monkeypatch, caplog):
    ps._CACHE.value, ps._CACHE.ts, ps._CACHE.equity = (None, None, None)
    logger_once._emitted_keys.clear()
    calls = {"n": 0}

    def fake_fetch(cfg):
        calls["n"] += 1
        return 50000.0

    monkeypatch.setattr(ps, "_get_equity_from_alpaca", fake_fetch)
    cfg = SimpleNamespace(alpaca_api_key="k", alpaca_secret_key_plain="s", alpaca_base_url="https://paper-api.alpaca.markets")
    tcfg = SimpleNamespace(capital_cap=0.02, max_position_mode="STATIC")

    with caplog.at_level(logging.WARNING):
        size1, _ = ps.resolve_max_position_size(cfg, tcfg, force_refresh=True)
        size2, meta2 = ps.resolve_max_position_size(cfg, tcfg)

    assert size1 == size2 == 1000.0
    assert meta2["source"] == "cache"
    assert calls["n"] == 1
    assert not any(r.msg == "EQUITY_MISSING" for r in caplog.records)


def test_failed_equity_fetch_warns_once_and_caches(monkeypatch, caplog):
    ps._CACHE.value, ps._CACHE.ts, ps._CACHE.equity = (None, None, None)
    logger_once._emitted_keys.clear()
    calls = {"n": 0}

    def fake_fetch(cfg):
        calls["n"] += 1
        return 0.0

    monkeypatch.setattr(ps, "_get_equity_from_alpaca", fake_fetch)
    cfg = SimpleNamespace(alpaca_api_key="k", alpaca_secret_key_plain="s", alpaca_base_url="https://paper-api.alpaca.markets")
    tcfg = SimpleNamespace(capital_cap=0.02, max_position_mode="STATIC")

    with caplog.at_level(logging.WARNING):
        size1, _ = ps.resolve_max_position_size(cfg, tcfg, force_refresh=True)
        size2, meta2 = ps.resolve_max_position_size(cfg, tcfg)

    assert size1 == size2
    assert meta2["source"] == "cache"
    assert calls["n"] == 1
    assert [r.msg for r in caplog.records].count("EQUITY_MISSING") == 1


def test_equity_recovered_emits_warning_once(monkeypatch, caplog):
    ps._CACHE.value, ps._CACHE.ts, ps._CACHE.equity = (None, None, None)
    logger_once._emitted_keys.clear()

    monkeypatch.setattr(ps, "_get_equity_from_alpaca", lambda cfg: 0.0)

    cfg = SimpleNamespace(alpaca_api_key="k", alpaca_secret_key_plain="s", alpaca_base_url="https://paper-api.alpaca.markets")
    tcfg = SimpleNamespace(capital_cap=0.02, max_position_mode="STATIC")

    with caplog.at_level(logging.INFO):
        ps.resolve_max_position_size(cfg, tcfg, force_refresh=True)
        ps._CACHE.equity = 1000.0  # simulate later successful fetch
        ps.resolve_max_position_size(cfg, tcfg)
        ps.resolve_max_position_size(cfg, tcfg)

    equity_warns = [r.msg for r in caplog.records if r.msg == "EQUITY_MISSING"]
    sizing_msgs = [r.msg for r in caplog.records if r.name == "ai_trading.position_sizing"]
    assert equity_warns.count("EQUITY_MISSING") == 1
    assert sizing_msgs.count("CONFIG_AUTOFIX") == 1
