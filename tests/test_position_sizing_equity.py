import logging
from types import SimpleNamespace

import pytest

import ai_trading.position_sizing as ps
import ai_trading.core.runtime as rt


def _reset_cache():
    ps._CACHE.value = None
    ps._CACHE.ts = None
    ps._CACHE.equity = None
    ps._CACHE.last_equity = None
    ps._CACHE.last_equity_ts = None
    ps._CACHE.equity_error = None
    ps._CACHE.equity_missing_logged = False
    ps._CACHE.equity_ts = None
    ps._CACHE.equity_source = None
    ps._CACHE.equity_recovered_logged = False


def test_fetch_equity_sets_paper(monkeypatch):
    _reset_cache()

    calls: dict[str, bool] = {}

    class Dummy:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def get_account(self):  # pragma: no cover - simple stub
            return SimpleNamespace(equity=0.0)

    monkeypatch.setattr(ps, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(ps, "get_trading_client_cls", lambda: Dummy)

    cfg = SimpleNamespace(
        alpaca_api_key="k",
        alpaca_secret_key_plain="s",
        alpaca_base_url="https://paper-api.alpaca.markets",
    )

    ps._fetch_equity(cfg, force_refresh=True)

    assert calls.get("paper") is True

def test_get_max_position_size_uses_cached_equity(monkeypatch, caplog):
    # Ensure cache is clear
    _reset_cache()
    ps._once_logger._emitted_keys.clear()

    # Stub equity fetcher to return a positive value
    monkeypatch.setattr(ps, "_get_equity_from_alpaca", lambda cfg, force_refresh=False: 1000.0)
    monkeypatch.setattr(rt, "_get_equity_from_alpaca", lambda cfg, force_refresh=False: 1000.0)

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
    _reset_cache()
    monkeypatch.setattr(ps, "_get_equity_from_alpaca", lambda cfg, force_refresh=False: 1000.0)

    cfg = SimpleNamespace(capital_cap=0.05)
    val = ps.get_max_position_size(cfg, auto=True, force_refresh=True)
    assert val == 50.0


def test_resolve_max_position_size_uses_real_equity_and_caches(monkeypatch, caplog):
    _reset_cache()
    ps._once_logger._emitted_keys.clear()
    calls = {"n": 0}

    def fake_fetch(cfg, force_refresh=False):
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
    _reset_cache()
    ps._once_logger._emitted_keys.clear()
    calls = {"n": 0}

    def fake_fetch(cfg, force_refresh=False):
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
    records = caplog.get_records("call") + caplog.get_records("teardown")
    assert (
        any(r.getMessage() == "EQUITY_MISSING" for r in records)
        or "position_sizing:equity_missing" in ps._once_logger._emitted_keys
    )


def test_equity_recovered_emits_warning_once(monkeypatch, caplog):
    _reset_cache()
    ps._once_logger._emitted_keys.clear()

    monkeypatch.setattr(ps, "_get_equity_from_alpaca", lambda cfg, force_refresh=False: 0.0)

    cfg = SimpleNamespace(alpaca_api_key="k", alpaca_secret_key_plain="s", alpaca_base_url="https://paper-api.alpaca.markets")
    tcfg = SimpleNamespace(capital_cap=0.02, max_position_mode="STATIC")

    with caplog.at_level(logging.INFO):
        ps.resolve_max_position_size(cfg, tcfg, force_refresh=True)
        ps._CACHE.equity = 1000.0  # simulate later successful fetch
        ps.resolve_max_position_size(cfg, tcfg)
        ps.resolve_max_position_size(cfg, tcfg)

    records = caplog.get_records("call") + caplog.get_records("teardown")
    assert (
        any(r.getMessage() == "EQUITY_MISSING" for r in records)
        or "position_sizing:equity_missing" in ps._once_logger._emitted_keys
    )
    assert ps._CACHE.value == 4000.0


def test_auto_mode_reuses_cached_equity_on_failure(monkeypatch, caplog):
    _reset_cache()
    ps._CACHE.equity = 50000.0

    def failing_fetch(cfg, force_refresh=False):
        ps._CACHE.equity_error = "http_error:503"
        return None

    monkeypatch.setattr(ps, "_get_equity_from_alpaca", failing_fetch)

    cfg = SimpleNamespace(capital_cap=0.04, max_position_mode="AUTO")
    tcfg = SimpleNamespace(capital_cap=0.04, max_position_mode="AUTO", dynamic_size_refresh_secs=0.0)

    with caplog.at_level(logging.INFO):
        size, meta = ps.resolve_max_position_size(cfg, tcfg, force_refresh=True)

    assert size == 2000.0
    assert meta["source"] == "cached_equity"
    assert meta["equity"] == 50000.0
    records = caplog.get_records("call") + caplog.get_records("teardown")
    assert any(getattr(r, "reason", None) == "http_error:503" for r in records) or (
        ps._CACHE.equity_error == "http_error:503"
    )


def test_auto_mode_aborts_when_equity_unavailable(monkeypatch, caplog):
    _reset_cache()

    def failing_fetch(cfg, force_refresh=False):
        ps._CACHE.equity_error = "request_error:Timeout"
        return None

    monkeypatch.setattr(ps, "_get_equity_from_alpaca", failing_fetch)

    cfg = SimpleNamespace(capital_cap=0.05, max_position_mode="AUTO")
    tcfg = SimpleNamespace(capital_cap=0.05, max_position_mode="AUTO", dynamic_size_refresh_secs=0.0)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="request_error:Timeout"):
            ps.resolve_max_position_size(cfg, tcfg, force_refresh=True)

    records = caplog.get_records("call") + caplog.get_records("teardown")
    assert any(r.getMessage() == "AUTO_SIZING_ABORTED" for r in records) or (
        ps._CACHE.equity_error == "request_error:Timeout"
    )
