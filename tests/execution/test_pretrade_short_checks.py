import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading
from tests.execution.test_live_trading_degraded_feed import DummyLiveEngine


@pytest.fixture(autouse=True)
def _patch_short_prechecks(monkeypatch) -> None:
    monkeypatch.setattr(live_trading, "_safe_mode_guard", lambda *a, **k: False)
    monkeypatch.setattr(live_trading, "_require_bid_ask_quotes", lambda: False)
    monkeypatch.setattr(live_trading, "quote_fresh_enough", lambda *a, **k: True)
    monkeypatch.setattr(live_trading, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(live_trading, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(
        live_trading,
        "_call_preflight_capacity",
        lambda *a, **k: live_trading.CapacityCheck(True, int(a[3]), None),
    )
    monkeypatch.setattr(
        live_trading.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: False,
    )
    config = type(
        "Cfg",
        (),
        {
            "min_quote_freshness_ms": 1500,
            "degraded_feed_mode": "widen",
            "degraded_feed_limit_widen_bps": 0,
            "execution_require_realtime_nbbo": False,
            "execution_market_on_degraded": True,
        },
    )()
    monkeypatch.setattr(live_trading, "get_trading_config", lambda: config)


def _asset_payload(**overrides: bool) -> SimpleNamespace:
    base = {
        "shortable": True,
        "easy_to_borrow": True,
        "marginable": True,
        "short_sale_restriction": "none",
        "locate_required": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _quote_payload() -> dict:
    return {
        "bid": 100.0,
        "ask": 101.0,
        "ts": datetime.now(UTC),
    }


def test_shorting_disabled_logs_and_skips(monkeypatch, caplog) -> None:
    engine = DummyLiveEngine()
    engine.trading_client = SimpleNamespace(get_asset=lambda _symbol: _asset_payload())
    engine._cycle_account.update({"shorting_enabled": False, "margin_enabled": True})
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    caplog.set_level(logging.INFO, logger="ai_trading.execution.live_trading")
    result = engine.execute_order(
        "AAPL",
        "short",
        10,
        order_type="market",
        quote=_quote_payload(),
    )

    assert result is None
    short_skip = [rec for rec in caplog.records if rec.msg == "SHORT_ORDER_SKIPPED_LONG_ONLY_MODE"]
    assert short_skip, "Expected SHORT_ORDER_SKIPPED_LONG_ONLY_MODE log"
    assert any(rec.msg == "ACCOUNT_SHORTING_DISABLED" for rec in caplog.records)


def test_margin_disabled_logs_and_skips(monkeypatch, caplog) -> None:
    engine = DummyLiveEngine()
    engine.trading_client = SimpleNamespace(get_asset=lambda _symbol: _asset_payload())
    engine._cycle_account.update({"shorting_enabled": True, "margin_enabled": False})
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    caplog.set_level(logging.INFO, logger="ai_trading.execution.live_trading")
    result = engine.execute_order(
        "AAPL",
        "short",
        5,
        order_type="market",
        quote=_quote_payload(),
    )

    assert result is None
    short_skip = [rec for rec in caplog.records if rec.msg == "SHORT_ORDER_SKIPPED_LONG_ONLY_MODE"]
    assert short_skip, "Expected SHORT_ORDER_SKIPPED_LONG_ONLY_MODE log"
    assert any(rec.msg == "ACCOUNT_MARGIN_DISABLED" for rec in caplog.records)


def test_shortability_failure_does_not_block_long_order(monkeypatch) -> None:
    engine = DummyLiveEngine()
    engine.trading_client = SimpleNamespace(get_asset=lambda _symbol: _asset_payload(shortable=False, easy_to_borrow=False))
    engine._cycle_account.update({"shorting_enabled": True, "margin_enabled": True})
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    short_result = engine.execute_order(
        "AAPL",
        "short",
        5,
        order_type="market",
        quote=_quote_payload(),
    )

    assert short_result is None

    engine.trading_client = SimpleNamespace(get_asset=lambda _symbol: _asset_payload())

    long_result = engine.execute_order(
        "MSFT",
        "buy",
        5,
        order_type="market",
        quote=_quote_payload(),
    )

    assert long_result is not None
    assert engine.last_submitted is not None
    assert engine.last_submitted.get("symbol") == "MSFT"


def test_config_disables_shorts(monkeypatch, caplog) -> None:
    monkeypatch.setenv("TRADING__ALLOW_SHORTS", "0")
    engine = DummyLiveEngine()
    engine.trading_client = SimpleNamespace(get_asset=lambda _symbol: _asset_payload())
    engine._cycle_account.update({"shorting_enabled": True, "margin_enabled": True})
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    caplog.set_level(logging.INFO, logger="ai_trading.execution.live_trading")
    result = engine.execute_order(
        "AAPL",
        "short",
        5,
        order_type="market",
        quote=_quote_payload(),
    )

    assert result is None
    assert any(rec.msg == "SHORT_ORDER_SKIPPED_LONG_ONLY_MODE" for rec in caplog.records)
