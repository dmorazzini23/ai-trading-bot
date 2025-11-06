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

    caplog.set_level(logging.WARNING, logger="ai_trading.execution.live_trading")
    result = engine.execute_order(
        "AAPL",
        "short",
        10,
        order_type="market",
        quote=_quote_payload(),
    )

    assert result is None
    records = [rec for rec in caplog.records if rec.msg == "PRECHECK_SHORTABILITY_FAILED"]
    assert records, "Expected PRECHECK_SHORTABILITY_FAILED log"
    entry = records[-1]
    assert entry.reason == "account_shorting_disabled"
    assert entry.account_shorting_enabled is False
    assert entry.account_margin_enabled is True


def test_margin_disabled_logs_and_skips(monkeypatch, caplog) -> None:
    engine = DummyLiveEngine()
    engine.trading_client = SimpleNamespace(get_asset=lambda _symbol: _asset_payload())
    engine._cycle_account.update({"shorting_enabled": True, "margin_enabled": False})
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    caplog.set_level(logging.WARNING, logger="ai_trading.execution.live_trading")
    result = engine.execute_order(
        "AAPL",
        "short",
        5,
        order_type="market",
        quote=_quote_payload(),
    )

    assert result is None
    records = [rec for rec in caplog.records if rec.msg == "PRECHECK_MARGIN_DISABLED"]
    assert records, "Expected PRECHECK_MARGIN_DISABLED log"
    entry = records[-1]
    assert entry.reason == "account_margin_disabled"
    assert entry.account_margin_enabled is False
