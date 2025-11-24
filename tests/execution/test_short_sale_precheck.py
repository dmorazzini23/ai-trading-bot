from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading


@pytest.fixture(autouse=True)
def _reset_long_only(monkeypatch):
    monkeypatch.setattr(live_trading, "_LONG_ONLY_ACCOUNT_MODE", False)
    monkeypatch.setattr(live_trading, "_LONG_ONLY_ACCOUNT_REASON", None)
    monkeypatch.setattr(live_trading, "_ACCOUNT_MARGIN_WARNING_LOGGED", False)
    monkeypatch.setattr(live_trading, "_ACCOUNT_SHORTING_WARNING_LOGGED", False)
    monkeypatch.setattr(live_trading, "_CONFIG_LONG_ONLY_LOGGED", False)


def _shortable_asset(**kwargs):
    defaults = {
        "shortable": True,
        "easy_to_borrow": True,
        "marginable": True,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _margin_account(**kwargs):
    defaults = {
        "shorting_enabled": True,
        "margin_enabled": True,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _client_for(asset):
    return SimpleNamespace(get_asset=lambda symbol: asset)


def test_short_precheck_allows_shortable_margin_account():
    client = _client_for(_shortable_asset())
    account = _margin_account()

    ok, extras, reason = live_trading._short_sale_precheck(
        None,
        client,
        symbol="AMZN",
        side="sell",
        closing_position=False,
        account_snapshot=account,
    )

    assert ok is True
    assert reason is None
    assert extras is not None
    assert extras["asset_shortable"] is True
    assert extras["account_shorting_enabled"] is True
    assert extras["account_margin_enabled"] is True


def test_short_precheck_blocks_non_shortable_asset():
    client = _client_for(_shortable_asset(shortable=False))
    account = _margin_account()

    ok, extras, reason = live_trading._short_sale_precheck(
        None,
        client,
        symbol="META",
        side="sell",
        closing_position=False,
        account_snapshot=account,
    )

    assert ok is False
    assert reason == "shortability"
    assert extras is not None
    assert extras["reason"] == "asset_not_shortable"


def test_short_precheck_allows_when_account_snapshot_missing():
    client = _client_for(_shortable_asset())

    ok, extras, reason = live_trading._short_sale_precheck(
        None,
        client,
        symbol="NVDA",
        side="sell",
        closing_position=False,
        account_snapshot=None,
    )

    assert ok is True
    assert reason is None
    assert extras is not None
    assert extras["asset_lookup_failed"] is False
    assert extras["account_shorting_enabled"] in {True, None}


def test_short_precheck_sets_long_only_for_cash_account(caplog):
    caplog.set_level(logging.WARNING)
    client = _client_for(_shortable_asset())
    account = _margin_account(shorting_enabled=False, margin_enabled=False)

    ok, extras, reason = live_trading._short_sale_precheck(
        None,
        client,
        symbol="AAPL",
        side="sell",
        closing_position=False,
        account_snapshot=account,
    )

    assert ok is False
    assert reason == "long_only"
    assert extras is not None
    assert extras["long_only_source"] in {"account_margin_disabled", "account_shorting_disabled"}
    long_only_active, long_only_reason = live_trading._long_only_state()
    assert long_only_active is True
    assert long_only_reason in {"account_margin_disabled", "account_shorting_disabled"}
    logged_msgs = {record.msg for record in caplog.records}
    assert ("ACCOUNT_MARGIN_DISABLED" in logged_msgs) or ("ACCOUNT_SHORTING_DISABLED" in logged_msgs)
