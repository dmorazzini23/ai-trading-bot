"""Capacity exhaustion should raise a non-retryable broker error."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from ai_trading.execution import live_trading as lt
from ai_trading.execution import guards


class DummyAPIError(lt.APIError):
    def __init__(
        self,
        status_code: int | None = 403,
        code: str | None = "40310000",
        message: str = "insufficient day trading buying power",
    ) -> None:
        payload = {"message": message, "code": code}
        http_error = None
        if status_code is not None:
            http_error = SimpleNamespace(
                response=SimpleNamespace(status_code=status_code),
                request=None,
            )
        super().__init__(json.dumps(payload), http_error=http_error)
        self._status_code = status_code
        self._code = code
        self._message = message

    @property
    def status_code(self):  # type: ignore[override]
        return self._status_code

    @property
    def code(self):  # type: ignore[override]
        return self._code

    @property
    def message(self):  # type: ignore[override]
        return self._message


def _engine() -> lt.ExecutionEngine:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {"capacity_skips": 0, "skipped_orders": 0, "retry_count": 0}
    engine.logger = lt.logger
    return engine


def test_nonretryable_capacity_error(monkeypatch, caplog):
    engine = _engine()
    caplog.set_level(logging.DEBUG)

    err = DummyAPIError()
    result = engine._handle_nonretryable_api_error(err, {"symbol": "AAPL"})

    assert isinstance(result, lt.NonRetryableBrokerError)
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1
    assert result.args[0] == "insufficient_day_trading_buying_power"
    assert result.detail == "insufficient day trading buying power"
    messages = [rec.getMessage() for rec in caplog.records]
    assert "BROKER_CAPACITY_EXCEEDED" in messages
    assert "BROKER_CAPACITY_EXCEEDED_DETAIL" in messages
    info_records = [rec for rec in caplog.records if rec.getMessage() == "BROKER_CAPACITY_EXCEEDED"]
    assert info_records and info_records[0].levelno == logging.INFO
    assert getattr(info_records[0], "reason", None) == "insufficient_day_trading_buying_power"


def test_capacity_error_passthrough_for_other_codes():
    engine = _engine()
    err = DummyAPIError(status_code=500, code="other", message="different")
    assert engine._handle_nonretryable_api_error(err, {"symbol": "AAPL"}) is None
    assert engine.stats["capacity_skips"] == 0


def test_skip_shorting_when_asset_not_shortable(monkeypatch, caplog):
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine._refresh_settings = lambda: None
    engine._ensure_initialized = lambda: True
    engine._pre_execution_checks = lambda: True
    engine.is_initialized = True
    engine.shadow_mode = False
    engine.stats = {}

    class DummyClient:
        def __init__(self):
            self.symbols = []

        def get_asset(self, symbol):
            self.symbols.append(symbol)
            return type("Asset", (), {"shortable": False})()

    client = DummyClient()
    engine.trading_client = client

    called = {"preflight": False}

    def fake_preflight(symbol, side, price_hint, quantity, trading_client, account=None):
        called["preflight"] = True
        return lt.CapacityCheck(True, quantity, None)

    monkeypatch.setattr(lt, "preflight_capacity", fake_preflight)

    caplog.set_level(logging.WARNING)

    result = engine.submit_market_order("AAPL", "sell", 1)

    assert result is None
    assert called["preflight"] is False
    assert client.symbols == ["AAPL"]
    messages = [record.getMessage() for record in caplog.records]
    assert any("ORDER_SKIPPED_NONRETRYABLE" in msg for msg in messages)
    assert any("shorting_disabled" in msg for msg in messages)


def test_skip_when_pdt_limit_reached(monkeypatch, caplog):
    guards.STATE.pdt = guards.PDTState()
    guards.STATE.shadow_cycle = False
    guards.STATE.shadow_cycle_forced = False
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine._refresh_settings = lambda: None
    engine._ensure_initialized = lambda: True
    engine._pre_execution_checks = lambda: True
    engine.is_initialized = True
    engine.shadow_mode = False
    engine.stats = {}
    monkeypatch.setenv("EXECUTION_DAYTRADE_LIMIT", "3")

    account_snapshot = {"pattern_day_trader": True, "daytrade_count": 3}
    engine._get_account_snapshot = lambda: account_snapshot

    called = {"preflight": False}

    def forbidden_preflight(*args, **kwargs):
        called["preflight"] = True
        raise AssertionError("preflight should not be called when PDT blocks")

    monkeypatch.setattr(lt, "preflight_capacity", forbidden_preflight)

    caplog.set_level(logging.INFO)

    result = engine.submit_market_order("AAPL", "buy", 1)

    assert result is None
    assert not called["preflight"]
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("ORDER_SKIPPED_NONRETRYABLE" == msg for msg in messages)
    reasons = [getattr(record, "reason", None) for record in caplog.records]
    assert "pdt_limit_reached" in reasons


def test_pdt_limit_imminent_warns_but_allows_trade(monkeypatch, caplog):
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine._refresh_settings = lambda: None
    engine._ensure_initialized = lambda: True
    engine._pre_execution_checks = lambda: True
    engine.is_initialized = True
    engine.shadow_mode = False
    engine.stats = {
        "total_execution_time": 0.0,
        "total_orders": 0,
        "successful_orders": 0,
        "failed_orders": 0,
    }
    engine.trading_client = object()
    monkeypatch.setenv("EXECUTION_DAYTRADE_LIMIT", "3")

    account_snapshot = {"pattern_day_trader": True, "daytrade_count": 2}
    engine._get_account_snapshot = lambda: account_snapshot

    preflight_calls = {"count": 0}

    def fake_preflight(symbol, side, price_hint, quantity, trading_client, account=None):
        preflight_calls["count"] += 1
        return lt.CapacityCheck(True, quantity, None)

    monkeypatch.setattr(lt, "preflight_capacity", fake_preflight)
    monkeypatch.setattr(engine, "_execute_with_retry", lambda fn, payload: {"id": "ok"})

    caplog.set_level(logging.WARNING)

    result = engine.submit_market_order("AAPL", "buy", 1)

    assert result == {"id": "ok"}
    assert preflight_calls["count"] == 1
    assert engine.stats["total_orders"] == 1
    assert engine.stats["successful_orders"] == 1
    assert engine.stats["failed_orders"] == 0
    assert "capacity_skips" not in engine.stats

    warnings = [record for record in caplog.records if record.getMessage() == "PDT_LIMIT_IMMINENT"]
    assert warnings, "expected PDT warning"
    warning = warnings[0]
    assert warning.levelno == logging.WARNING
    assert getattr(warning, "daytrade_count", None) == 2
    assert getattr(warning, "daytrade_limit", None) == 3


def test_preflight_helper_supports_account_kwarg():
    lt._preflight_supports_account_kwarg.cache_clear()
    account = {"id": "acct"}
    captured: dict[str, object] = {}

    def new_signature(symbol, side, price_hint, quantity, broker, *, account=None):
        captured["args"] = (symbol, side, price_hint, quantity, broker)
        captured["account"] = account
        return lt.CapacityCheck(True, int(quantity), None)

    check = lt._call_preflight_capacity(
        "AAPL",
        "buy",
        123.45,
        5,
        object(),
        account,
        preflight_fn=new_signature,
    )

    assert check.can_submit is True
    assert captured["account"] is account
    assert captured["args"][:4] == ("AAPL", "buy", 123.45, 5)


def test_preflight_helper_legacy_signature():
    lt._preflight_supports_account_kwarg.cache_clear()
    called: dict[str, object] = {}

    def legacy_signature(symbol, side, price_hint, quantity, broker):
        called["args"] = (symbol, side, price_hint, quantity, broker)
        return lt.CapacityCheck(True, int(quantity), None)

    check = lt._call_preflight_capacity(
        "MSFT",
        "sell",
        250.0,
        3,
        object(),
        {"id": "acct"},
        preflight_fn=legacy_signature,
    )

    assert check.can_submit is True
    assert called["args"][:4] == ("MSFT", "sell", 250.0, 3)
