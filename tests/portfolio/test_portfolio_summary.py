import logging
from types import SimpleNamespace

import pytest

from ai_trading.portfolio.core import log_portfolio_summary


class _DummyAccount:
    cash = 500.0
    equity = 1000.0


class _DummyAPI:
    def __init__(self, positions=None):
        self._positions = positions or []

    def get_account(self):
        return _DummyAccount()

    def list_positions(self):
        return list(self._positions)


class _DummyOrder:
    def __init__(self, symbol: str, qty: int, limit_price: float):
        self.symbol = symbol
        self.qty = qty
        self.limit_price = limit_price


class _DummyEngine:
    def __init__(self, pending):
        self._pending = pending

    def get_pending_orders(self):
        return list(self._pending)


class _DummyRiskEngine:
    def _adaptive_global_cap(self):
        return 0.0


def test_log_portfolio_summary_reports_pending_exposure(caplog):
    ctx = SimpleNamespace(
        api=_DummyAPI(),
        execution_engine=_DummyEngine([_DummyOrder("AAPL", 2, 50.0)]),
        risk_engine=_DummyRiskEngine(),
    )

    with caplog.at_level(logging.INFO):
        log_portfolio_summary(ctx)

    summary_records = [record for record in caplog.records if record.getMessage().startswith("Portfolio summary")]
    assert summary_records, "expected portfolio summary log"
    message = summary_records[0].getMessage()
    assert "pending_exposure" in message
    assert "10.00%" in message or "10.0%" in message
