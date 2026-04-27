from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.portfolio import core


pd = pytest.importorskip("pandas")


def test_compute_portfolio_weights_inverse_price_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = {"A": 10.0, "B": 20.0, "BAD": None}
    monkeypatch.setattr(core, "get_execution_latest_price", lambda _ctx, symbol: prices.get(symbol))

    weights = core.compute_portfolio_weights(SimpleNamespace(portfolio_weight_method="inverse_price"), ["A", "B", "BAD"])

    assert weights == {"A": pytest.approx(2 / 3), "B": pytest.approx(1 / 3)}

    monkeypatch.setattr(core, "get_execution_latest_price", lambda _ctx, _symbol: 0.0)
    assert core.compute_portfolio_weights(SimpleNamespace(portfolio_weight_method="inverse_price"), ["A"]) == {}
    assert core.compute_portfolio_weights(SimpleNamespace(), []) == {}


def test_compute_portfolio_weights_inverse_vol_falls_back_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {
        "LOW": pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]}),
        "HIGH": pd.DataFrame({"close": [100, 110, 90, 115, 85, 120]}),
    }
    ctx = SimpleNamespace(
        portfolio_weight_method="inverse_vol",
        data_fetcher=SimpleNamespace(get_daily_df=lambda _ctx, symbol: frames[symbol]),
    )
    monkeypatch.setattr(core, "get_execution_latest_price", lambda _ctx, _symbol: 100.0)

    weights = core.compute_portfolio_weights(ctx, ["LOW", "HIGH"])

    assert weights["LOW"] > weights["HIGH"]
    assert sum(weights.values()) == pytest.approx(1.0)

    ctx.data_fetcher = SimpleNamespace(get_daily_df=lambda _ctx, _symbol: pd.DataFrame({"open": [1, 2]}))
    assert core.compute_portfolio_weights(ctx, ["LOW", "HIGH"]) == {
        "LOW": pytest.approx(0.5),
        "HIGH": pytest.approx(0.5),
    }


def test_log_portfolio_summary_uses_ledger_positions_and_pending_market_price(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    account = SimpleNamespace(cash="1000", equity="2000")
    api = SimpleNamespace(get_account=lambda: account, list_positions=lambda: [])
    risk_engine = SimpleNamespace(
        _positions={"AAPL": 3},
        _adaptive_global_cap=lambda: 12.5,
    )
    pending = SimpleNamespace(symbol="MSFT", qty="2", limit_price=None, price=None)
    execution_engine = SimpleNamespace(get_pending_orders=lambda: [pending])
    ctx = SimpleNamespace(
        api=api,
        risk_engine=risk_engine,
        execution_engine=execution_engine,
        portfolio_weights={"AAPL": 1.0},
    )
    monkeypatch.setattr(core, "get_latest_price", lambda _ctx, symbol: {"AAPL": 100.0, "MSFT": 50.0}[symbol])

    with caplog.at_level(logging.INFO):
        core.log_portfolio_summary(ctx)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Portfolio summary (ledger)" in message and "15.00%" in message for message in messages)
    assert any("CYCLE SUMMARY adaptive_cap=12.5" in message for message in messages)
