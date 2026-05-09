from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd

from ai_trading import rebalancer


def test_portfolio_first_rebalance_success_path(monkeypatch) -> None:
    settings = SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True)
    monkeypatch.setattr(rebalancer, "get_settings", lambda: settings)

    class _RegimeDetector:
        def detect_current_regime(self, market_data):
            assert market_data["prices"]["AAPL"] == 101.0
            return SimpleNamespace(value="normal"), SimpleNamespace()

        def calculate_dynamic_thresholds(self, _regime, _metrics):
            return SimpleNamespace(minimum_improvement_threshold=0.02, rebalance_drift_threshold=0.04)

    class _Optimizer:
        improvement_threshold = 0.0
        rebalance_drift_threshold = 0.0

        def should_trigger_rebalance(self, current_positions, target_weights, current_prices):
            assert current_positions == {"AAPL": 2.0}
            assert target_weights == {"AAPL": 1.0}
            assert current_prices == {"AAPL": 101.0, "SPY": 101.0}
            return True, "drift"

    class _TaxRebalancer:
        regime_detector = _RegimeDetector()
        portfolio_optimizer = _Optimizer()

        def calculate_optimal_rebalance(self, current_positions, target_weights, current_prices, account_equity):
            assert current_positions["AAPL"]["quantity"] == 2.0
            assert account_equity == 202.0
            return {"portfolio_drift": 0.1, "total_tax_impact": 0.0, "rebalance_trades": []}

    class _Fetcher:
        def get_daily_df(self, _ctx, _symbol):
            return pd.DataFrame({"close": [100.0, 101.0], "volume": [1000, 1200]})

    monkeypatch.setattr(rebalancer, "_rebalancer", _TaxRebalancer())
    ctx = SimpleNamespace(
        portfolio_positions={"AAPL": 2, "ZERO": 0},
        target_weights={"AAPL": 1.0},
        data_fetcher=_Fetcher(),
    )

    rebalancer.portfolio_first_rebalance(ctx)

    assert ctx.rebalance_plan["portfolio_drift"] == 0.1
    assert isinstance(ctx.last_portfolio_rebalance, datetime)


def test_rebalance_helpers_extract_targets_market_data_and_reasons(monkeypatch) -> None:
    ctx = SimpleNamespace(
        positions={"AAPL": "3", "BAD": "not-a-number", "TINY": 0.0001},
        data_fetcher=SimpleNamespace(
            get_daily_df=lambda _ctx, symbol: pd.DataFrame(
                {"close": [100.0, 102.0, 101.0], "volume": [10, 20, 30]}
            )
            if symbol != "SPY"
            else pd.DataFrame({"close": [400.0], "volume": [100]})
        ),
    )

    assert rebalancer._get_current_positions_for_rebalancing(ctx) == {"AAPL": 3.0}
    assert rebalancer._get_target_weights_for_rebalancing(ctx) == {}
    market_data = rebalancer._prepare_rebalancing_market_data(ctx)
    assert market_data["prices"]["AAPL"] == 101.0
    assert market_data["volumes"]["AAPL"] == 20.0
    assert market_data["returns"]["AAPL"]

    monkeypatch.setattr(rebalancer, "_rebalancer", None)
    no_prior = rebalancer._check_portfolio_first_rebalancing(SimpleNamespace(), 0.01, 0.05)
    assert no_prior == (True, "No previous rebalancing recorded")

    old_ctx = SimpleNamespace(last_portfolio_rebalance=datetime.now(UTC) - timedelta(days=95))
    due = rebalancer._check_portfolio_first_rebalancing(old_ctx, 0.01, 0.05)
    assert due[0] is True
    assert "Quarterly rebalance due" in due[1]

    recent_ctx = SimpleNamespace(last_portfolio_rebalance=datetime.now(UTC) - timedelta(days=10))
    skip = rebalancer._check_portfolio_first_rebalancing(recent_ctx, 0.01, 0.05)
    assert skip[0] is False


def test_tax_recommendations_and_error_fallbacks(monkeypatch) -> None:
    tax = rebalancer.TaxAwareRebalancer()
    assert tax._generate_rebalance_recommendations([]) == ["Rebalancing plan appears tax-efficient"]

    bad_trade: dict[str, Any] = {"trade_value": "bad", "tax_impact": {"tax_liability": 1.0}}
    assert tax._generate_rebalance_recommendations([bad_trade]) == [
        "Manual review recommended due to analysis error"
    ]

    assert tax.calculate_optimal_rebalance(
        {"AAPL": {"quantity": 1}},
        {"AAPL": 1.0},
        {"AAPL": object()},
        1000.0,
    )["rebalance_trades"] == []

    monkeypatch.setattr(
        rebalancer,
        "_get_current_positions_for_rebalancing",
        lambda _ctx: (_ for _ in ()).throw(ValueError("bad positions")),
    )
    assert rebalancer._prepare_rebalancing_market_data(SimpleNamespace()) == {}


def test_enhanced_rebalance_branches_and_loop_error(monkeypatch) -> None:
    settings = SimpleNamespace(
        ENABLE_PORTFOLIO_FEATURES=True,
        portfolio_drift_threshold=0.05,
        rebalance_sleep_seconds=1,
    )
    monkeypatch.setattr(rebalancer, "get_settings", lambda: settings)
    monkeypatch.setattr(rebalancer, "rebalance_interval_min", lambda: 1)
    rebalancer._last_rebalance = datetime.now(UTC) - timedelta(minutes=2)
    monkeypatch.setattr(rebalancer, "compute_portfolio_weights", lambda _ctx, _symbols: {"AAPL": 0.8})
    monkeypatch.setattr(rebalancer, "init_rebalancer", lambda: None)
    monkeypatch.setattr(rebalancer, "_check_portfolio_first_rebalancing", lambda *_args: (True, "drift"))
    calls: list[str] = []
    def record_portfolio_first(_ctx: object) -> None:
        calls.append("portfolio_first")

    monkeypatch.setattr(rebalancer, "portfolio_first_rebalance", record_portfolio_first)

    rebalancer.enhanced_maybe_rebalance(SimpleNamespace(portfolio_weights={"AAPL": 0.5}))
    assert calls == ["portfolio_first"]

    settings.ENABLE_PORTFOLIO_FEATURES = False
    rebalancer._last_rebalance = datetime.now(UTC) - timedelta(minutes=2)
    def record_basic(_ctx: object) -> None:
        calls.append("basic")

    monkeypatch.setattr(rebalancer, "rebalance_portfolio", record_basic)
    rebalancer.enhanced_maybe_rebalance(SimpleNamespace(portfolio_weights={"AAPL": 0.5}))
    assert calls[-1] == "basic"

    def fake_thread(target, daemon=False):
        class _Thread:
            def start(self):
                try:
                    target()
                except StopIteration:
                    pass

        return _Thread()

    sleep_calls: list[int] = []
    monkeypatch.setattr(rebalancer.threading, "Thread", fake_thread)
    monkeypatch.setattr(rebalancer, "maybe_rebalance", lambda _ctx: (_ for _ in ()).throw(ValueError("boom")))
    def stop_after_sleep(seconds: int) -> None:
        sleep_calls.append(seconds)
        raise StopIteration

    monkeypatch.setattr(rebalancer.time, "sleep", stop_after_sleep)
    rebalancer.start_rebalancer("ctx")
    assert sleep_calls == [1]
