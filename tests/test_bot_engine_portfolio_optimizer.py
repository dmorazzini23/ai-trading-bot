from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core import bot_engine


def test_build_symbol_return_correlation_matrix_handles_basic_inputs() -> None:
    matrix = bot_engine._build_symbol_return_correlation_matrix(
        {
            "AAPL": [0.01, 0.02, -0.01, 0.03],
            "MSFT": [0.011, 0.019, -0.009, 0.028],
            "NVDA": [0.005],
        }
    )

    assert matrix["AAPL"]["AAPL"] == 1.0
    assert matrix["MSFT"]["MSFT"] == 1.0
    assert matrix["AAPL"]["NVDA"] == 0.0
    assert matrix["MSFT"]["NVDA"] == 0.0
    assert -1.0 <= matrix["AAPL"]["MSFT"] <= 1.0


def test_portfolio_optimizer_allows_trade_maps_reject_decision() -> None:
    optimizer = SimpleNamespace(
        make_portfolio_decision=lambda **_kwargs: (
            "reject",
            "correlation too high",
        )
    )

    allowed, context = bot_engine._portfolio_optimizer_allows_trade(
        optimizer=optimizer,
        symbol="AAPL",
        proposed_position=15.0,
        current_positions={"AAPL": 10.0},
        market_data={"prices": {"AAPL": 100.0}, "returns": {}, "correlations": {}},
    )

    assert allowed is False
    assert context["decision"] == "reject"
    assert context["reason"] == "correlation too high"


def test_execution_model_lineage_uses_cache_meta_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("AI_TRADING_MODEL_ID", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_VERSION", raising=False)
    monkeypatch.setattr(
        bot_engine,
        "_MODEL_CACHE_META",
        {"path": "/tmp/models/live.joblib", "signature": (1234, 1700000000)},
        raising=False,
    )

    lineage = bot_engine._execution_model_lineage()

    assert lineage["model_id"] == "live.joblib"
    assert lineage["model_version"] == "1234:1700000000"
