from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ai_trading.core import bot_engine


def test_module_isolation_preserves_real_numpy() -> None:
    with patch.dict(sys.modules, {}, clear=True):
        assert sys.modules.get("numpy") is np
        assert isinstance(np.asarray([0.01, 0.02]), np.ndarray)


def test_build_symbol_return_correlation_matrix_handles_basic_inputs() -> None:
    assert isinstance(bot_engine.np.asarray([0.01, 0.02]), np.ndarray)
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


def test_execution_model_lineage_names_rule_scorer_when_env_missing(monkeypatch) -> None:
    for name in (
        "AI_TRADING_MODEL_ID",
        "AI_TRADING_MODEL_VERSION",
        "AI_TRADING_DATASET_HASH",
        "AI_TRADING_FEATURE_VERSION",
        "AI_TRADING_MODEL_ARTIFACT_HASH",
    ):
        monkeypatch.delenv(name, raising=False)

    lineage = bot_engine._execution_model_lineage()

    # Openings must carry model_id/model_version for the submit-context check;
    # the decision source is the rule scorer, not a model artifact.
    assert lineage == {"model_id": "rule_scorer", "model_version": "score_from_bars"}
    assert not hasattr(bot_engine, "_MODEL_CACHE_META")


def test_execution_model_lineage_explicit_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_MODEL_ID", "ml-prod")
    monkeypatch.setenv("AI_TRADING_MODEL_VERSION", "v1")
    monkeypatch.setenv("AI_TRADING_DATASET_HASH", "ds-123")
    monkeypatch.setenv("AI_TRADING_FEATURE_VERSION", "feat-9")
    monkeypatch.setenv("AI_TRADING_MODEL_ARTIFACT_HASH", "artifact-xyz")

    lineage = bot_engine._execution_model_lineage()

    assert lineage == {
        "model_id": "ml-prod",
        "model_version": "v1",
        "dataset_hash": "ds-123",
        "feature_version": "feat-9",
        "model_artifact_hash": "artifact-xyz",
    }
