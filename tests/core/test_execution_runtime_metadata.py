from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core import bot_engine
from ai_trading.core.execution_runtime_metadata import (
    load_execution_prerank_runtime_state,
    store_execution_candidate_ranking_runtime_state,
)


def test_resolve_order_type_capabilities_ignores_empty_cached_mapping() -> None:
    class _Exec:
        order_type_capabilities = {"limit": True, "market": True}

    runtime = SimpleNamespace(
        broker_order_type_capabilities={},
        execution_engine=_Exec(),
    )

    caps = bot_engine._resolve_order_type_capabilities(runtime)

    assert caps == {"limit": True, "market": True}
    assert runtime.broker_order_type_capabilities == {"limit": True, "market": True}


def test_load_execution_prerank_runtime_state_normalizes_cycle_and_history() -> None:
    runtime = SimpleNamespace(
        portfolio_weights={"aapl": 0.2},
        execution_candidate_rank={" msft ": 5.0},
        execution_opportunity_quality_by_symbol={"goog": 0.9},
        _execution_prerank_cycle_idx="bad-cycle",
        _execution_candidate_last_selected_cycle={
            " aapl ": "4",
            "MSFT": "bad",
            " ": 7,
        },
    )

    state = load_execution_prerank_runtime_state(runtime)

    assert state.weights == {"AAPL": 0.2}
    assert state.runtime_rank == {"MSFT": 5.0}
    assert state.opportunity_quality == {"GOOG": 0.9}
    assert state.prerank_cycle == 1
    assert runtime._execution_prerank_cycle_idx == 1
    assert state.last_selected_cycles == {"AAPL": 4}


def test_store_execution_candidate_ranking_runtime_state_copies_inputs() -> None:
    runtime = SimpleNamespace()
    opportunity_quality = {"aapl": 0.91}
    allowed_symbols = {"msft", "aapl"}
    gate = {"thresholds": {"top": 0.8}}
    candidate_rank = {"AAPL": 1.5, "MSFT": float("nan")}
    expected_edge = {"AAPL": 12.5}
    expected_capture = {"AAPL": 8.0}
    realism_factor = {"AAPL": 0.95}
    learning_signals = {"AAPL": {"score": 1.0}}
    rank_context = {"nested": {"enabled": True}}

    store_execution_candidate_ranking_runtime_state(
        runtime,
        opportunity_quality_by_symbol=opportunity_quality,
        opportunity_allowed_symbols=allowed_symbols,
        opportunity_quality_gate=gate,
        candidate_rank=candidate_rank,
        candidate_expected_net_edge=expected_edge,
        candidate_expected_capture=expected_capture,
        edge_realism_rank_factor_by_symbol=realism_factor,
        counterfactual_signal_by_symbol=learning_signals,
        rank_context=rank_context,
    )

    opportunity_quality["aapl"] = 0.10
    gate["thresholds"]["top"] = 0.1
    candidate_rank["AAPL"] = -5.0
    expected_edge["AAPL"] = 3.0
    expected_capture["AAPL"] = 2.0
    realism_factor["AAPL"] = 0.1
    learning_signals["AAPL"]["score"] = -1.0
    rank_context["nested"]["enabled"] = False

    assert runtime.execution_opportunity_quality_by_symbol == {"AAPL": 0.91}
    assert runtime.execution_opportunity_quality_allowed_symbols == ["AAPL", "MSFT"]
    assert runtime.execution_opportunity_quality_gate == {"thresholds": {"top": 0.8}}
    assert runtime.execution_candidate_rank == {"AAPL": 1.5}
    assert runtime.execution_candidate_rank_expected_edge_bps == {"AAPL": 12.5}
    assert runtime.execution_candidate_rank_expected_capture_bps == {"AAPL": 8.0}
    assert runtime.execution_candidate_rank_realism_factor == {"AAPL": 0.95}
    assert runtime.execution_candidate_rank_learning_signals == {
        "AAPL": {"score": 1.0}
    }
    assert runtime.execution_candidate_rank_context == {"nested": {"enabled": True}}
