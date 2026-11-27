"""Behaviour-neutrality checks for the removal of ``BotState.current_regime``.

The field was never written on the netting path, so every record carried its
"sideways" default. These tests pin that the recorded values, and the
reachable ``update_bot_mode`` outcomes, are unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.contracts import build_decision_journal
from ai_trading.core import bot_engine
from ai_trading.core.constants import RECORDED_MARKET_REGIME_LABEL
from ai_trading.core.decision_log import DecisionRecorder
from ai_trading.core.execution_intent import _regime_context_from_snapshot
from ai_trading.core.netting import NettedTarget, SleeveProposal


def test_recorded_label_is_the_old_default_and_normalizes_to_itself() -> None:
    assert RECORDED_MARKET_REGIME_LABEL == "sideways"
    # Both record sites used to pass the field through _normalize_regime_name.
    assert bot_engine._normalize_regime_name(RECORDED_MARKET_REGIME_LABEL) == "sideways"
    assert bot_engine._normalize_regime_name("") == "sideways"


def test_regime_state_field_and_uncalled_detector_are_gone() -> None:
    assert "current_regime" not in bot_engine.BotState.__dataclass_fields__
    for name in (
        "check_market_regime",
        "detect_regime_state",
        "_market_breadth",
        "_regime_allows_trading",
    ):
        assert not hasattr(bot_engine, name), name


def _patch_mode_switch(monkeypatch: pytest.MonkeyPatch, *, avg_r: float, dd: float, enabled: bool = True) -> list[str]:
    cfg = SimpleNamespace(
        trading_mode_adaptive_enabled=enabled,
        trading_mode_precedence="strict_mode",
        trading_mode="balanced",
    )
    calls: list[str] = []

    def _avg() -> float:
        calls.append("avg")
        return avg_r

    monkeypatch.setattr(bot_engine, "_get_trading_config", lambda: cfg)
    monkeypatch.setattr(bot_engine, "_trading_config_from_env", lambda **_kw: cfg)
    monkeypatch.setattr(bot_engine, "_average_reward", _avg)
    monkeypatch.setattr(bot_engine, "_current_drawdown", lambda: dd)
    monkeypatch.setattr(bot_engine, "params", {})
    monkeypatch.setattr(bot_engine, "ctx", None)
    return calls


# (start mode, drawdown, average reward, expected mode). Before the change the
# third branch ("aggressive") also required regime == "trending", which the
# always-"sideways" field never satisfied, so a high reward gave "balanced".
@pytest.mark.parametrize(
    ("start", "dd", "avg_r", "expected"),
    [
        ("balanced", 0.06, 0.10, "conservative"),
        ("balanced", 0.00, -0.02, "conservative"),
        ("balanced", 0.06, -0.02, "conservative"),
        ("conservative", 0.00, 0.10, "balanced"),
        ("conservative", 0.00, 0.00, "balanced"),
        ("aggressive", 0.00, 0.10, "balanced"),
        ("balanced", 0.05, -0.01, "balanced"),
    ],
)
def test_update_bot_mode_reachable_truth_table(
    monkeypatch: pytest.MonkeyPatch, start: str, dd: float, avg_r: float, expected: str
) -> None:
    _patch_mode_switch(monkeypatch, avg_r=avg_r, dd=dd)
    state = SimpleNamespace(mode_obj=SimpleNamespace(mode=start))

    bot_engine.update_bot_mode(state)  # type: ignore[arg-type]

    assert state.mode_obj.mode == expected


def test_update_bot_mode_never_selects_aggressive(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mode_switch(monkeypatch, avg_r=1.0, dd=0.0)
    # Even a state object carrying a "trending" attribute no longer matters.
    state = SimpleNamespace(mode_obj=SimpleNamespace(mode="balanced"), current_regime="trending")

    bot_engine.update_bot_mode(state)  # type: ignore[arg-type]

    assert state.mode_obj.mode == "balanced"


def test_update_bot_mode_returns_early_when_adaptive_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_mode_switch(monkeypatch, avg_r=-1.0, dd=1.0, enabled=False)
    mode_obj = SimpleNamespace(mode="aggressive")
    state = SimpleNamespace(mode_obj=mode_obj)

    bot_engine.update_bot_mode(state)  # type: ignore[arg-type]

    assert state.mode_obj is mode_obj
    assert calls == []


def test_prerank_shadow_snapshot_records_sideways_with_bot_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "1")
    payloads: list[dict[str, Any]] = []
    monkeypatch.setattr(bot_engine, "_record_shadow_prediction", lambda payload: payloads.append(dict(payload)))
    monkeypatch.setattr(bot_engine, "_ml_shadow_provider_snapshot", lambda: {})
    monkeypatch.setattr(bot_engine, "_ml_shadow_quote_snapshot", lambda *a, **k: {})

    for runtime in (SimpleNamespace(state=bot_engine.BotState()), SimpleNamespace(state={}), None):
        bot_engine._record_prerank_shadow_snapshot(
            ranked_symbols=["AAPL"],
            selected_symbols=["AAPL"],
            rank_source="runtime_rank",
            top_n=1,
            runtime=runtime,
        )

    assert [p["market_regime"] for p in payloads] == ["sideways", "sideways", "sideways"]


class _StopCycle(Exception):
    pass


def test_netting_cycle_decision_quote_snapshot_records_sideways(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _CapturingRecorder(DecisionRecorder):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            raise _StopCycle

    cfg = SimpleNamespace(decision_log_path=None, paper_sampling_allowed_symbols=(), execution_mode="paper")
    monkeypatch.setattr("ai_trading.core.decision_log.DecisionRecorder", _CapturingRecorder)
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _runtime: cfg)
    monkeypatch.setattr(bot_engine, "_active_effective_policy", lambda _state, _cfg: None)
    monkeypatch.setattr(
        "ai_trading.core.governance_runtime.run_netting_cycle_governance",
        lambda _state: SimpleNamespace(now=datetime(2026, 11, 27, 15, 0, tzinfo=UTC), market_open_now=True),
    )
    monkeypatch.setattr(bot_engine, "_ml_shadow_quote_snapshot", lambda symbol=None: {"status": "fresh"})

    with pytest.raises(_StopCycle):
        bot_engine._run_netting_cycle(bot_engine.BotState(), SimpleNamespace(), "loop", 0.0)

    quote = captured["quote_snapshot_func"]("AAPL")
    assert quote["market_regime"] == "sideways"
    assert quote["execution_profile"] == "standard"


def _recorder(quote: dict[str, Any]) -> DecisionRecorder:
    return DecisionRecorder(
        runtime=type("Runtime", (), {"execution_candidate_rank_expected_edge_bps": {}})(),
        path="runtime/decision_records.jsonl",
        write_impl=lambda _record, _path: None,
        dedupe_gate_root_causes=lambda gates: list(dict.fromkeys(gates)),
        session_bucket_from_ts=lambda _ts: "midday",
        safe_float=lambda value: float(value) if value is not None else None,
        quote_snapshot_func=lambda _symbol: dict(quote),
    )


def _build(recorder: DecisionRecorder, config_snapshot: dict[str, Any]) -> Any:
    bar_ts = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)
    proposal = SleeveProposal(
        symbol="AAPL",
        sleeve="day",
        bar_ts=bar_ts,
        target_dollars=1000.0,
        expected_edge_bps=12.0,
        expected_cost_bps=3.0,
        score=0.7,
        confidence=0.8,
    )
    target = NettedTarget(
        symbol="AAPL", bar_ts=bar_ts, target_dollars=1000.0, target_shares=5.0, proposals=[proposal]
    )
    return recorder.build_record(
        symbol="AAPL",
        bar_ts=bar_ts,
        net_target=target,
        sleeves=[proposal],
        gates=["OK_TRADE"],
        config_snapshot=config_snapshot,
    )


def test_decision_record_sideways_derives_no_trend_or_volatility_regime() -> None:
    record = _build(_recorder({"market_regime": RECORDED_MARKET_REGIME_LABEL}), {})

    assert record.metrics["market_regime"] == "sideways"
    assert "trend_regime" not in record.metrics
    assert "volatility_regime" not in record.metrics
    assert build_decision_journal(record).metadata["market_regime"] == "sideways"


def test_current_regime_key_is_no_longer_a_market_regime_source() -> None:
    # Nothing produces a "current_regime" key; it is no longer consulted.
    record = _build(_recorder({}), {"current_regime": "trending"})
    assert record.metrics.get("market_regime") in (None, "")
    assert record.metrics["metadata_missing_reasons"]["market_regime"] == "market_regime_unavailable"
    assert "market_regime" not in build_decision_journal(record).metadata

    context = _regime_context_from_snapshot({"current_regime": "trending"}, bar_ts=None)
    assert "market_regime" not in context
    context = _regime_context_from_snapshot(
        {"market_regime": "sideways", "regime_profile": "balanced"}, bar_ts=None
    )
    assert context["market_regime"] == "sideways"
