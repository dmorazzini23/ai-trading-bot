from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core.governance_runtime import run_netting_cycle_governance


def test_run_netting_cycle_governance_halts_on_unexpected_exception(monkeypatch):
    from ai_trading.core import bot_engine

    state = SimpleNamespace(halt_trading=False, halt_reason="")
    errors: list[tuple[str, dict[str, object] | None]] = []

    monkeypatch.setattr(bot_engine, "market_is_open", lambda _now: True)
    monkeypatch.setattr(
        bot_engine,
        "_run_post_trade_learning_update",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "_run_tca_cost_calibration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "_run_policy_ablation_rollback",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "_run_replay_governance",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad governance")),
    )
    monkeypatch.setattr(
        bot_engine,
        "_run_walk_forward_governance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine.logger,
        "error",
        lambda event, extra=None: errors.append((event, extra)),
    )

    snapshot = run_netting_cycle_governance(state)

    assert isinstance(snapshot.now, datetime)
    assert snapshot.now.tzinfo is UTC
    assert snapshot.market_open_now is True
    assert state.halt_trading is True
    assert state.halt_reason == "bad governance"
    assert errors == [
        (
            "NETTING_GOVERNANCE_GUARD_FAILED",
            {
                "error_type": "ValueError",
                "detail": "bad governance",
                "step": "replay_governance",
            },
        )
    ]


def test_run_netting_cycle_governance_halts_on_learning_hook_exception(monkeypatch):
    from ai_trading.core import bot_engine

    state = SimpleNamespace(halt_trading=False, halt_reason="")
    errors: list[tuple[str, dict[str, object] | None]] = []

    monkeypatch.setattr(bot_engine, "market_is_open", lambda _now: True)
    monkeypatch.setattr(
        bot_engine,
        "_run_post_trade_learning_update",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("learning blew up")),
    )
    monkeypatch.setattr(
        bot_engine,
        "_run_tca_cost_calibration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "_run_policy_ablation_rollback",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "_run_replay_governance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "_run_walk_forward_governance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bot_engine.logger,
        "error",
        lambda event, extra=None: errors.append((event, extra)),
    )

    snapshot = run_netting_cycle_governance(state)

    assert isinstance(snapshot.now, datetime)
    assert snapshot.market_open_now is True
    assert state.halt_trading is True
    assert state.halt_reason == "learning blew up"
    assert errors == [
        (
            "NETTING_GOVERNANCE_GUARD_FAILED",
            {
                "error_type": "RuntimeError",
                "detail": "learning blew up",
                "step": "post_trade_learning_update",
            },
        )
    ]
