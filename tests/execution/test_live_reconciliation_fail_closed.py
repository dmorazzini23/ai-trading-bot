from __future__ import annotations

from typing import Any, cast

from ai_trading.execution.live_trading import ExecutionEngine


def test_failover_post_submit_reconcile_fails_closed_in_live(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_EXECUTION_STRICT_RECONCILIATION", "1")
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine_any = cast(Any, engine)
    engine.execution_mode = "live"
    engine_any._broker_sync = None
    engine_any.synchronize_broker_state = lambda: None

    def _reconcile_durable_intents(*, open_orders) -> None:
        raise RuntimeError("intent reconcile failed")

    engine_any._reconcile_durable_intents = _reconcile_durable_intents
    engine_any._reconcile_pending_order_runtime_artifacts = lambda **_kwargs: None

    try:
        engine._failover_post_submit_reconcile(provider="secondary")
    except RuntimeError as exc:
        assert str(exc) == "FAILOVER_POST_SUBMIT_RECONCILIATION_FAILED"
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected strict live reconciliation failure")


def test_reconciliation_auto_repair_blocks_openings_on_strict_failure(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_EXECUTION_STRICT_RECONCILIATION", "1")
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine_any = cast(Any, engine)
    engine.execution_mode = "live"
    engine_any._reconciliation_auto_repair_last_mono = 0.0
    engine_any._reconciliation_auto_repair_last_context = {}
    engine_any._reconciliation_openings_block_until_mono = 0.0
    engine_any.synchronize_broker_state = lambda: None

    def _reconcile_durable_intents(*, open_orders) -> None:
        raise RuntimeError("intent reconcile failed")

    engine_any._reconcile_durable_intents = _reconcile_durable_intents
    engine_any._backfill_pending_tca_from_fill_events = lambda: {"backfilled": 0}
    engine_any._finalize_stale_pending_tca_events = lambda: {"finalized": 0}
    engine_any._record_execution_quality_event = lambda _payload: None

    context = engine._attempt_reconciliation_auto_repair(
        symbol="AAPL",
        reconcile_summary={"mismatch_count": 1},
        events_in_window=3,
        window_sec=60.0,
        burst_count=2,
        cooldown_sec=120.0,
    )

    assert context["fail_closed"] is True
    assert context["reason"] == "reconciliation_errors"
    assert engine_any._reconciliation_openings_block_until_mono > 0.0
