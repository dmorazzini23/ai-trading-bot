"""Regressions for the seven September repository-audit findings."""
from datetime import UTC, datetime
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from alpaca.common.exceptions import APIError
from requests import HTTPError, Response

from ai_trading.core.errors import classify_exception
from ai_trading.core.retry import retry_idempotent
from ai_trading.oms.reconcile import fetch_broker_positions, reconcile
from ai_trading.services.reconciliation import ReconciliationService


@pytest.mark.parametrize("qty", ["invalid", None, "nan", "inf"])
def test_bad_snapshot_preserves_targets(qty):
    ctx = SimpleNamespace(api=SimpleNamespace(get_all_positions=lambda: [
        {"symbol": "QQQ", "qty": "1"}, {"symbol": "SPY", "qty": qty}]),
        stop_targets={"SPY": 90}, take_profit_targets={"SPY": 110})
    with pytest.raises(RuntimeError):
        ReconciliationService().reconcile_position_targets(ctx, logger=Mock(), targets_lock=Lock())
    assert ctx.stop_targets == {"SPY": 90}
    assert ctx.take_profit_targets == {"SPY": 110}


@pytest.mark.parametrize("qty", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reconciliation_rejected(qty):
    with pytest.raises(ValueError):
        reconcile({"SPY": 10}, {"SPY": qty})
    with pytest.raises(RuntimeError):
        fetch_broker_positions(SimpleNamespace(get_all_positions=lambda: [{"symbol": "SPY", "qty": qty}]))


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError):
        reconcile({}, {}, tolerance_shares=tolerance)


@pytest.mark.parametrize("code,attempts", [(429, 3), (503, 3), (401, 1)])
def test_actual_sdk_errors_reach_retry_and_breaker(code, attempts):
    response = Response()
    response.status_code = code
    error = APIError('{"code":%d,"message":"broker failure"}' % code,
                     HTTPError(response=response))
    operation = Mock(side_effect=error)
    breakers = Mock()
    with pytest.raises(APIError):
        retry_idempotent(operation, dep="broker_positions", breakers=breakers,
                         classify_exception=classify_exception, max_attempts=3,
                         base_delay=0, jitter=0)
    assert operation.call_count == attempts
    assert breakers.record_failure.call_count == attempts


@pytest.mark.parametrize("mode", ["cancel_error", "status_error", "timeout", "filled"])
def test_opposite_cancellation_requires_confirmation(monkeypatch, mode):
    from ai_trading.execution import live_trading as live
    engine = object.__new__(live.LiveTradingExecutionEngine)
    engine._cancel_order_alpaca = Mock(side_effect=RuntimeError("cancel") if mode == "cancel_error" else None)
    engine._get_order_status_alpaca = Mock(side_effect=RuntimeError("status") if mode == "status_error" else None,
                                         return_value={"status": "filled" if mode == "filled" else "new"})
    counter = iter([0, .1, 1, 2, 3, 4, 5, 6, 7])
    monkeypatch.setattr(live, "monotonic_time", lambda: next(counter))
    monkeypatch.setattr(live.time, "sleep", lambda _: None)
    assert engine._cancel_opposite_orders([{"id": "one"}], "SPY", "buy", timeout=.5) == []


def test_disabled_validation_is_not_confirmation():
    from ai_trading.training.after_hours import _oof_promotion_authority_guard
    best = SimpleNamespace(score_orientation="direct", score_orientation_report={}, selected_threshold=.7)
    gate = {"enabled": False, "gate_passed": True, "reason": "disabled"}
    result = _oof_promotion_authority_guard(best=best, default_threshold=.5,
        live_execution_quality_gate=gate, champion_challenger_ab=gate, promotion_confidence_gate=gate)
    assert result["gate_passed"] is False


def test_broker_api_error_sets_reconciliation_halt(monkeypatch):
    from ai_trading.core import governance_runtime as governance
    response = Response()
    response.status_code = 401
    error = APIError('{"code":401,"message":"unauthorized"}', HTTPError(response=response))
    breakers = Mock()
    breakers.allow.return_value = True
    be = SimpleNamespace(_dependency_breakers=lambda _: breakers,
                         retry_idempotent=retry_idempotent, classify_exception=classify_exception,
                         logger=Mock(), _handle_error=Mock())
    monkeypatch.setattr(governance, "_bot_engine", lambda: be)
    state = SimpleNamespace(position_cache={"SPY": 10}, recon_halt=False, last_recon_ts=None)
    runtime = SimpleNamespace(api=SimpleNamespace(get_all_positions=Mock(side_effect=error)))
    now = datetime(2024, 1, 2, 15, tzinfo=UTC)
    assert governance.run_reconciliation_if_due(state, runtime,
        SimpleNamespace(recon_enabled=True, recon_interval_seconds=300), now) is False
    assert state.recon_halt is True and state.last_recon_ts == now
    assert breakers.record_failure.call_count >= 1


def test_cancelled_partial_fill_blocks_and_confirmed_zero_fill_passes(monkeypatch):
    from ai_trading.execution import live_trading as live
    engine = object.__new__(live.LiveTradingExecutionEngine)
    engine._cancel_order_alpaca = Mock()
    monkeypatch.setattr(live, "monotonic_time", lambda: 0)
    engine._get_order_status_alpaca = Mock(return_value={"status": "canceled", "filled_qty": "2"})
    assert engine._cancel_opposite_orders([{"id": "one"}], "SPY", "buy") == []
    engine._get_order_status_alpaca.return_value = {"status": "canceled", "filled_qty": "0"}
    assert engine._cancel_opposite_orders([{"id": "one"}], "SPY", "buy") == ["one"]


def test_failed_approval_never_records_success_audit(tmp_path, monkeypatch):
    from ai_trading.governance.promotion import ModelPromotion
    promotion = ModelPromotion(model_registry=Mock(), base_path=str(tmp_path))
    monkeypatch.setattr(promotion, "_append_jsonl_event", lambda **kw: None)
    audit = Mock()
    monkeypatch.setattr(promotion, "_append_governance_audit_event", audit)
    with pytest.raises(OSError):
        promotion.record_promotion_approval(strategy="s", model_id="m", approver="operator", decision="rejected")
    audit.assert_not_called()


def test_shadow_identity_and_source_time_survive_restart(tmp_path):
    from ai_trading.governance.promotion import ModelPromotion
    promotion = ModelPromotion(model_registry=Mock(), base_path=str(tmp_path))
    payload = {"session_id": "first", "source_start": "2024-01-02T14:00:00Z",
               "source_end": "2024-01-02T14:30:00Z", "trade_count": 20}
    for _ in range(5):
        promotion.update_shadow_metrics("m", payload)
    other = ModelPromotion(model_registry=Mock(), base_path=str(tmp_path))
    other.update_shadow_metrics("m", payload)
    result = other._load_shadow_metrics("m")
    assert result.sessions_completed == 1 and result.total_trades == 20
    assert result.last_updated == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="identity conflict"):
        other.update_shadow_metrics("m", {**payload, "trade_count": 30})
    with pytest.raises(ValueError, match="overlap"):
        other.update_shadow_metrics("m", {**payload, "session_id": "second"})
    with pytest.raises(ValueError, match="session_id"):
        other.update_shadow_metrics("m", {"trade_count": 20})


def test_concurrent_shadow_retry_counts_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from ai_trading.governance.promotion import ModelPromotion
    payload = {"session_id": "one", "source_start": "2024-01-02T14:00:00Z",
               "source_end": "2024-01-02T14:30:00Z", "trade_count": 20}
    instances = [ModelPromotion(model_registry=Mock(), base_path=str(tmp_path)) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda p: p.update_shadow_metrics("m", payload), instances))
    metrics = instances[0]._load_shadow_metrics("m")
    assert metrics.sessions_completed == 1 and metrics.total_trades == 20


def test_repeated_shadow_start_preserves_evidence_and_status(tmp_path):
    from ai_trading.governance.promotion import ModelPromotion
    registry = Mock()
    registry.model_index = {'m': {'strategy': 'test'}}
    registry.get_shadow_models.return_value = []
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path))
    assert promotion.start_shadow_testing('m') is True
    payload = {'session_id': 'one', 'source_start': '2024-01-02T14:00:00Z',
               'source_end': '2024-01-02T14:30:00Z', 'trade_count': 20}
    promotion.update_shadow_metrics('m', payload)
    registry.update_governance_status.reset_mock()
    before = (tmp_path / 'm_shadow_metrics.json').read_bytes()
    assert promotion.start_shadow_testing('m') is True
    assert (tmp_path / 'm_shadow_metrics.json').read_bytes() == before
    registry.update_governance_status.assert_not_called()


def test_shadow_start_waits_for_update_lock(tmp_path, monkeypatch):
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, get_ident
    from ai_trading.governance.promotion import ModelPromotion, PromotionMetrics
    registry = Mock()
    registry.model_index = {'m': {'strategy': 'test'}}
    registry.get_shadow_models.return_value = []
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path))
    entered = Event()
    main_thread = get_ident()
    original_flock = fcntl.flock

    def observed_flock(fd, operation):
        if get_ident() != main_thread and operation == fcntl.LOCK_EX:
            entered.set()
        return original_flock(fd, operation)

    monkeypatch.setattr(fcntl, 'flock', observed_flock)

    def start():
        return promotion.start_shadow_testing('m')

    with ThreadPoolExecutor(max_workers=1) as pool:
        with (tmp_path / 'm_shadow_metrics.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            future = pool.submit(start)
            assert entered.wait(5)
            try:
                assert not future.done()
                promotion._save_shadow_metrics('m', PromotionMetrics(sessions_completed=2))
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        assert future.result(timeout=5) is True
    assert promotion._load_shadow_metrics('m').sessions_completed == 2
    registry.update_governance_status.assert_not_called()


def test_shadow_start_preserves_unreadable_artifact(tmp_path):
    from ai_trading.governance.promotion import ModelPromotion
    registry = Mock()
    registry.model_index = {'m': {'strategy': 'test'}}
    registry.get_shadow_models.return_value = []
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path))
    path = tmp_path / 'm_shadow_metrics.json'
    path.write_text('{broken')
    assert promotion.start_shadow_testing('m') is False
    assert path.read_text() == '{broken'
    registry.update_governance_status.assert_not_called()


@pytest.mark.parametrize('identities,expected', [
    ({}, False), ({'one': 'a' * 64}, False),
    ({'one': 'a' * 64, 'two': 'invalid'}, False),
    ({'one': 'a' * 64, 'two': 'b' * 64}, True),
])
def test_eligibility_requires_identity_for_every_session(tmp_path, identities, expected):
    from ai_trading.governance.promotion import ModelPromotion, PromotionMetrics
    registry = Mock()
    registry.load_model.return_value = (None, {})
    promotion = ModelPromotion(model_registry=registry, base_path=str(tmp_path))
    promotion._save_shadow_metrics('m', PromotionMetrics(
        sessions_completed=2, observation_hashes=identities, last_updated=datetime.now(UTC)))
    eligible, details = promotion.check_promotion_eligibility('m')
    assert details['checks']['shadow_observation_provenance'] is expected
    if not expected:
        assert eligible is False
