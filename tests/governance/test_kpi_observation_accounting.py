"""Persistent rollback counts measure observations, not evaluator calls."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from ai_trading.governance.promotion import ModelPromotion


def manager(path):
    registry = Mock()
    registry.model_index = {}
    registry.get_production_model.return_value = ('m', {})
    result = ModelPromotion(model_registry=registry, base_path=str(path))
    result.reconcile_active_production_pointer = Mock(return_value={'consistent': True})
    return result


def evaluate(p, identity=None, drawdown=.2):
    return p.evaluate_live_kpis_and_maybe_rollback(
        strategy='s', live_kpis={'max_drawdown': drawdown, 'observation_id': identity},
        allow_rollback=False)


def test_repeated_snapshot_across_restart_counts_once(tmp_path):
    for _ in range(4):
        assert evaluate(manager(tmp_path))['consecutive_breach_count'] == 1
    assert evaluate(manager(tmp_path), 'new')['consecutive_breach_count'] == 2


def test_concurrent_identical_snapshots_count_once(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: evaluate(manager(tmp_path), 'one'), range(8)))
    assert {r['consecutive_breach_count'] for r in results} == {1}


def test_concurrent_distinct_observations_are_not_lost(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: evaluate(manager(tmp_path), str(i)), range(8)))
    state = manager(tmp_path)._load_live_kpi_breach_state()['strategies']['s']
    assert state['consecutive_breach_count'] == 8
    assert len(state['observation_hashes']) == 8


def test_conflict_and_healthy_observation_keep_dedup_history(tmp_path):
    p = manager(tmp_path)
    evaluate(p, 'one')
    with pytest.raises(ValueError, match='identity conflict'):
        evaluate(p, 'one', .3)
    evaluate(p, 'healthy', .0)
    assert evaluate(p, 'one')['consecutive_breach_count'] == 0
    assert evaluate(p, 'two')['consecutive_breach_count'] == 1


def test_legacy_count_is_not_authority(tmp_path):
    p = manager(tmp_path)
    p._write_live_kpi_breach_state({'version': 1, 'strategies': {
        's': {'model_id': 'm', 'consecutive_breach_count': 99}}})
    assert evaluate(p, 'one')['consecutive_breach_count'] == 1
    assert p._load_live_kpi_breach_state()['strategies']['s']['legacy_unverified_count'] == 99


def test_write_failure_cannot_trigger_rollback(tmp_path, monkeypatch):
    import ai_trading.governance.promotion as mod
    p = manager(tmp_path)
    p.rollback_to_previous_production = Mock()
    monkeypatch.setattr(mod, 'atomic_write_text', Mock(side_effect=OSError('disk full')))
    with pytest.raises(OSError):
        p.evaluate_live_kpis_and_maybe_rollback(strategy='s', live_kpis={'max_drawdown': .2})
    p.rollback_to_previous_production.assert_not_called()
