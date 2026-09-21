import json
from pathlib import Path

import pytest

from ai_trading.execution import live_trading as lt


@pytest.fixture
def engine(monkeypatch, tmp_path):
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    monkeypatch.setattr(engine, '_edge_realism_state_path', lambda: tmp_path / 'edge.json')
    monkeypatch.setattr(engine, '_edge_realism_bucket_context', lambda **kwargs: {'bucket_key': 'aapl:buy:regular'})
    monkeypatch.setattr(lt, 'monotonic_time', lambda: 100.)
    monkeypatch.setattr(lt, '_resolve_bool_env', lambda key: True)
    monkeypatch.setattr(lt, '_config_float', lambda key, default: default)
    monkeypatch.setattr(lt, '_config_int', lambda key, default: default)
    engine._edge_realism_state = engine._default_edge_realism_state()
    return engine


def test_feedback_roundtrip_preserves_counts_and_running_means(engine):
    for expected, realized in [(10., 5.), (20., 20.)]:
        engine._update_edge_realism_feedback(symbol='AAPL', side='buy', expected_net_edge_bps=expected,
                                             session_regime='regular', market_regime='normal', volatility_regime='normal',
                                             realized_net_edge_bps=realized)
    engine._persist_edge_realism_state(force=True)
    assert engine._edge_realism_updates_since_persist == 0
    engine._edge_realism_state = {}
    engine._load_edge_realism_state()
    summary = engine._edge_realism_state['global']
    assert summary['samples'] == 2
    assert summary['mean_realized_net_edge_bps'] == 12.5
    assert summary['mean_expected_net_edge_bps'] == 15.
    assert summary['mean_realized_to_expected_ratio'] == .75
    assert engine._edge_realism_state['buckets']['aapl:buy:regular']['samples'] == 2
    assert not engine._edge_realism_state_path().with_suffix('.json.tmp').exists()


@pytest.mark.parametrize('raw', ['broken', '[]', '{}', '{"buckets":{"":{},"x":null,"y":{"samples":0}}}'])
def test_invalid_persisted_buckets_do_not_become_evidence(engine, raw):
    engine._edge_realism_state_path().write_text(raw)
    engine._load_edge_realism_state()
    assert engine._edge_realism_state['global']['samples'] == 0
    assert engine._edge_realism_state['buckets'] == {}


def test_persist_throttle_and_disabled_write_preserve_disk(engine, monkeypatch):
    engine._edge_realism_last_persist_mono = 99.
    engine._edge_realism_updates_since_persist = 1
    engine._persist_edge_realism_state()
    assert not engine._edge_realism_state_path().exists()
    monkeypatch.setattr(lt, '_resolve_bool_env', lambda key: False)
    engine._persist_edge_realism_state(force=True)
    assert not engine._edge_realism_state_path().exists()


def test_write_failure_preserves_prior_state_and_pending_updates(engine, monkeypatch):
    path = engine._edge_realism_state_path()
    path.write_text(json.dumps({'sentinel': 'original'}))
    engine._edge_realism_updates_since_persist = 10
    def fail(*args, **kwargs):
        raise PermissionError('read only')
    monkeypatch.setattr(Path, 'replace', fail)
    engine._persist_edge_realism_state(force=True)
    assert json.loads(path.read_text()) == {'sentinel': 'original'}
    assert engine._edge_realism_updates_since_persist == 10
