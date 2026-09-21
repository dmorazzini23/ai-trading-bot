from types import SimpleNamespace

import pytest

from ai_trading import health_payload as health
from ai_trading.config import management


@pytest.mark.parametrize('mode,paper,url,profile,enabled,reason,expected', [
    ('paper', True, 'https://paper-api.alpaca.markets', 'paper', True, 'required_model_stale', 'paper_diagnostics_only'),
    ('paper', True, 'https://paper-api.alpaca.markets', 'paper', True, 'required_model_unavailable', 'paper_diagnostics_only'),
    ('live', False, 'https://api.alpaca.markets', 'live_canary', True, 'required_model_stale', 'model_blocked'),
    ('paper', True, 'https://paper-api.alpaca.markets', 'paper', False, 'required_model_stale', 'model_blocked'),
    ('paper', True, 'https://paper-api.alpaca.markets', 'paper', True, 'required_model_invalid', 'model_blocked'),
])
def test_diagnostics_status_preserves_authority_boundaries(monkeypatch, mode, paper, url, profile, enabled, reason, expected):
    monkeypatch.setenv('AI_TRADING_PAPER_SAMPLING_STALE_MODEL_DIAGNOSTICS_ENABLED', '1')
    monkeypatch.setattr(management, 'get_trading_config', lambda: SimpleNamespace(
        paper_sampling_enabled=enabled, execution_mode=mode, paper=paper,
        alpaca_base_url=url, launch_profile=profile))
    model = {'enabled': True, 'ok': False, 'reason': reason}
    result = health._operating_mode_snapshot(model)
    assert result['mode'] == expected
    assert not result['model_ready']
    assert not result['promotion_authority']
    assert not result['diagnostic_results_are_model_performance']
    assert model == {'enabled': True, 'ok': False, 'reason': reason}


def test_ready_and_optional_model_status_never_claims_diagnostics():
    assert health._operating_mode_snapshot({'enabled': True, 'ok': True})['mode'] == 'model_ready'
    assert health._operating_mode_snapshot({'enabled': False})['mode'] == 'model_not_required'


def test_configuration_failure_keeps_model_blocked(monkeypatch):
    def unavailable():
        raise ValueError('invalid configuration')
    monkeypatch.setattr(management, 'get_trading_config', unavailable)
    result = health._operating_mode_snapshot({'enabled': True, 'ok': False, 'reason': 'required_model_stale'})
    assert result['mode'] == 'model_blocked'
    assert not result['paper_diagnostics_enabled']
