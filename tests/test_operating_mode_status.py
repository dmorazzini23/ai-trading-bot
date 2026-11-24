from types import SimpleNamespace

import pytest

from ai_trading import health_payload as health
from ai_trading.config import management


@pytest.mark.parametrize('mode,paper,url,profile,enabled,expected', [
    ('paper', True, 'https://paper-api.alpaca.markets', 'paper_trade', True, 'paper_rule_sampling'),
    ('paper', True, 'https://paper-api.alpaca.markets', 'paper_trade', False, 'close_only'),
    ('live', False, 'https://api.alpaca.markets', 'live_canary', True, 'close_only'),
    ('paper', True, 'https://api.alpaca.markets', 'paper_trade', True, 'close_only'),
])
def test_strategy_state_preserves_authority_boundaries(monkeypatch, mode, paper, url, profile, enabled, expected):
    monkeypatch.setattr(management, 'get_trading_config', lambda: SimpleNamespace(
        paper_sampling_enabled=enabled, execution_mode=mode, paper=paper,
        alpaca_base_url=url, launch_profile=profile))
    result = health._operating_mode_snapshot()
    assert result['mode'] == expected
    assert result['strategy_state'] == 'no_qualified_strategy'
    assert result['qualified_strategy'] is False
    assert result['rule_paper_sampling_enabled'] is (expected == 'paper_rule_sampling')
    assert result['results_are_strategy_performance'] is False
    assert result['promotion_authority'] is False


def test_configuration_failure_reports_close_only(monkeypatch):
    def unavailable():
        raise ValueError('invalid configuration')
    monkeypatch.setattr(management, 'get_trading_config', unavailable)
    result = health._operating_mode_snapshot()
    assert result['mode'] == 'close_only'
    assert result['rule_paper_sampling_enabled'] is False
