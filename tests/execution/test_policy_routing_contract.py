from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ai_trading.execution import engine as module, policy_selector


@pytest.mark.parametrize('policy', list(policy_selector.ExecutionPolicy))
def test_selected_policy_preserves_market_inputs_and_routing(monkeypatch, policy):
    engine = module.ExecutionEngine.__new__(module.ExecutionEngine)
    engine._policy_selector_enabled = True
    engine.logger = Mock()
    select = Mock(return_value=SimpleNamespace(policy=policy, reasons=['test'],
                                               participation_rate=.02, urgency_score=.5))
    monkeypatch.setattr(policy_selector, 'select_execution_policy', select)
    monkeypatch.setattr(module, 'get_env', lambda key, default=None: default)
    monkeypatch.setattr(module, '_env_bool', lambda key, default: default)
    result = engine._select_execution_policy(symbol='AAPL', side=module.OrderSide.BUY,
        quantity=5, limit_price=None, kwargs={'bid': 100., 'ask': 100.1, 'expected_price': 100.,
        'volume_1d': 1000, 'volatility': .01, 'urgency': .5})
    args = select.call_args.kwargs
    assert args['spread_bps'] == pytest.approx(10.)
    assert args['order_notional'] == 500.
    assert args['avg_daily_volume_notional'] == 100000.
    assert args['data_provenance'] == 'iex'
    assert result['policy'] == policy.value
    assert result['selected'] is True
    if policy == policy_selector.ExecutionPolicy.TWAP:
        assert result['use_twap'] is True


def test_disabled_selector_does_not_route():
    engine = module.ExecutionEngine.__new__(module.ExecutionEngine)
    engine._policy_selector_enabled = False
    assert engine._select_execution_policy(symbol='AAPL', side=module.OrderSide.BUY,
        quantity=5, limit_price=None, kwargs={}) == {'selected': False}
