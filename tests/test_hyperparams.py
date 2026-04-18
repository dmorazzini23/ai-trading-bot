from ai_trading.defaults import load_default_json


def test_best_hyperparams_sensible():
    params = load_default_json('best_hyperparams.json')
    assert 1 <= params.get('fast_period', 0) < params.get('slow_period', 100), 'Fast period should be < slow period'
    assert params.get('signal_period', 0) > 0, 'Signal period must be positive'
