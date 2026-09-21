from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from ai_trading.core import bot_engine as bot


@pytest.mark.parametrize('column', ['signal_name', 'signal'])
def test_weight_loader_normalizes_clamps_and_caches_without_aliasing(tmp_path, monkeypatch, column):
    path = tmp_path / 'weights.csv'
    path.write_text(f'{column},weight\n Momentum ,3\nreversal,-1\nnan,inf\nbad,invalid\n')
    monkeypatch.setattr(bot, 'SIGNAL_WEIGHTS_FILE', str(path))
    manager = bot.SignalManager()
    weights = manager.load_signal_weights()
    assert weights == {'momentum': 2., 'reversal': 0.}
    weights['momentum'] = 500
    monkeypatch.setattr(bot.pd, 'read_csv', lambda *a, **k: pytest.fail('fresh cache should avoid IO'))
    assert manager.load_signal_weights() == {'momentum': 2., 'reversal': 0.}


@pytest.mark.parametrize('raw', [None, 'signal_name,weight\n', 'unexpected,columns\nx,2\n', 'signal_name\nx\n'])
def test_missing_empty_or_invalid_weights_cannot_supply_evidence(tmp_path, monkeypatch, raw):
    path = tmp_path / 'weights.csv'
    if raw is not None:
        path.write_text(raw)
    monkeypatch.setattr(bot, 'SIGNAL_WEIGHTS_FILE', str(path))
    assert bot.SignalManager().load_signal_weights() == {}


@pytest.mark.parametrize('old_column', [None, 'signal_name', 'signal', 'unexpected'])
def test_weight_update_blends_old_weights_and_preserves_canonical_schema(tmp_path, monkeypatch, old_column):
    path = tmp_path / 'weights.csv'
    if old_column:
        path.write_text(f'{old_column},weight\nwinner,0.5\nloser,0.5\n')
    monkeypatch.setattr(bot, 'SIGNAL_WEIGHTS_FILE', str(path))
    frame = pd.DataFrame([
        dict(entry_price=100., exit_price=101., signal_tags='winner', side='buy', confidence=1., exit_time=datetime.now(UTC).isoformat()),
        dict(entry_price=100., exit_price=101., signal_tags='loser', side='sell', confidence=1., exit_time=datetime.now(UTC).isoformat()),
    ])
    monkeypatch.setattr(bot, '_read_trade_log', lambda *a, **k: frame.copy())
    optimize = Mock()
    monkeypatch.setattr(bot, 'optimize_signals', optimize)
    bot.update_signal_weights()
    result = pd.read_csv(path)
    assert list(result.columns) == ['signal_name', 'weight']
    weights = result.set_index('signal_name').weight.to_dict()
    assert weights == ({'winner': .6, 'loser': .4} if old_column in {'signal_name', 'signal'}
                       else {'winner': 1., 'loser': 0.})
    optimize.assert_called_once()
