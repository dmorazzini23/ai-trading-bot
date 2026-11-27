from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from ai_trading.core import bot_engine as bot


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
