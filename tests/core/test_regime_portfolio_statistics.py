from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ai_trading.core import bot_engine as bot






@pytest.mark.parametrize('cached', [False, True])
def test_spy_volatility_uses_sufficient_history_and_caches_daily_result(monkeypatch, cached):
    stats = {'mean': None, 'std': None, 'last': None, 'last_update': None}
    monkeypatch.setattr(bot, '_VOL_STATS', stats)
    frame = pd.DataFrame({'high': [102.] * 300, 'low': [99.] * 300, 'close': [100.] * 300})
    atr = pd.Series(np.arange(300) / 100 + 1)
    monkeypatch.setattr(bot.ta, 'atr', lambda *a, **k: atr)
    fetch = Mock(return_value=frame)
    ctx = NS(data_fetcher=NS(get_daily_df=fetch, _daily_cache={bot.REGIME_SYMBOLS[0]: (0, frame)} if cached else {}))
    bot.compute_spy_vol_stats(ctx)
    assert stats['mean'] == pytest.approx(atr.iloc[-252:].mean())
    assert stats['std'] == pytest.approx(atr.iloc[-252:].std())
    assert stats['last'] == atr.iloc[-1]
    assert fetch.call_count == int(not cached)
    bot.compute_spy_vol_stats(ctx)
    assert fetch.call_count == int(not cached)


def test_spy_volatility_retry_exhaustion_halts_without_replacing_prior_statistics(monkeypatch):
    stats = {'mean': 2., 'std': .5, 'last': 3., 'last_update': None}
    monkeypatch.setattr(bot, '_VOL_STATS', stats)
    monkeypatch.setattr(bot, 'MAX_VOL_FETCH_RETRIES', 2)
    monkeypatch.setattr(bot.time, 'sleep', lambda *a: None)
    fetch = Mock(side_effect=bot.DataFetchError('temporary failure'))
    halt = Mock()
    bot.compute_spy_vol_stats(NS(data_fetcher=NS(get_daily_df=fetch), halt_manager=NS(manual_halt_trading=halt)))
    assert fetch.call_count == 2
    halt.assert_called_once_with('volatility data unavailable')
    assert stats == {'mean': 2., 'std': .5, 'last': 3., 'last_update': None}
