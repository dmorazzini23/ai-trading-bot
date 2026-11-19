from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine as bot


def test_sleeve_statistics_use_recent_cost_evidence_and_drawdown():
    records = [{'sleeve': 'day', 'is_bps': v} for v in [100, -20, 30, -10]]
    records += [{'sleeve': 'other', 'is_bps': -100}, {'sleeve': 'day', 'is_bps': 'bad'}]
    values = bot._build_sleeve_perf_states_from_tca(records=records,
        sleeves=[SimpleNamespace(name='day'), SimpleNamespace(name='empty')], expectancy_window=3)
    assert values['day'].rolling_expectancy == pytest.approx(0.)
    assert values['day'].drawdown == pytest.approx(.003)
    assert values['day'].trade_count == 3
    assert values['day'].confidence == pytest.approx(2 / 3)
    assert values['empty'].trade_count == 0

