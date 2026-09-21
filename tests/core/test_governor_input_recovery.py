import json
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core import bot_engine as bot
from ai_trading.tools import runtime_performance_report as report


@pytest.mark.parametrize('kind', ['csv', 'parquet', 'pickle', 'jsonl'])
def test_governor_recovers_supported_local_formats(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(report, '_load_trade_rows', lambda path: None)
    records = [{'symbol': 'AAPL', 'qty': 2, 'pnl': -3.}]
    path = tmp_path / f'trades.{kind}'
    if kind == 'jsonl':
        path.write_text('\n' + json.dumps(records[0]) + '\n[]\n')
    else:
        frame = pd.DataFrame(records)
        if kind == 'csv':
            frame.to_csv(path, index=False)
        elif kind == 'parquet':
            frame.to_parquet(path)
        else:
            frame.to_pickle(path)
    assert bot._profitability_governor_load_rows(path) == records


@pytest.mark.parametrize('suffix', ['jsonl', 'csv', 'parquet', 'pickle', 'other'])
def test_corrupt_governor_inputs_return_no_trade_evidence(monkeypatch, tmp_path, suffix):
    monkeypatch.setattr(report, '_load_trade_rows', lambda path: None)
    path = tmp_path / f'trades.{suffix}'
    path.write_text('{unreadable\x00')
    if suffix == 'csv':
        monkeypatch.setattr(pd, 'read_csv', lambda *a, **k: (_ for _ in ()).throw(ValueError('malformed')))
    assert bot._profitability_governor_load_rows(path) == []
    assert bot._profitability_governor_load_rows(tmp_path / 'absent') == []


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
