import logging
import os
from types import SimpleNamespace
import sys

os.environ.setdefault('MAX_DRAWDOWN_THRESHOLD', '0.2')
import ai_trading.portfolio.core as core


def test_log_portfolio_summary_uses_ledger_when_broker_empty(monkeypatch, caplog):
    ctx = SimpleNamespace(
        api=SimpleNamespace(
            get_account=lambda: SimpleNamespace(cash=1000, equity=2000),
            list_positions=lambda: [],
        ),
        risk_engine=SimpleNamespace(_positions={'AAPL': 10}, _adaptive_global_cap=lambda: 0.0),
    )
    class _Pandas:
        class errors(Exception):
            class EmptyDataError(Exception):
                pass
    sys.modules['pandas'] = _Pandas
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(core, 'get_latest_price', lambda ctx, symbol: 100.0)
    core.log_portfolio_summary(ctx)
    assert any('ledger' in record.getMessage() for record in caplog.records)
    assert any('exposure=50.00%' in record.getMessage() for record in caplog.records)

