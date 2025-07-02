import types
import pandas as pd
import numpy as np


def test_trade_logic_end_to_end(monkeypatch, caplog):
    import sys, types as t
    sys.modules.setdefault('schedule', t.ModuleType('schedule'))
    sys.modules.setdefault('yfinance', t.ModuleType('yfinance'))
    import bot_engine as bot
    df = pd.DataFrame({
        'close': np.linspace(1,5,30),
        'high': np.linspace(1,5,30),
        'low': np.linspace(1,5,30),
        'volume': np.arange(30)+1,
        'macd': 0.1,
        'macds':0.1,
        'atr':0.5,
        'vwap':1.0
    })
    monkeypatch.setattr(bot, '_fetch_feature_data', lambda ctx, st, sym: (df, df, None))
    sm = bot.SignalManager()
    monkeypatch.setattr(sm, 'evaluate', lambda ctx, st, df, sym, model: (1.0, 0.8, 'test'))
    monkeypatch.setattr(bot, 'signal_manager', sm)
    monkeypatch.setattr(bot, 'pre_trade_checks', lambda *a, **k: True)
    monkeypatch.setattr(bot, '_current_position_qty', lambda *a, **k: 0)
    monkeypatch.setattr(bot, '_recent_rebalance_flag', lambda *a, **k: False)
    called = {}
    def dummy_enter(ctx, state, symbol, bal, feat_df, score, conf, strat):
        called['qty'] = int(bal * 0.01)
        return True
    monkeypatch.setattr(bot, '_enter_long', dummy_enter)
    ctx = types.SimpleNamespace(
        api=types.SimpleNamespace(get_account=lambda: types.SimpleNamespace(cash='10000', buying_power='10000')),
        data_fetcher=types.SimpleNamespace(get_daily_df=lambda c,s: df),
        trade_logger=types.SimpleNamespace(log_entry=lambda *a, **k: None),
        risk_engine=bot.RiskEngine(),
        portfolio_weights={'TST':0.01},
        stop_targets={},
        take_profit_targets={},
        rebalance_buys={},
        max_position_dollars=1e6,
        params={},
        capital_scaler=bot.CapitalScalingEngine(),
        market_open=bot.MARKET_OPEN,
        market_close=bot.MARKET_CLOSE,
    )
    state = types.SimpleNamespace(trade_cooldowns={}, last_trade_direction={}, long_positions=set(), short_positions=set(), position_cache={}, no_signal_events=0)
    caplog.set_level('INFO')
    assert bot.trade_logic(ctx, state, 'TST', 10000, model=None, regime_ok=True)
    assert called.get('qty') == 100
    assert 'PROCESSING_SYMBOL' in caplog.text

