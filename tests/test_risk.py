import types


class DummyLock:
    data = ''
    def __init__(self, *a, **k):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc, val, tb):
        return False
    def write(self, s):
        self.__class__.data = s
    def read(self):
        return self.__class__.data
    def seek(self, *a):
        pass
    def truncate(self):
        pass


def setup_ctx():
    ctx = types.SimpleNamespace(
        kelly_fraction=0.6,
        capital_scaler=types.SimpleNamespace(compression_factor=lambda b: 1.0),
        max_position_dollars=1e9,
    )
    return ctx


def manual_fractional(balance, price, atr, win_prob, peak):
    from ai_trading.core import bot_engine as bot
    base_frac = 0.6
    comp = 1.0
    base_frac *= comp
    drawdown = (peak - balance) / peak
    if drawdown > 0.10:
        frac = 0.3
    elif drawdown > 0.05:
        frac = 0.45
    else:
        frac = base_frac
    edge = win_prob - (1 - win_prob) / 1.5
    kelly = max(edge / 1.5, 0) * frac
    dollars = kelly * balance
    raw = dollars / atr
    cap_scale = frac / base_frac if base_frac else 1.0
    cap_pos = (balance * bot.CAPITAL_CAP * cap_scale) / price
    risk_cap = (balance * bot.DOLLAR_RISK_LIMIT) / atr
    dollar_cap = 1e9 / price
    size = int(round(min(raw, cap_pos, risk_cap, dollar_cap, bot.MAX_POSITION_SIZE)))
    return max(size, 1)


def test_fractional_kelly_drawdown(monkeypatch, tmp_path):
    import sys
    import types
    sys.modules.setdefault('schedule', types.ModuleType('schedule'))
    sys.modules.setdefault('yfinance', types.ModuleType('yfinance'))
    from ai_trading.core import bot_engine as bot
    ctx = setup_ctx()
    monkeypatch.setattr(bot, 'PEAK_EQUITY_FILE', tmp_path / 'p.txt')
    monkeypatch.setattr(bot, 'is_high_vol_thr_spy', lambda: False)
    monkeypatch.setattr(bot.os.path, 'exists', lambda p: False)
    monkeypatch.setattr(bot, 'portalocker', types.SimpleNamespace(
        Lock=DummyLock,
        lock=lambda f, mode: None,  # Mock lock function
        unlock=lambda f: None,     # Mock unlock function
        LOCK_EX=1                  # Mock lock constant
    ))
    
    # Mock the open function to return DummyLock instance
    import builtins
    def mock_open(filename, mode='r'):
        return DummyLock()
    monkeypatch.setattr(builtins, 'open', mock_open)
    
    size1 = bot.fractional_kelly_size(ctx, 10000, 50, 2.0, 0.6)
    DummyLock.data = str(10600)
    monkeypatch.setattr(bot.os.path, 'exists', lambda p: True)
    size2 = bot.fractional_kelly_size(ctx, 10000, 50, 2.0, 0.6)
    exp1 = manual_fractional(10000,50,2.0,0.6,10000)
    exp2 = manual_fractional(10000,50,2.0,0.6,10600)
    assert size1 == exp1
    assert size2 == exp2
    assert size2 < size1


