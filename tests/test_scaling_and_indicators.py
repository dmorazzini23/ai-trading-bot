import pandas as pd
from indicators import calculate_macd, calculate_atr, calculate_vwap
from risk_engine import calculate_atr_stop, calculate_bollinger_stop
from capital_scaling import volatility_parity_position, dynamic_fractional_kelly
from meta_learning import volatility_regime_filter
from trade_logic import pyramiding_logic


def test_macd_runs():
    close = pd.Series([1,2,3,4,5,6,7,8,9,10])
    macd, signal = calculate_macd(close)
    assert len(macd) == len(close)
    assert len(signal) == len(close)


def test_atr_behavior():
    high = pd.Series([10,11,12,13,14])
    low = pd.Series([5,6,7,8,9])
    close = pd.Series([7,8,9,10,11])
    atr = calculate_atr(high, low, close)
    assert not atr.isnull().all()


def test_vwap_calculation_basic():
    high = pd.Series([10,11,12])
    low = pd.Series([5,6,7])
    close = pd.Series([7,8,9])
    volume = pd.Series([1000,1100,1200])
    vwap = calculate_vwap(high, low, close, volume)
    assert vwap.iloc[-1] > 0


def test_atr_stop_adjustment_helper():
    stop_low_vol = calculate_atr_stop(100, 2)
    stop_high_vol = calculate_atr_stop(100, 10)
    assert stop_high_vol < stop_low_vol


def test_dynamic_fractional_kelly():
    base = 0.5
    adjusted = dynamic_fractional_kelly(base, drawdown=0.15, volatility_spike=True)
    assert adjusted < base


def test_pyramiding_logic_adds_helper():
    new_pos = pyramiding_logic(1, profit_in_atr=1.5, base_size=1)
    assert new_pos > 1


def test_volatility_filter_logic_helper():
    assert volatility_regime_filter(7, 100) == 'high_vol'
    assert volatility_regime_filter(3, 100) == 'low_vol'
