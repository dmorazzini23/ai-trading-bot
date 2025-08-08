from ai_trading.rebalancer import apply_no_trade_bands

def test_no_trade_bands_suppresses_small_moves():
    cur = {"AAPL": 0.20, "MSFT": 0.20}
    tgt = {"AAPL": 0.2019, "MSFT": 0.1981}  # deltas ~19 bps
    out = apply_no_trade_bands(cur, tgt, band_bps=25.0)
    assert out["AAPL"] == cur["AAPL"]
    assert out["MSFT"] == cur["MSFT"]

def test_no_trade_bands_allows_large_moves():
    cur = {"AAPL": 0.20, "MSFT": 0.20}
    tgt = {"AAPL": 0.25, "MSFT": 0.15}  # deltas 50 bps
    out = apply_no_trade_bands(cur, tgt, band_bps=25.0)
    assert out["AAPL"] == tgt["AAPL"]
    assert out["MSFT"] == tgt["MSFT"]

def test_no_trade_bands_handles_missing_symbols():
    cur = {"AAPL": 0.20}
    tgt = {"AAPL": 0.2019, "MSFT": 0.10}  # MSFT not in current
    out = apply_no_trade_bands(cur, tgt, band_bps=25.0)
    assert out["AAPL"] == cur["AAPL"]  # Small move suppressed
    assert out["MSFT"] == tgt["MSFT"]  # Large move from 0 allowed

def test_no_trade_bands_custom_threshold():
    cur = {"AAPL": 0.20, "MSFT": 0.20}
    tgt = {"AAPL": 0.2030, "MSFT": 0.1970}  # deltas 30 bps
    
    # With 25 bps threshold, moves should be allowed
    out = apply_no_trade_bands(cur, tgt, band_bps=25.0)
    assert out["AAPL"] == tgt["AAPL"]
    assert out["MSFT"] == tgt["MSFT"]
    
    # With 50 bps threshold, moves should be suppressed
    out = apply_no_trade_bands(cur, tgt, band_bps=50.0)
    assert out["AAPL"] == cur["AAPL"]
    assert out["MSFT"] == cur["MSFT"]