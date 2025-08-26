from __future__ import annotations
from dataclasses import dataclass

from ai_trading.utils.lazy_imports import load_pandas

def _safe_series(x: 'pd.Series | None', size: int, fill: float = 0.0) -> 'pd.Series':
    pd = load_pandas()
    if x is None:
        return pd.Series([fill] * size)
    x = pd.to_numeric(x, errors='coerce').fillna(fill)
    if len(x) >= size:
        return x.tail(size)
    return pd.concat([pd.Series([fill] * (size - len(x))), x], ignore_index=True)

def rsi(close: 'pd.Series', length: int = 14) -> 'pd.Series':
    pd = load_pandas()
    import numpy as np

    close = pd.to_numeric(close, errors='coerce')
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1 / length, adjust=False).mean()
    ad = dn.ewm(alpha=1 / length, adjust=False).mean()
    rs = au / (ad + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)

def atr(high: 'pd.Series', low: 'pd.Series', close: 'pd.Series', length: int = 14) -> 'pd.Series':
    pd = load_pandas()

    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')
    close = pd.to_numeric(close, errors='coerce')
    pc = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

def vwap_bias(close, high, low, volume, length: int = 20) -> 'pd.Series':
    pd = load_pandas()
    import numpy as np

    close = pd.to_numeric(close, errors='coerce')
    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')
    volume = pd.to_numeric(volume, errors='coerce')
    typical = (high + low + close) / 3.0
    pv = (typical * volume).rolling(length, min_periods=1).sum()
    vv = volume.rolling(length, min_periods=1).sum() + 1e-12
    vwap = pv / vv
    diff = (close - vwap) / (np.abs(vwap) + 1e-12)
    return np.tanh(diff)

def bollinger_position(close: 'pd.Series', length: int = 20, nstd: float = 2.0) -> 'pd.Series':
    pd = load_pandas()
    import numpy as np

    close = pd.to_numeric(close, errors='coerce')
    ma = close.rolling(length, min_periods=1).mean()
    sd = close.rolling(length, min_periods=1).std().fillna(0.0)
    upper = ma + nstd * sd
    lower = ma - nstd * sd
    rng = (upper - lower).replace(0.0, np.nan)
    pos = (close - ma) / (rng + 1e-12)
    return pos.clip(-1.0, 1.0).fillna(0.0)

def obv(close: 'pd.Series', volume: 'pd.Series') -> 'pd.Series':
    pd = load_pandas()
    import numpy as np

    close = pd.to_numeric(close, errors='coerce')
    volume = pd.to_numeric(volume, errors='coerce').fillna(0.0)
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum().fillna(0.0)

@dataclass
class FeatureConfig:
    window: int = 64

def compute_features(df: 'pd.DataFrame', cfg: FeatureConfig | None = None) -> 'np.ndarray':
    pd = load_pandas()
    import numpy as np

    if cfg is None:
        cfg = FeatureConfig()
    w = cfg.window
    df = df.copy()
    close = df.get('close', df.get('Close'))
    high = df.get('high', df.get('High', close))
    low = df.get('low', df.get('Low', close))
    vol = df.get('volume', df.get('Volume'))
    close = _safe_series(close, len(df), 0.0)
    rets = close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0).tail(w)
    rsi_s = rsi(close).fillna(50.0).tail(w) / 100.0
    atr_s = atr(high, low, close).fillna(0.0).tail(w)
    vwapb = vwap_bias(close, high, low, _safe_series(vol, len(df), 0.0)).fillna(0.0).tail(w)
    bbpos = bollinger_position(close).tail(w)
    obv_s = obv(close, _safe_series(vol, len(df), 0.0))
    obv_s = (
        (obv_s - obv_s.rolling(w, min_periods=1).mean())
        / (obv_s.rolling(w, min_periods=1).std() + 1e-08)
    ).fillna(0.0).clip(-1.0, 1.0).tail(w)
    parts = [rets, rsi_s, atr_s, vwapb, bbpos, obv_s]
    vec = np.concatenate([p.to_numpy(dtype=np.float32) for p in parts], axis=0)
    return vec
