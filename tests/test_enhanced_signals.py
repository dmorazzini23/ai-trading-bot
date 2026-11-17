import numpy as np
import pytest
pd = pytest.importorskip("pandas")
try:
    import ai_trading.risk.engine as risk_engine  # AI-AGENT-REF: normalized import
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("risk_engine not available", allow_module_level=True)
from ai_trading import signals














def test_prepare_indicators_cache_filename_uses_data_fingerprint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_macd(frame):
        out = frame.copy()
        out["macd"] = out["close"]
        out["signal"] = out["close"] * 0
        out["histogram"] = out["close"]
        return out

    monkeypatch.setattr(signals, "_apply_macd", fake_macd)
    monkeypatch.setattr(signals, "_apply_psar", lambda frame: frame)
    monkeypatch.setattr(pd, "read_parquet", lambda *_a, **_k: pytest.fail("stale cache read"))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, **_k: path.touch())

    def frame(offset):
        close = np.arange(1, 8, dtype=float) + offset
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.full(len(close), 100),
            }
        )

    signals.prepare_indicators(frame(0), "AAPL")
    signals.prepare_indicators(frame(10), "AAPL")

    cache_files = sorted(tmp_path.glob("cache_AAPL_*.parquet"))
    assert len(cache_files) == 2




