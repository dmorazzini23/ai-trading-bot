import numpy as np
import pytest
pd = pytest.importorskip("pandas")
try:
    import ai_trading.risk.engine as risk_engine  # AI-AGENT-REF: normalized import
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("risk_engine not available", allow_module_level=True)
from ai_trading import signals


def test_dynamic_position_size_scaling():
    s1 = risk_engine.dynamic_position_size(10000, volatility=0.02, drawdown=0.05)
    s2 = risk_engine.dynamic_position_size(10000, volatility=0.02, drawdown=0.15)
    assert s2 < s1 and s1 > 0


def test_signal_matrix_and_vote():
    close = np.linspace(100, 105, 30)
    df = pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
    })
    matrix = signals.compute_signal_matrix(df)
    assert not matrix.empty
    vote = signals.ensemble_vote_signals(matrix)
    assert len(vote) == len(matrix)


def test_signal_matrix_inverts_mean_reversion_zscore():
    close = np.array([100.0] * 25 + [120.0])
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
        },
        index=pd.date_range("2026-04-27 14:30", periods=len(close), freq="min", tz="UTC"),
    )

    matrix = signals.compute_signal_matrix(df)

    assert matrix["mean_rev_z"].iloc[-1] < 0.0


def test_signal_matrix_recomputes_same_last_bar_for_different_data():
    index = pd.date_range("2026-04-27 14:30", periods=40, freq="min", tz="UTC")
    close_a = np.linspace(100, 120, len(index))
    close_b = np.linspace(200, 160, len(index))
    frame_a = pd.DataFrame(
        {"close": close_a, "high": close_a + 1.0, "low": close_a - 1.0},
        index=index,
    )
    frame_b = pd.DataFrame(
        {"close": close_b, "high": close_b + 1.0, "low": close_b - 1.0},
        index=index,
    )

    matrix_a = signals.compute_signal_matrix(frame_a)
    matrix_b = signals.compute_signal_matrix(frame_b)

    assert not matrix_a.equals(matrix_b)


def test_hmm_regime_predictions_align_invalid_rows_to_neutral(monkeypatch):
    class DummyHMM:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _train):
            return self

        def predict(self, arr):
            return np.arange(1, len(arr) + 1)

    monkeypatch.setattr(signals, "_get_gaussian_hmm", lambda: DummyHMM)
    close = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, np.inf, 106.0])
    frame = pd.DataFrame({"close": close})

    regimes = signals.detect_market_regime_hmm(frame, n_components=2)
    valid_returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan).notna()

    assert len(regimes) == len(frame)
    assert np.all(regimes[~valid_returns.to_numpy()] == 0)
    assert np.any(regimes[valid_returns.to_numpy()] != 0)


def test_signal_decision_rejects_non_finite_values(monkeypatch):
    close = np.linspace(100, 110, 40)
    frame = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
        }
    )
    pipeline = signals.SignalDecisionPipeline({"ensemble_min_agree": 0})
    monkeypatch.setattr(signals, "robust_signal_price", lambda _df: np.inf)

    assert pipeline.evaluate_signal_with_costs("AAPL", frame, 0.1)["reason"] == "REJECT_INVALID_INPUT"

    monkeypatch.setattr(signals, "robust_signal_price", lambda _df: 100.0)
    monkeypatch.setattr(signals.SignalDecisionPipeline, "_calculate_current_atr", lambda self, df: np.nan)

    assert pipeline.evaluate_signal_with_costs("AAPL", frame, 0.1)["reason"] == "REJECT_INVALID_INPUT"

    monkeypatch.setattr(signals.SignalDecisionPipeline, "_calculate_current_atr", lambda self, df: 1.0)
    monkeypatch.setattr(
        signals.SignalDecisionPipeline,
        "_estimate_transaction_costs",
        lambda self, symbol, price, quantity: {
            "total_cost_pct": np.inf,
            "total_cost": np.inf,
            "notional": price * quantity,
        },
    )

    assert pipeline.evaluate_signal_with_costs("AAPL", frame, np.nan)["reason"] == "REJECT_INVALID_INPUT"
    assert pipeline.evaluate_signal_with_costs("AAPL", frame, 0.1, quantity=np.inf)["reason"] == "REJECT_INVALID_INPUT"
    assert pipeline.evaluate_signal_with_costs("AAPL", frame, 0.1)["reason"] == "REJECT_INVALID_INPUT"


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


def test_prepare_indicators_parallel_assigns_results_when_cache_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_PARQUET", "1")
    symbols = ["AAPL", "MSFT"]
    data = {
        sym: pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [2.0, 3.0],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [100, 100],
            }
        )
        for sym in symbols
    }

    def fake_prepare(frame, ticker=None):
        out = frame.copy()
        out["prepared_for"] = ticker
        return out

    monkeypatch.setattr(signals, "prepare_indicators", fake_prepare)

    signals.prepare_indicators_parallel(symbols, data)

    assert data["AAPL"]["prepared_for"].iloc[0] == "AAPL"
    assert data["MSFT"]["prepared_for"].iloc[0] == "MSFT"


def test_classify_regime_basic():
    df = pd.DataFrame({"close": np.linspace(100, 120, 40)})
    regime = signals.classify_regime(df)
    assert regime.iloc[-1] in {"trend", "mean_revert"}
