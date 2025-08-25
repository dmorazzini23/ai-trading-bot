from tests.optdeps import require
require("numpy")
import numpy as np
from ai_trading.indicators import obv as obv_vec


def obv_loop(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    obv_vals = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv_vals.append(obv_vals[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv_vals.append(obv_vals[-1] - volumes[i])
        else:
            obv_vals.append(obv_vals[-1])
    return np.asarray(obv_vals, dtype=float)


def test_obv_vectorized_matches_reference_loop():
    rng = np.random.default_rng(123)
    for n in (1, 2, 5, 10, 256, 1024):
        closes = rng.normal(size=n).cumsum()  # realistic random walk
        volumes = rng.integers(low=1, high=1000, size=n).astype(float)
        ref = obv_loop(closes, volumes)
        got = obv_vec(closes, volumes)
        assert got.shape == ref.shape
        # exact equality holds for OBV semantics
        assert np.allclose(got, ref, atol=0.0)
