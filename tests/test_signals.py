import numpy as np
import pandas as pd
import pytest

np.random.seed(0)

from signals import GaussianHMM, detect_market_regime_hmm


def test_hmm_regime_detection():
    if GaussianHMM is None:
        pytest.skip("hmmlearn not installed")
    df = pd.DataFrame({"Close": np.random.rand(100) + 100})
    df = detect_market_regime_hmm(df)
    assert "Regime" in df.columns
