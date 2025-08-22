import numpy as np
import pandas as pd
from ai_trading import signals


class _DummyProba:
    # AI-AGENT-REF: stub model with predict_proba
    def predict_proba(self, X):
        n = len(X)
        p1 = np.linspace(0.0, 1.0, n)
        return np.c_[1.0 - p1, p1]


def test_score_candidates_predict_proba():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=["r1", "r2", "r3"])
    out = signals.score_candidates(X, _DummyProba())
    assert "score" in out.columns
    assert out["score"].between(0, 1).all()

