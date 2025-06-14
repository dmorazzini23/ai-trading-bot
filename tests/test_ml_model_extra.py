import sys
from pathlib import Path
import numpy as np
import pandas as pd

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_model import MLModel


class DummyPipe:
    def __init__(self):
        self.fitted = False
        self.version = '1'

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError('not fitted')
        return np.ones(len(X))


def make_df():
    return pd.DataFrame({'a':[1.0,2.0]})


def test_validate_errors():
    model = MLModel(DummyPipe())
    with pytest.raises(TypeError):
        model.predict([1,2])
    df = make_df()
    df.loc[0,'a'] = np.nan
    with pytest.raises(ValueError):
        model.predict(df)


def test_fit_and_predict(tmp_path):
    model = MLModel(DummyPipe())
    df = make_df()
    mse = model.fit(df, np.array([0,1]))
    assert mse >= 0
    preds = model.predict(df)
    assert len(preds) == len(df)
    save_path = model.save(tmp_path/'m.pkl')
    assert Path(save_path).exists()
    loaded = MLModel.load(save_path)
    assert isinstance(loaded.pipeline, DummyPipe)
