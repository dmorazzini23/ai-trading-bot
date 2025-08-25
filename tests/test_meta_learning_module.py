from tests.optdeps import require
require("numpy")
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_trading import meta_learning


def test_load_weights_missing(tmp_path, caplog):
    path = tmp_path / "w.csv"
    arr = meta_learning.load_weights(str(path), default=np.array([1.0, 2.0]))
    assert arr.tolist() == [1.0, 2.0]
    assert path.exists()


def test_update_weights(tmp_path):
    w_path = tmp_path / "w.csv"
    history = tmp_path / "hist.json"
    np.savetxt(w_path, np.array([0.1, 0.2]), delimiter=",")
    result = meta_learning.update_weights(str(w_path), np.array([0.3, 0.4]), {"m": 1}, str(history), n_history=2)
    assert result
    data = np.loadtxt(w_path, delimiter=",")
    assert list(data) == [0.3, 0.4]
    hist = json.load(open(history))
    assert hist


def test_update_weights_no_change(tmp_path):
    w_path = tmp_path / "w.csv"
    np.savetxt(w_path, np.array([0.1, 0.2]), delimiter=",")
    result = meta_learning.update_weights(str(w_path), np.array([0.1, 0.2]), {"m": 1})
    assert not result


def test_load_weights_corrupted(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("bad,data")
    arr = meta_learning.load_weights(str(p), default=np.array([0.5]))
    assert arr.tolist() == [0.5]


def test_update_weights_history_error(tmp_path):
    w = tmp_path / "w.csv"
    h = tmp_path / "hist.json"
    np.savetxt(w, np.array([0.1]), delimiter=",")
    h.write_text("{bad json")
    assert meta_learning.update_weights(str(w), np.array([0.2]), {"m": 1}, str(h))
    assert json.loads(h.read_text())
