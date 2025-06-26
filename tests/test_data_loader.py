import types
from pathlib import Path

import pandas as pd

import backtester.data_loader as data_loader


def test_load_symbol_data_repairs_and_fetches(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bad = data_dir / "AAPL.csv"
    pd.DataFrame({"Open": [1], "High": [2]}).to_csv(bad, index=False)

    monkeypatch.setattr(data_loader, "DATA_DIR", data_dir)

    fake_df = pd.DataFrame(
        {
            "timestamp": ["2023-01-01"],
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [10],
        }
    )

    monkeypatch.setattr(data_loader, "_fetch_from_alpaca", lambda *a, **k: fake_df)

    df = data_loader.load_symbol_data("AAPL")
    assert set(df.columns) == {"timestamp", "Open", "High", "Low", "Close", "Volume"}
    saved = pd.read_csv(bad)
    assert set(saved.columns) == {"timestamp", "Open", "High", "Low", "Close", "Volume"}

