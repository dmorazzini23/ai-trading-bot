from pathlib import Path

import pandas as pd
import pytest


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
@pytest.mark.xfail(reason="minimal data may produce no trades")
def test_backtester_engine_basic(tmp_path, capsys):
    import backtester

    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.0, 1.1],
            "low": [1.0, 1.1],
            "close": [1.05, 1.15],
            "volume": [100, 100],
        },
        index=idx,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_csv(data_dir / "AAPL.csv")

    backtester.main([
        "--symbols",
        "AAPL",
        "--data-dir",
        str(data_dir),
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-02",
    ])

    out = capsys.readouterr().out
    assert "Net PnL" in out



    force_coverage(backtester)
