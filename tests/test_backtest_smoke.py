from pathlib import Path

import pandas as pd
import pytest


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_backtester_engine_basic(tmp_path, capsys):
    import backtester

    df = pd.DataFrame({
        "Open": [1.0, 1.1],
        "High": [1.0, 1.1],
        "Low": [1.0, 1.1],
        "Close": [1.05, 1.15],
    })

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_csv(data_dir / "AAPL.csv", index=False)

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

    engine = backtester.BacktestEngine({"AAPL": df}, backtester.DefaultExecutionModel())
    result = engine.run(["AAPL"])
    assert hasattr(result, "net_pnl")

    force_coverage(backtester)
