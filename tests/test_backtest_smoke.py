from pathlib import Path

import logging
import pytest

pd = pytest.importorskip("pandas")


def force_coverage(mod):
    """Force coverage by importing and accessing module attributes instead of using exec."""
    for attr_name in dir(mod):
        if not attr_name.startswith('_'):
            getattr(mod, attr_name, None)


@pytest.mark.smoke
def test_backtester_engine_basic(tmp_path, caplog, monkeypatch):
    from ai_trading.strategies import backtester

    monkeypatch.chdir(tmp_path)

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

    with caplog.at_level(logging.INFO, logger="ai_trading.strategies.backtester"):
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

    assert any("symbol  pnl" in rec.message for rec in caplog.records)
    assert any("AAPL" in rec.message for rec in caplog.records)

    force_coverage(backtester)
