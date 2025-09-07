import csv
from pathlib import Path


def test_talib_audit_creates_data_dir_and_file(tmp_path, monkeypatch):
    """ai_trading.talib.audit.log_trade should create data/trades.csv with 664 perms."""
    monkeypatch.chdir(tmp_path)
    from ai_trading.talib import audit

    audit.log_trade(
        symbol="TEST",
        qty=1,
        side="buy",
        fill_price=1.0,
        timestamp="2024-01-01T00:00:00Z",
        extra_info="TEST_MODE",
        exposure=0.5,
    )

    log_path = tmp_path / "data" / "trades.csv"
    assert log_path.exists()
    assert log_path.parent.exists()

    mode = oct(log_path.stat().st_mode)[-3:]
    assert mode == "664", f"Expected file permissions 664, got {mode}"

    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "TEST"
    assert rows[0]["side"] == "buy"
