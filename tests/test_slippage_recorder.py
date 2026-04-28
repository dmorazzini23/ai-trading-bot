from pathlib import Path
import csv
import shutil

from ai_trading.slippage import log_slippage


def test_slippage_log_file_created():
    log_file = Path("logs/slippage.csv")
    if log_file.parent.exists():
        shutil.rmtree(log_file.parent)
    log_slippage("AAPL", 100.0, 100.0, log_file)
    assert log_file.exists()
    assert log_file.parent.is_dir()


def test_slippage_log_side_normalizes_sell_adverse_price(tmp_path):
    log_file = tmp_path / "slippage.csv"

    log_slippage("AAPL", 100.0, 99.0, log_file, side="sell")
    log_slippage("AAPL", 100.0, 101.0, log_file, side="sell")

    rows = list(csv.DictReader(log_file.open(newline="")))
    assert float(rows[0]["slippage_bps"]) == 100.0
    assert float(rows[1]["slippage_bps"]) == -100.0
