from pathlib import Path
import shutil

from ai_trading.slippage import log_slippage


def test_slippage_log_file_created():
    log_file = Path("logs/slippage.csv")
    if log_file.parent.exists():
        shutil.rmtree(log_file.parent)
    log_slippage("AAPL", 100.0, 100.0, log_file)
    assert log_file.exists()
    assert log_file.parent.is_dir()
