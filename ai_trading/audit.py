import csv
import csv
import logging
from pathlib import Path

# AI-AGENT-REF: minimal audit logger stub for tests
logger = logging.getLogger(__name__)
TRADE_LOG_FILE = "trades.csv"


def log_trade(symbol, qty, side, fill_price, status="filled", extra_info="", timestamp=""):
    path = Path(TRADE_LOG_FILE)
    headers = [
        "symbol", "entry_time", "entry_price", "exit_time", "exit_price",
        "qty", "side", "strategy", "classification", "signal_tags",
        "confidence", "reward",
    ]
    exists = path.exists()
    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "symbol": symbol,
                "entry_time": timestamp,
                "entry_price": fill_price,
                "exit_time": "",
                "exit_price": "",
                "qty": qty,
                "side": side,
                "strategy": extra_info,
                "classification": "",
                "signal_tags": "",
                "confidence": "",
                "reward": "",
            })
    except PermissionError:
        import importlib
        process_manager = importlib.import_module("ai_trading.utils.process_manager")

        process_manager.fix_file_permissions(path)
        logger.warning("ProcessManager attempted to fix permissions", extra={"path": str(path)})
