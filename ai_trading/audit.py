import csv
from pathlib import Path

# AI-AGENT-REF: minimal audit logger stub for tests
TRADE_LOG_FILE = "trades.csv"


def log_trade(symbol, qty, side, fill_price, status="filled", extra_info="", timestamp=""):
    path = Path(TRADE_LOG_FILE)
    headers = [
        "symbol", "entry_time", "entry_price", "exit_time", "exit_price",
        "qty", "side", "strategy", "classification", "signal_tags",
        "confidence", "reward",
    ]
    exists = path.exists()
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
