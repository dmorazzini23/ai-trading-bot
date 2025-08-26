import csv
from ai_trading.logging import get_logger
from pathlib import Path
logger = get_logger(__name__)
TRADE_LOG_FILE = 'trades.csv'

def fix_file_permissions(path: str) -> bool:
    from ai_trading.utils import process_manager
    if hasattr(process_manager, 'fix_file_permissions'):
        process_manager.fix_file_permissions(path)
    else:
        return False
    return True

def log_trade(symbol, qty, side, fill_price, status='filled', extra_info='', timestamp=''):
    path = Path(TRADE_LOG_FILE)
    headers = ['symbol', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'qty', 'side', 'strategy', 'classification', 'signal_tags', 'confidence', 'reward']
    exists = path.exists()
    try:
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not exists:
                writer.writeheader()
            writer.writerow({'symbol': symbol, 'entry_time': timestamp, 'entry_price': fill_price, 'exit_time': '', 'exit_price': '', 'qty': qty, 'side': side, 'strategy': extra_info, 'classification': '', 'signal_tags': '', 'confidence': '', 'reward': ''})
    except PermissionError:
        if fix_file_permissions(path):
            logger.warning('ProcessManager attempted to fix permissions', extra={'path': str(path)})