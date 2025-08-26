import csv
import logging
import os
import uuid
import json
try:
    from ai_trading.validation.validate_env import Settings
    settings = Settings()
except (json.JSONDecodeError, ValueError, OSError, KeyError, TypeError):
    settings = None
from ai_trading.config import management as config
from ai_trading.config.management import TradingConfig
CONFIG = TradingConfig()
TRADE_LOG_FILE = config.TRADE_LOG_FILE
logger = logging.getLogger(__name__)
_disable_trade_log = False
_fields = ['symbol', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'qty', 'side', 'strategy', 'classification', 'signal_tags', 'confidence', 'reward']
_simple_fields = ['id', 'timestamp', 'symbol', 'side', 'qty', 'price', 'exposure', 'mode', 'result']

def log_trade(symbol, qty, side, fill_price, timestamp, extra_info=None, exposure=None):
    """Persist a trade event to ``TRADE_LOG_FILE`` and log a summary."""
    global _disable_trade_log
    if isinstance(side, int | float) and isinstance(qty, str):
        logger.warning('Parameter order correction: swapping qty and side parameters')
        qty, side = (side, qty)
    if not symbol or not isinstance(symbol, str):
        logger.error('Invalid symbol provided: %s', symbol)
        return
    if not side or not isinstance(side, str):
        logger.error('Invalid side provided: %s', side)
        return
    if not isinstance(qty, int | float) or qty == 0:
        logger.error('Invalid quantity: %s', qty)
        return
    if not isinstance(fill_price, int | float) or fill_price <= 0:
        logger.error('Invalid fill_price: %s', fill_price)
        return
    if _disable_trade_log:
        return
    use_simple_format = extra_info and ('TEST' in str(extra_info).upper() or 'AUDIT' in str(extra_info).upper())
    logger.info('Trade Log | symbol=%s, qty=%s, side=%s, fill_price=%.2f, exposure=%s, timestamp=%s', symbol, qty, side, fill_price, f'{exposure:.4f}' if exposure is not None else 'n/a', timestamp)
    log_dir = os.path.dirname(TRADE_LOG_FILE) or '.'
    try:
        os.makedirs(log_dir, mode=493, exist_ok=True)
        if not os.access(log_dir, os.W_OK):
            logger.warning('Trade log directory %s is not writable', log_dir)
    except OSError as e:
        logger.error('Failed to create trade log directory %s: %s', log_dir, e)
        if not _disable_trade_log:
            _disable_trade_log = True
            logger.warning('Trade log disabled due to directory creation failure')
        return
    file_existed = os.path.exists(TRADE_LOG_FILE)
    if not file_existed:
        try:
            with open(TRADE_LOG_FILE, 'a', newline=''):
                pass
            os.chmod(TRADE_LOG_FILE, 436)
        except (OSError, PermissionError) as e:
            logger.error('Failed to create trade log file %s: %s', TRADE_LOG_FILE, e)
            if not _disable_trade_log:
                _disable_trade_log = True
                logger.warning('Trade log disabled due to file creation failure')
            return
    try:
        fields_to_use = _simple_fields if use_simple_format else _fields
        file_is_empty = not file_existed or os.path.getsize(TRADE_LOG_FILE) == 0
        with open(TRADE_LOG_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields_to_use, quoting=csv.QUOTE_MINIMAL)
            if file_is_empty:
                writer.writeheader()
            if use_simple_format:
                writer.writerow({'id': str(uuid.uuid4()), 'timestamp': timestamp, 'symbol': symbol, 'side': side, 'qty': str(qty), 'price': str(fill_price), 'exposure': str(exposure) if exposure is not None else '', 'mode': extra_info or '', 'result': ''})
            else:
                writer.writerow({'symbol': symbol, 'entry_time': timestamp, 'entry_price': fill_price, 'exit_time': '', 'exit_price': '', 'qty': qty, 'side': side, 'strategy': extra_info or '', 'classification': '', 'signal_tags': '', 'confidence': '', 'reward': ''})
    except PermissionError as exc:
        logger.error('ERROR [audit] permission denied writing %s: %s', TRADE_LOG_FILE, exc)
        try:
            from ai_trading.process_manager import ProcessManager
            pm = ProcessManager()
            repair_result = pm.fix_file_permissions([TRADE_LOG_FILE])
            if repair_result['paths_fixed']:
                logger.info('Successfully repaired file permissions, retrying trade log')
                try:
                    with open(TRADE_LOG_FILE, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=_fields, quoting=csv.QUOTE_MINIMAL)
                        if not file_existed:
                            writer.writeheader()
                        writer.writerow({'symbol': symbol, 'entry_time': timestamp, 'entry_price': fill_price, 'exit_time': '', 'exit_price': '', 'qty': qty, 'side': side, 'strategy': extra_info or '', 'classification': '', 'signal_tags': '', 'confidence': '', 'reward': ''})
                    logger.info('Trade log successfully written after permission repair')
                    return
                except (json.JSONDecodeError, ValueError, OSError, PermissionError, KeyError, TypeError) as retry_exc:
                    logger.error('Trade log retry failed after permission repair: %s', retry_exc)
            else:
                logger.warning('Failed to repair file permissions automatically')
        except (json.JSONDecodeError, ValueError, OSError, PermissionError, KeyError, TypeError) as repair_exc:
            logger.warning('Permission repair attempt failed: %s', repair_exc)
        if not _disable_trade_log:
            _disable_trade_log = True
            logger.warning('Trade log disabled due to permission error')
    except (json.JSONDecodeError, ValueError, OSError, PermissionError, KeyError, TypeError) as exc:
        logger.error('Failed to record trade: %s', exc)

def log_json_audit(details: dict) -> None:
    """Write detailed trade audit record to JSON file."""
    os.makedirs(config.TRADE_AUDIT_DIR, exist_ok=True)
    order_id = details.get('client_order_id') or str(uuid.uuid4())
    fname = os.path.join(config.TRADE_AUDIT_DIR, f'{order_id}.json')
    try:
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, default=str)
    except (json.JSONDecodeError, ValueError, OSError, PermissionError, KeyError, TypeError) as exc:
        logger.warning('Failed JSON audit log %s: %s', fname, exc)
