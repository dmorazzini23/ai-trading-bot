import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logger


def test_get_logger_singleton(tmp_path):
    lg1 = logger.get_logger('test')
    lg2 = logger.get_logger('test')
    assert lg1 is lg2
    assert lg1.handlers
