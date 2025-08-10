import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_trading.logging as logger  # Use centralized logging module


def test_get_logger_singleton(tmp_path):
    lg1 = logger.get_logger("test")
    lg2 = logger.get_logger("test")
    assert lg1 is lg2
    # Updated test: With our new design, child loggers use propagation 
    # instead of having their own handlers to prevent duplicates
    assert lg1.propagate  # Should propagate to root logger
    # Root logger should have handlers
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 0
