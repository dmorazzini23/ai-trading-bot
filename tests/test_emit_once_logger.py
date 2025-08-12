"""Tests for EmitOnceLogger functionality."""

import logging
from io import StringIO

import pytest

from ai_trading.logging import EmitOnceLogger


@pytest.fixture
def logger_with_capture():
    """Create a logger that captures output to a string."""
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    logger = logging.getLogger('test_emit_once')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear any existing handlers
    logger.addHandler(ch)
    
    return logger, log_capture_string


def test_emit_once_info_prevents_duplicates(logger_with_capture):
    """Test that EmitOnceLogger prevents duplicate info messages."""
    logger, log_capture = logger_with_capture
    emit_once = EmitOnceLogger(logger)
    
    # First call should emit
    emit_once.info("Test message")
    output1 = log_capture.getvalue()
    assert "Test message" in output1
    
    # Second call with same message should not emit
    emit_once.info("Test message")
    output2 = log_capture.getvalue()
    assert output1 == output2  # No new content added
    
    # Different message should emit
    emit_once.info("Different message")
    output3 = log_capture.getvalue()
    assert "Different message" in output3
    assert len(output3) > len(output2)


def test_emit_once_with_custom_key(logger_with_capture):
    """Test that EmitOnceLogger works with custom keys."""
    logger, log_capture = logger_with_capture
    emit_once = EmitOnceLogger(logger)
    
    # First call with custom key
    emit_once.info("Message 1", key="startup_banner")
    output1 = log_capture.getvalue()
    assert "Message 1" in output1
    
    # Second call with same key but different message should not emit
    emit_once.info("Message 2", key="startup_banner")
    output2 = log_capture.getvalue()
    assert output1 == output2  # No new content added
    assert "Message 2" not in output2
    
    # Call with different key should emit
    emit_once.info("Message 3", key="different_key")
    output3 = log_capture.getvalue()
    assert "Message 3" in output3


def test_emit_once_different_levels(logger_with_capture):
    """Test that EmitOnceLogger works across different log levels."""
    logger, log_capture = logger_with_capture
    emit_once = EmitOnceLogger(logger)
    
    # Test each level
    emit_once.debug("Debug message")
    emit_once.info("Info message")
    emit_once.warning("Warning message")
    emit_once.error("Error message")
    
    output = log_capture.getvalue()
    assert "DEBUG - Debug message" in output
    assert "INFO - Info message" in output
    assert "WARNING - Warning message" in output
    assert "ERROR - Error message" in output
    
    # Try duplicates - should not appear again
    original_length = len(output)
    emit_once.debug("Debug message")
    emit_once.info("Info message")
    emit_once.warning("Warning message") 
    emit_once.error("Error message")
    
    final_output = log_capture.getvalue()
    assert len(final_output) == original_length  # No new content


def test_emit_once_thread_safe():
    """Test that EmitOnceLogger is thread-safe."""
    import threading
    import time
    
    logger = logging.getLogger('test_thread_safe')
    logger.handlers.clear()
    emit_once = EmitOnceLogger(logger)
    
    results = []
    
    def emit_messages():
        for i in range(100):
            # This should only be emitted once total across all threads
            emit_once.info("Thread safe test message")
            time.sleep(0.001)  # Small delay to encourage race conditions
    
    # Start multiple threads
    threads = [threading.Thread(target=emit_messages) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # The key should only be in the set once
    assert len(emit_once._emitted_keys) == 1
    assert "Thread safe test message" in emit_once._emitted_keys