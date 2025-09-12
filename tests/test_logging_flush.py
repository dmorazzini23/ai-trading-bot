import logging
import ai_trading.logging as base_logger
import ai_trading.logging.setup as log_setup


def reset_logging_state() -> None:
    base_logger._configured = False
    base_logger._LOGGING_CONFIGURED = False
    base_logger._listener = None
    logging.getLogger().handlers.clear()


def test_setup_logging_flushes_handlers(tmp_path, monkeypatch):
    reset_logging_state()
    try:
        monkeypatch.setenv("PYTEST_RUNNING", "1")
        log_file = tmp_path / "flush.log"
        log_setup.setup_logging(log_file=str(log_file))
        assert "Logging configured successfully - no duplicates possible" in log_file.read_text()
    finally:
        reset_logging_state()
