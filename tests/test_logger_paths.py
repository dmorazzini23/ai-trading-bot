import logging

import ai_trading.logging as base_logger
import ai_trading.logging.setup as log_setup


def reset_logging_state():
    base_logger._configured = False
    base_logger._LOGGING_CONFIGURED = False
    base_logger._listener = None
    logging.getLogger().handlers.clear()


def test_logger_paths_tracks_files(tmp_path):
    reset_logging_state()
    try:
        log_setup.setup_logging(log_file=str(tmp_path / "test.log"))
        paths = log_setup.get_logger_paths()
        assert str(tmp_path / "test.log") in paths

        # Subsequent calls should not duplicate the path
        log_setup.setup_logging(log_file=str(tmp_path / "test.log"))
        assert paths == log_setup.get_logger_paths()
    finally:
        reset_logging_state()
