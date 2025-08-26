import logging
from ai_trading.logging import get_logger
from ai_trading.logging_filters import SecretFilter


class _CaptureHandler(logging.Handler):
    """Capture a single log record for assertions."""

    def __init__(self):
        super().__init__()
        self.last = None

    def emit(self, record):
        self.last = record


def test_secret_filter_masks_extra():
    log = get_logger("test.secret_filter")
    handler = _CaptureHandler()
    handler.addFilter(SecretFilter())
    log.logger.addHandler(handler)
    try:
        log.info("msg", extra={"api_key": "supersecret", "value": 1})
    finally:
        log.logger.removeHandler(handler)
    assert handler.last.api_key == "***REDACTED***"
    assert handler.last.value == 1
