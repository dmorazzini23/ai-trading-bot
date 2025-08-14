import logging

from ai_trading.logging import get_logger


class _CaptureHandler(logging.Handler):
    """Capture the last log record for assertions."""  # AI-AGENT-REF: test handler

    def __init__(self):
        super().__init__()
        self.last = None

    def emit(self, record):  # noqa: D401 - simple capture
        self.last = record


def test_reserved_keys_are_prefixed():
    """Reserved LogRecord attributes should be prefixed with ``x_``."""
    log = get_logger("test.sanitize")
    handler = _CaptureHandler()
    log.logger.addHandler(handler)
    try:
        log.info(
            "probe",
            extra={
                "module": "core",
                "filename": "x.py",
                "lineno": 321,
                "ok": True,
            },
        )
    finally:
        log.logger.removeHandler(handler)

    record = handler.last
    assert record.x_module == "core"
    assert record.x_filename == "x.py"
    assert record.x_lineno == 321
    assert record.ok is True
