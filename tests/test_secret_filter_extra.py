import logging
from types import SimpleNamespace
from ai_trading.logging import get_logger
from ai_trading import logging_filters
from ai_trading.logging_filters import SecretFilter
from ai_trading.logging.redact import _ENV_MASK


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
    assert handler.last.api_key == _ENV_MASK
    assert handler.last.value == 1


def test_secret_filter_preserves_has_secret_boolean():
    log = get_logger("test.secret_filter.has_secret")
    handler = _CaptureHandler()
    handler.addFilter(SecretFilter())
    log.logger.addHandler(handler)
    try:
        log.info("msg", extra={"has_secret": False, "has_key": True})
    finally:
        log.logger.removeHandler(handler)
    assert handler.last.has_secret is False
    assert handler.last.has_key is True


def test_secret_filter_masks_embedded_secret_args_and_formatted_message(monkeypatch):
    secret = "live-secret-token-12345"
    monkeypatch.setitem(
        logging_filters.sys.modules,
        "ai_trading.config.management",
        SimpleNamespace(merged_env_snapshot=lambda: {"ALPACA_SECRET_KEY": secret}),
    )
    record = logging.LogRecord(
        name="test.secret_filter.embedded",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Authorization header: %s",
        args=(f"Bearer {secret}",),
        exc_info=None,
    )

    SecretFilter().filter(record)

    rendered = record.getMessage()
    assert secret not in rendered
    assert rendered == f"Authorization header: Bearer {_ENV_MASK}"


def test_secret_filter_masks_embedded_secret_dict_args(monkeypatch):
    secret = "dict-secret-token-12345"
    monkeypatch.setitem(
        logging_filters.sys.modules,
        "ai_trading.config.management",
        SimpleNamespace(merged_env_snapshot=lambda: {"AUTH_TOKEN": secret}),
    )
    record = logging.LogRecord(
        name="test.secret_filter.dict",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=%(token)s",
        args=({"token": f"prefix-{secret}-suffix"},),
        exc_info=None,
    )

    SecretFilter().filter(record)

    rendered = record.getMessage()
    assert secret not in rendered
    assert rendered == f"token=prefix-{_ENV_MASK}-suffix"
