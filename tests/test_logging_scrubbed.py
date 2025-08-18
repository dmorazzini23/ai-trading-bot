import logging
import os

from ai_trading.logging import get_logger


def test_no_secrets_in_logs(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("NEWS_API_KEY", "TOPSECRETKEY")
    logger = get_logger(__name__)
    logger.info("boot with key=%s", os.getenv("NEWS_API_KEY"))
    joined = "\n".join(m.message for m in caplog.records)
    assert "TOPSECRETKEY" not in joined
