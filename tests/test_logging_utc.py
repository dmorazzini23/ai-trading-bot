import json
import logging

from ai_trading.logging.json_formatter import JSONFormatter


def test_json_formatter_emits_utc_z_suffix():
    formatter = JSONFormatter()
    record = logging.makeLogRecord(
        {
            "msg": "hello",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "name": "ai_trading.test",
        }
    )
    payload = json.loads(formatter.format(record))
    timestamp = payload["ts"]
    assert timestamp.endswith("Z")
    assert "." in timestamp
