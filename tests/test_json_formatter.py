from tests.optdeps import require
require("numpy")
import json
import logging

import ai_trading.logging as logger  # Use centralized logging module


def _make_record(**extra):
    rec = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=None,
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_json_formatter_custom_fields_and_masking():
    fmt = logger.JSONFormatter("%Y-%m-%dT%H:%M:%SZ")
    rec = _make_record(symbol="AAPL", api_key="abcdef1234", pathname="skip")
    out = fmt.format(rec)
    data = json.loads(out)
    assert set(data) >= {"ts", "level", "name", "msg", "symbol", "api_key"}
    assert data["api_key"].endswith("1234") and set(data["api_key"]) <= set("*1234")
    assert "pathname" not in data


def test_json_formatter_extra_fields_and_mask_keys():
    fmt = logger.JSONFormatter(
        "%Y-%m-%dT%H:%M:%SZ",
        extra_fields={"service": "trade", "secret": "top"},
        mask_keys=["secret", "symbol"],
    )
    rec = _make_record(symbol="AAPL")
    out = fmt.format(rec)
    data = json.loads(out)
    assert data["service"] == "trade"
    assert data["secret"] == "***"
    assert data["symbol"] == "***"


def test_json_formatter_exc_info():
    fmt = logger.JSONFormatter("%Y-%m-%dT%H:%M:%SZ")
    rec = _make_record()
    rec.exc_info = (ValueError, ValueError("boom"), None)
    out = fmt.format(rec)
    data = json.loads(out)
    assert data["exc"].startswith("ValueError: boom")
    assert "exc_info" not in data


def test_json_formatter_serializes_nonstandard_types():
    fmt = logger.JSONFormatter("%Y-%m-%dT%H:%M:%SZ")
    from datetime import UTC, date, datetime

    import numpy as np

    class Foo:
        def __str__(self):
            return "FOO"

    rec = _make_record(
        arr=np.array([1, 2, 3]),
        dt=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        d=date(2024, 1, 3),
        foo=Foo(),
    )
    out = fmt.format(rec)
    data = json.loads(out)
    assert data["arr"] == [1, 2, 3]
    assert data["dt"].startswith("2024-01-02T03:04:05")
    assert data["d"] == "2024-01-03"
    assert data["foo"] == "FOO"
