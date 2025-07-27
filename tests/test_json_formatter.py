import json
import logging
import logger


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
    fmt = logger.JSONFormatter("%(asctime)sZ")
    rec = _make_record(symbol="AAPL", api_key="abcdef1234", pathname="skip")
    out = fmt.format(rec)
    data = json.loads(out)
    assert set(data) >= {"ts", "level", "name", "msg", "symbol", "api_key"}
    assert data["api_key"].endswith("1234") and set(data["api_key"]) <= set("*1234")
    assert "pathname" not in data


def test_json_formatter_exc_info():
    fmt = logger.JSONFormatter("%(asctime)sZ")
    rec = _make_record()
    rec.exc_info = (ValueError, ValueError("boom"), None)
    out = fmt.format(rec)
    data = json.loads(out)
    assert data["exc"].startswith("ValueError: boom")
    assert "exc_info" not in data
