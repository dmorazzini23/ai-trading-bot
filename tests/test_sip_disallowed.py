from datetime import datetime, UTC
import json
import sys
import types


validation_stub = types.ModuleType("ai_trading.validation")
require_env_stub = types.ModuleType("ai_trading.validation.require_env")


def _require_env_vars(*_a, **_k):
    return None


def require_env_vars(*_a, **_k):  # noqa: D401
    return True


def should_halt_trading(*_a, **_k):
    return False


require_env_stub._require_env_vars = _require_env_vars
require_env_stub.require_env_vars = require_env_vars
require_env_stub.should_halt_trading = should_halt_trading
validation_stub.require_env = require_env_stub
validation_stub._require_env_vars = _require_env_vars
validation_stub.require_env_vars = require_env_vars
validation_stub.should_halt_trading = should_halt_trading
sys.modules.setdefault("ai_trading.validation", validation_stub)
sys.modules.setdefault("ai_trading.validation.require_env", require_env_stub)

import ai_trading.data.fetch as data_fetcher


class _RespOK:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    text = json.dumps({"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]})

    def json(self):
        return json.loads(self.text)


def test_sip_disallowed_falls_back_to_iex(monkeypatch, caplog):
    feeds: list[str] = []

    def fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
        feeds.append(params.get("feed"))
        return _RespOK()

    monkeypatch.setattr(data_fetcher._HTTP_SESSION, "get", fake_get)
    monkeypatch.setattr(data_fetcher, "_ALLOW_SIP", False, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_DISALLOWED_WARNED", False, raising=False)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    with caplog.at_level("WARNING"):
        df = data_fetcher.get_bars("AAPL", "1Min", start, end, feed="sip")
    assert feeds == ["iex"]
    assert not df.empty
    assert "SIP_" in caplog.text
    caplog.clear()
    data_fetcher.get_bars("AAPL", "1Min", start, end, feed="sip")
    assert "SIP_" not in caplog.text
