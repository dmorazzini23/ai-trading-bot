from datetime import datetime, UTC
import json
import types

from ai_trading.data.fetch import sip_disallowed


class _RespOK:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    text = json.dumps({"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]})

    def json(self):
        return json.loads(self.text)


def test_sip_disallowed_falls_back_to_iex(monkeypatch):
    feeds: list[str] = []
    calls: list[str] = []

    def fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
        calls.append(params.get("feed"))
        return _RespOK()

    session = types.SimpleNamespace(get=fake_get)
    monkeypatch.setattr(sip_disallowed, "_ALLOW_SIP", False, raising=False)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    df = sip_disallowed.fetch_bars("AAPL", start, end, "1Min", session=session, feeds_used=feeds)
    assert calls == ["iex"]
    assert "sip" in feeds
    assert (getattr(df, "empty", False) is False) and df
