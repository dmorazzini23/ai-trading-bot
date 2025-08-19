import json
from json import JSONDecodeError

from ai_trading.utils import http


class DummyResp:
    status_code = 200
    content = b"{}"

    def json(self):
        return json.loads(self.content)


def test_map_get_aggregates_decode_error(monkeypatch):
    def fake_get(url, timeout=None):
        if url == "bad":
            raise JSONDecodeError("bad", "doc", 0)
        return DummyResp()

    monkeypatch.setattr(http, "get", fake_get)
    urls = ["ok", "bad"]
    results = http.map_get(urls)
    good, error = results[0], results[1]
    assert good[0] == ("ok", 200, b"{}")
    assert error[0] is None
    assert isinstance(error[1], JSONDecodeError)
