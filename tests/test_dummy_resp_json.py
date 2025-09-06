from tests.helpers.dummy_http import DummyResp


def test_dummy_resp_json_returns_payload():
    resp = DummyResp({"foo": "bar"})
    assert resp.json() == {"foo": "bar"}
    assert resp.content == b'{"foo": "bar"}'


def test_dummy_resp_json_default_empty():
    assert DummyResp().json() == {}
