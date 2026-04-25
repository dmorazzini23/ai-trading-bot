from __future__ import annotations

from typing import Any

from ai_trading.data import fallback_cache


class _RaisesJson:
    def __init__(self, raw: Any) -> None:
        self.data = raw

    def json(self) -> Any:
        raise ValueError("bad json helper")


class _ReadResponse:
    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def read(self) -> Any:
        return self._raw


class _BodyResponse:
    def __init__(self, raw: Any) -> None:
        self.body = raw


class _BadAttributes:
    @property
    def data(self) -> Any:
        raise RuntimeError("bad data")

    @property
    def text(self) -> Any:
        raise RuntimeError("bad text")

    @property
    def content(self) -> Any:
        raise RuntimeError("bad content")

    def read(self) -> Any:
        raise RuntimeError("bad read")

    @property
    def body(self) -> Any:
        raise RuntimeError("bad body")

    def __str__(self) -> str:
        return '{"fallback": true}'


class _BadStr:
    def __str__(self) -> str:
        raise RuntimeError("bad str")


def test_resp_json_prefers_native_json_helper() -> None:
    class Response:
        def json(self) -> dict[str, bool]:
            return {"native": True}

    assert fallback_cache.resp_json(Response()) == {"native": True}


def test_resp_json_falls_back_to_data_text_content_and_bytes() -> None:
    assert fallback_cache.resp_json(_RaisesJson(b'{"data": 1}')) == {"data": 1}
    assert fallback_cache.resp_json(type("TextResponse", (), {"text": '{"text": 2}'})()) == {
        "text": 2
    }
    assert fallback_cache.resp_json(
        type("ContentResponse", (), {"content": bytearray(b'{"content": 3}')})()
    ) == {"content": 3}


def test_resp_json_reads_file_like_body_and_plain_string() -> None:
    assert fallback_cache.resp_json(_ReadResponse('{"read": 4}')) == {"read": 4}
    assert fallback_cache.resp_json(_BodyResponse(b'{"body": 5}')) == {"body": 5}
    assert fallback_cache.resp_json('{"plain": 6}') == {"plain": 6}


def test_resp_json_defensive_failure_paths() -> None:
    assert fallback_cache.resp_json(_BadAttributes()) == {"fallback": True}
    assert fallback_cache.resp_json(_BadStr()) == {}
    assert fallback_cache.resp_json("") == {}
    assert fallback_cache.resp_json("not json") == {}
    assert fallback_cache.parse_resp('{"alias": 1}') == {"alias": 1}
    assert fallback_cache.parse_json('{"alias": 2}') == {"alias": 2}
