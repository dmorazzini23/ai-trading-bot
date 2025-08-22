from ai_trading.utils import http as H


class DummyResp:
    status_code = 200
    content = b"ok"


def test_pool_config_defaults(monkeypatch):
    monkeypatch.delenv("HTTP_MAX_WORKERS", raising=False)
    monkeypatch.delenv("HTTP_MAX_PER_HOST", raising=False)
    st = H.pool_stats()
    assert st["workers"] == 8
    assert st["per_host"] == 6
    assert st["pool_maxsize"] >= st["workers"]


def test_host_semaphore_respects_env(monkeypatch):
    def fake_get(url, timeout=None, headers=None):
        return DummyResp()

    monkeypatch.setattr(H, "get", fake_get)
    monkeypatch.setenv("HTTP_MAX_PER_HOST", "3")
    _ = H.map_get(["https://example.com"])
    assert H.pool_stats()["per_host"] == 3
